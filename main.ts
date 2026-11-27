// png-squeeze-01-run-20261127: squeeze 1280x720 RGBA screenshots into the smallest lossless PNG.
// One program, one run over the whole set (8 instances), 60 s wall on 1 CPU, 512 MB, no network.
// Strategy: decode each screenshot, screen colour forms (palette / grey / tRNS-key / RGB / RGBA)
// and per-row filters the way make.fun's scale compressor does, then write the winner with a
// fast optimal-ish deflate of our own (greedy lazy LZ77 + per-block optimal Huffman), falling
// back to zlib level 9 whenever it is smaller. Everything is deterministic: fixed work budgets,
// no clock, no randomness, so two runs write identical bytes.
// Node 22, standard library only.
import { inflateSync, deflateSync, constants } from 'node:zlib';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const WIDTH = 1280;
const HEIGHT = 720;

// --------------------------------------------------------------------------- PNG codec
const SIGNATURE = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();
function crc32(bytes, start = 0, end = bytes.length) {
  let c = -1;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function adler32(bytes) {
  let a = 1, b = 0;
  for (let i = 0; i < bytes.length;) {
    const end = Math.min(bytes.length, i + 5552);
    for (; i < end; i++) { a += bytes[i]; b += a; }
    a %= 65521; b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}
const ascii = (s) => Uint8Array.from(s, (ch) => ch.charCodeAt(0));
function pngOf(chunks) {
  let size = SIGNATURE.length;
  for (const [, data] of chunks) size += 12 + data.length;
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  out.set(SIGNATURE, 0);
  let at = SIGNATURE.length;
  for (const [type, data] of chunks) {
    view.setUint32(at, data.length);
    out.set(ascii(type), at + 4);
    out.set(data, at + 8);
    view.setUint32(at + 8 + data.length, crc32(out, at + 4, at + 8 + data.length));
    at += 12 + data.length;
  }
  return out;
}
function ihdrOf(width, height, colourType) {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, width); v.setUint32(4, height);
  d[8] = 8; d[9] = colourType;
  return d;
}
function paeth(a, b, c) {
  const p = a + b - c;
  const pa = p > a ? p - a : a - p;
  const pb = p > b ? p - b : b - p;
  const pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}
function filterRow(type, cur, prev, bpp, out) {
  const n = cur.length;
  if (type === 0) { out.set(cur); return; }
  if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (i >= bpp ? cur[i - bpp] : 0);
  } else if (type === 2) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (prev === null ? 0 : prev[i]);
  } else if (type === 3) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (((i >= bpp ? cur[i - bpp] : 0) + (prev === null ? 0 : prev[i])) >> 1);
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = cur[i] - paeth(a, b, c);
    }
  }
}
function signedSum(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) { const v = row[i]; s += v < 128 ? v : 256 - v; }
  return s;
}
function unfilterRow(type, row, prev, bpp, out) {
  const n = row.length;
  if (type === 0) { out.set(row); return; }
  if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = (row[i] + (i >= bpp ? out[i - bpp] : 0)) & 0xff;
  } else if (type === 2) {
    for (let i = 0; i < n; i++) out[i] = (row[i] + (prev === null ? 0 : prev[i])) & 0xff;
  } else if (type === 3) {
    for (let i = 0; i < n; i++) out[i] = (row[i] + (((i >= bpp ? out[i - bpp] : 0) + (prev === null ? 0 : prev[i])) >> 1)) & 0xff;
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? out[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = (row[i] + paeth(a, b, c)) & 0xff;
    }
  }
}
/** Decode any PNG the contest reads into RGBA (4 bytes a pixel), or throw. */
function decodePng(bytes) {
  if (bytes.length < 8 || SIGNATURE.some((b, i) => bytes[i] !== b)) throw new Error('not a PNG');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 8, header = null, palette = null, trns = null;
  const idat = [];
  let ended = false;
  while (at < bytes.length) {
    if (ended) throw new Error('bytes after IEND');
    const len = view.getUint32(at);
    const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
    if (crc32(bytes, at + 4, at + 8 + len) !== view.getUint32(at + 8 + len)) throw new Error(type + ' CRC');
    const data = bytes.subarray(at + 8, at + 8 + len);
    at += 12 + len;
    if (type === 'IHDR') {
      const dv = new DataView(data.buffer, data.byteOffset, 13);
      header = { width: dv.getUint32(0), height: dv.getUint32(4), depth: data[8], colourType: data[9], interlace: data[12] };
      if (header.depth !== 8 || !Object.hasOwn(CHANNELS, header.colourType) || header.interlace !== 0) throw new Error('unsupported IHDR');
    } else if (type === 'PLTE') { palette = data; }
    else if (type === 'tRNS') { trns = data; }
    else if (type === 'IDAT') { idat.push(data); }
    else if (type === 'IEND') { ended = true; }
    else throw new Error('unexpected chunk ' + type);
  }
  if (!header || idat.length === 0) throw new Error('missing IHDR/IDAT');
  const { width, height, colourType } = header;
  const ch = CHANNELS[colourType];
  const stride = width * ch;
  const raw = inflateSync(Buffer.concat(idat.map((d) => Buffer.from(d.buffer, d.byteOffset, d.byteLength))));
  if (raw.length !== height * (stride + 1)) throw new Error('inflated size mismatch');
  const samples = new Uint8Array(height * stride);
  const row = new Uint8Array(stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    unfilterRow(f, src, prev, ch, row);
    samples.set(row, y * stride);
  }
  const n = width * height;
  const rgba = new Uint8Array(n * 4);
  if (colourType === 6) return { width, height, rgba: samples };
  for (let i = 0, s = 0, d = 0; i < n; i++, d += 4) {
    if (colourType === 0) {
      rgba[d] = rgba[d + 1] = rgba[d + 2] = samples[s];
      rgba[d + 3] = trns ? (samples[s] === trns[1] ? 0 : 255) : 255;
      s += 1;
    } else if (colourType === 2) {
      rgba[d] = samples[s]; rgba[d + 1] = samples[s + 1]; rgba[d + 2] = samples[s + 2];
      rgba[d + 3] = trns && samples[s] === trns[1] && samples[s + 1] === trns[3] && samples[s + 2] === trns[5] ? 0 : 255;
      s += 3;
    } else if (colourType === 4) {
      rgba[d] = rgba[d + 1] = rgba[d + 2] = samples[s];
      rgba[d + 3] = samples[s + 1];
      s += 2;
    } else { // palette
      const idx = samples[s++];
      rgba[d] = palette[idx * 3]; rgba[d + 1] = palette[idx * 3 + 1]; rgba[d + 2] = palette[idx * 3 + 2];
      rgba[d + 3] = trns && idx < trns.length ? trns[idx] : 255;
    }
  }
  return { width, height, rgba };
}

// --------------------------------------------------------------------------- analysis + forms
function wordsOf(rgba) {
  const at = rgba.byteOffset % 4 === 0 ? rgba : rgba.slice();
  return new Uint32Array(at.buffer, at.byteOffset, at.length >>> 2);
}
function analyse(rgba) {
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  let opaque = true, grey = true;
  const counts = new Map();
  let many = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!many) {
      const c = u32[i];
      const k = counts.get(c);
      if (k === undefined) {
        if (counts.size === 256) many = true;
        else counts.set(c, 1);
      } else counts.set(c, k + 1);
    }
  }
  const colours = many ? null : [...counts.keys()].map((c) => {
    const b = new Uint8Array(Uint32Array.of(c).buffer);
    return { rgba: [b[0], b[1], b[2], b[3]], word: c, count: counts.get(c) };
  });
  return { opaque, grey, colours };
}
function transparentKey(rgba, grey) {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}
function samplesOf(rgba, ct) {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) { out[s] = rgba[d]; out[s + 1] = rgba[d + 3]; }
    else { out[s] = rgba[d]; out[s + 1] = rgba[d + 1]; out[s + 2] = rgba[d + 2]; }
  }
  return out;
}
// Hilbert-curve order of an RGB colour (0..7 per axis -> 9-bit index), for palette ordering.
function hilbert3(r, g, b) {
  let x = r >> 5, y = g >> 5, z = b >> 5, h = 0;
  for (let s = 4; s >= 0; s--) {
    const rx = (x >> s) & 1, ry = (y >> s) & 1, rz = (z >> s) & 1;
    h = (h << 3) | (rx * 4 + ry * 2 + rz);
    // rotate the cube so the curve stays continuous: swap x/y when rz==0, x/z when ry==0
    if (rz === 0) { if (ry === 1) { const t = x; x = y; y = t; } }
    else if (ry === 0) { const t = x; x = z; z = t; }
  }
  return h;
}
function paletteOrder(colours, order) {
  const lum = (c) => 299 * c.rgba[0] + 587 * c.rgba[1] + 114 * c.rgba[2];
  return [...colours].sort((x, y) => {
    const ta = x.rgba[3] === 255 ? 1 : 0;
    const tb = y.rgba[3] === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    if (order === 'hilbert') return hilbert3(x.rgba[0], x.rgba[1], x.rgba[2]) - hilbert3(y.rgba[0], y.rgba[1], y.rgba[2]) || x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
}
function paletteForm(rgba, colours, order) {
  const entries = paletteOrder(colours, order);
  const index = new Map(entries.map((c, i) => [c.word, i]));
  const plte = new Uint8Array(entries.length * 3);
  let transparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.rgba[0]; plte[i * 3 + 1] = c.rgba[1]; plte[i * 3 + 2] = c.rgba[2];
    if (c.rgba[3] !== 255) transparent = i + 1;
  });
  const trns = transparent === 0 ? null : Uint8Array.from(entries.slice(0, transparent), (c) => c.rgba[3]);
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  const indices = new Uint8Array(n);
  for (let i = 0; i < n; i++) indices[i] = index.get(u32[i]);
  return { plte, trns, indices };
}
function formsOf(rgba) {
  const { opaque, grey, colours } = analyse(rgba);
  const forms = [];
  if (colours !== null) {
    for (const order of ['count', 'luminance', 'hilbert']) {
      const p = paletteForm(rgba, colours, order);
      forms.push({ colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, filters: order === 'count' ? [0] : [0, 'minsum', 'brute'] });
    }
    if (!grey) return forms;
  }
  const key = opaque ? null : transparentKey(rgba, grey);
  let ct, trns = null;
  if (grey) {
    ct = opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    ct = opaque || key !== null ? 2 : 6;
    if (key !== null) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[ct];
  forms.push({ colourType: ct, bpp, samples: samplesOf(rgba, ct), plte: null, trns, filters: [0, 1, 2, 4, 'minsum', 'pairs', 'brute'] });
  return forms;
}
const chunkBytes = (form) => (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);

// --------------------------------------------------------------------------- filter screening
const BRUTE_CONTEXT = 32768;
const BRUTE_LEVEL = 4;
function pairsOf(row, seen, stamp) {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const pair = (row[i - 1] << 8) | row[i];
    if (seen[pair] !== stamp) { seen[pair] = stamp; n++; }
  }
  return n;
}
/** Filtered data for one (samples, choice); choice is a fixed type 0-4, 'minsum', 'pairs' or 'brute'. */
function filtered(samples, width, height, bpp, choice) {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial = [0, 1, 2, 3, 4].map((t) => { const r = new Uint8Array(stride + 1); r[0] = t; return r; });
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    if (typeof choice === 'number') {
      raw[at] = choice;
      filterRow(choice, cur, prev, bpp, raw.subarray(at + 1, at + 1 + stride));
      continue;
    }
    const before = raw.subarray(Math.max(0, at - BRUTE_CONTEXT), at);
    let type = 0, best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t].subarray(1));
      let cost;
      if (choice === 'minsum') cost = signedSum(trial[t].subarray(1));
      else if (choice === 'pairs') cost = pairsOf(trial[t].subarray(1), seen, ++stamp);
      else cost = deflateSync(trial[t], before.length === 0 ? { level: BRUTE_LEVEL, memLevel: 9 } : { level: BRUTE_LEVEL, memLevel: 9, dictionary: before }).length;
      if (cost < best) { best = cost; type = t; }
    }
    raw.set(trial[type], at);
  }
  return raw;
}
const SCREEN = { level: 6, memLevel: 8 };
/** Best (form, choice) by zlib-6 stream + chunk bytes; returns {form, choice, data, bytes}. */
function screen(rgba, width, height) {
  let best = null;
  for (const form of formsOf(rgba)) {
    const chunks = chunkBytes(form);
    for (const choice of form.filters) {
      const data = filtered(form.samples, width, height, form.bpp, choice);
      const bytes = deflateSync(data, SCREEN).length + chunks;
      if (best === null || bytes < best.bytes) best = { form, choice, data, bytes };
    }
  }
  return best;
}
function pngOfForm(width, height, form, idat) {
  const chunks = [['IHDR', ihdrOf(width, height, form.colourType)]];
  if (form.plte !== null) chunks.push(['PLTE', form.plte]);
  if (form.trns !== null) chunks.push(['tRNS', form.trns]);
  chunks.push(['IDAT', idat]);
  chunks.push(['IEND', new Uint8Array(0)]);
  return pngOf(chunks);
}

// --------------------------------------------------------------------------- deflate (custom, zopfli-lite)
// Length / distance code tables (RFC 1951).
const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const LEN_SYM = new Uint16Array(259); // length -> symbol 257..285 (needs >255!)
const LEN_EXT = new Uint8Array(259);
for (let s = 0; s < 29; s++) {
  const base = LEN_BASE[s];
  const top = s === 28 ? 258 : LEN_BASE[s + 1] - 1;
  for (let l = base; l <= top; l++) { LEN_SYM[l] = 257 + s; LEN_EXT[l] = LEN_EXTRA[s]; }
}
const DIST_SYM = new Uint8Array(32769); // distance -> symbol 0..29
const DIST_EXT = new Uint8Array(32769);
for (let s = 0; s < 30; s++) {
  const base = DIST_BASE[s];
  const top = s === 29 ? 32768 : DIST_BASE[s + 1] - 1;
  for (let d = base; d <= top; d++) { DIST_SYM[d] = s; DIST_EXT[d] = DIST_EXTRA[s]; }
}
// Fixed Huffman lengths.
const FIXED_LIT_LEN = new Uint8Array(288);
for (let i = 0; i < 288; i++) FIXED_LIT_LEN[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
const FIXED_DIST_LEN = new Uint8Array(30).fill(5);

/** Length-limited Huffman code lengths by package-merge (freq: array, maxBits: limit). */
function huffmanLengths(freq, maxBits) {
  const n = freq.length;
  const len = new Array(n).fill(0);
  const syms = [];
  for (let i = 0; i < n; i++) if (freq[i] > 0) syms.push(i);
  if (syms.length === 0) return len;
  if (syms.length === 1) { len[syms[0]] = 1; return len; }
  const limit = 2 * syms.length - 2;
  const singles = syms.map((s) => ({ w: freq[s], m: [s] })).sort((a, b) => a.w - b.w);
  const lists = [singles];
  for (let d = 1; d < maxBits; d++) {
    const prev = lists[d - 1];
    const packaged = [];
    for (let i = 0; i + 1 < prev.length; i += 2) {
      packaged.push({ w: prev[i].w + prev[i + 1].w, m: prev[i].m.concat(prev[i + 1].m) });
    }
    const merged = [];
    let a = 0, b = 0;
    while (merged.length < limit && (a < packaged.length || b < singles.length)) {
      if (b >= singles.length || (a < packaged.length && packaged[a].w < singles[b].w)) merged.push(packaged[a++]);
      else merged.push(singles[b++]);
    }
    lists.push(merged);
  }
  for (const pkg of lists[maxBits - 1]) for (const s of pkg.m) len[s]++;
  return len;
}
/** Canonical Huffman codes from lengths: returns {codes, lengths} with codes[s] the code value. */
function canonicalCodes(lengths) {
  const maxLen = Math.max(0, ...lengths);
  const blCount = new Array(maxLen + 1).fill(0);
  for (const l of lengths) if (l > 0) blCount[l]++;
  const nextCode = new Array(maxLen + 1).fill(0);
  let code = 0;
  for (let b = 1; b <= maxLen; b++) { code = (code + blCount[b - 1]) << 1; nextCode[b] = code; }
  const codes = new Array(lengths.length).fill(0);
  for (let s = 0; s < lengths.length; s++) {
    const l = lengths[s];
    if (l > 0) codes[s] = nextCode[l]++;
  }
  return { codes, lengths };
}
/** Bit writer: bits go LSB-first into bytes; Huffman codes are fed MSB-first via writeCode. */
class BitWriter {
  constructor(capacity) {
    this.buf = new Uint8Array(Math.max(1024, capacity));
    this.pos = 0; // byte position
    this.bit = 0; // bits used in current byte
  }
  ensure(n) {
    if (this.pos + n >= this.buf.length) {
      const nb = new Uint8Array(Math.max(this.buf.length * 2, this.pos + n + 1024));
      nb.set(this.buf.subarray(0, this.pos + 1));
      this.buf = nb;
    }
  }
  writeBits(value, count) {
    this.ensure(4);
    for (let i = 0; i < count; i++) {
      if ((value >> i) & 1) this.buf[this.pos] |= 1 << this.bit;
      if (++this.bit === 8) { this.bit = 0; this.pos++; this.ensure(4); }
    }
  }
  writeCode(code, len) {
    for (let i = len - 1; i >= 0; i--) {
      if ((code >> i) & 1) this.buf[this.pos] |= 1 << this.bit;
      if (++this.bit === 8) { this.bit = 0; this.pos++; this.ensure(4); }
    }
  }
  finish() {
    return this.buf.subarray(0, this.pos + (this.bit > 0 ? 1 : 0));
  }
}
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
/** RLE-encode a code-length sequence (values 0..15) into [symbol, extraValue, extraBits] triples. */
function rleLengths(lengths) {
  const out = [];
  let i = 0;
  while (i < lengths.length) {
    const v = lengths[i];
    let run = 1;
    while (i + run < lengths.length && lengths[i + run] === v) run++;
    const total = run;
    if (v === 0) {
      while (run >= 11) { const take = Math.min(run, 138); out.push([18, take - 11, 7]); run -= take; }
      if (run >= 3) { out.push([17, run - 3, 3]); run = 0; }
      while (run-- > 0) out.push([0, 0, 0]);
    } else {
      out.push([v, 0, 0]);
      run--;
      while (run >= 3) { const take = Math.min(run, 6); out.push([16, take - 3, 2]); run -= take; }
      while (run-- > 0) out.push([v, 0, 0]);
    }
    i += total;
  }
  return out;
}

// --------------------------------------------------------------------------- LZ77 tokeniser
// Deep hash-chain search (zlib-9 uses 4096) with a quick reject: a candidate that
// cannot beat the current best length is skipped without a full matchLen call.
// Lazy matching: defer to the match at p+1 when it is longer, or as long but nearer.
const MAX_CHAIN = 4096;
const GOOD_LEN = 128;
const MAX_MATCH = 258;
const MIN_MATCH = 3;
const WINDOW = 32768;
const BLOCK_INPUT = 524288; // input bytes per deflate block

function hash3(d, p) {
  return ((((d[p] << 16) ^ (d[p + 1] << 8) ^ d[p + 2]) * 2654435761) >>> 17) & 0x7fff;
}
/** Length of the match between data[a..] and data[b..], at most maxLen (word-compared). */
function matchLen(w32, data, a, b, maxLen) {
  let len = 0;
  if ((a & 3) === 0 && (b & 3) === 0) {
    const words = maxLen >>> 2;
    let aw = a >>> 2, bw = b >>> 2, w = 0;
    while (w < words && w32[aw] === w32[bw]) { aw++; bw++; w++; }
    len = w * 4;
  }
  while (len < maxLen && data[a + len] === data[b + len]) len++;
  return len;
}
/**
 * Tokenise `data` into a packed Int32Array: literal byte (0..255), or match
 * 256 + (len-3)*32768 + (dist-1). Returns {toks, count}.
 * Lazy matching with hash chains; fixed work budget per position (no clock).
 */
function tokenise(data) {
  const n = data.length;
  const toks = new Int32Array(n + 1);
  let count = 0;
  if (n === 0) return { toks, count };
  const head = new Int32Array(32768).fill(-1);
  const prev = new Int32Array(n).fill(-1);
  const w32 = new Uint32Array(data.buffer, data.byteOffset, data.length >>> 2);
  const insert = (p) => {
    if (p + 2 >= n) return;
    const h = hash3(data, p);
    prev[p] = head[h];
    head[h] = p;
  };
  const findBest = (p) => {
    if (p + MIN_MATCH > n) return null;
    const h = hash3(data, p);
    const maxLen = Math.min(MAX_MATCH, n - p);
    let bestLen = 0, bestDist = 0;
    let q = head[h];
    let steps = MAX_CHAIN;
    while (q >= 0 && steps-- > 0) {
      const dist = p - q;
      if (dist > WINDOW) break;
      if (bestLen >= MIN_MATCH && (q + bestLen >= n || data[q + bestLen] !== data[p + bestLen])) { q = prev[q]; continue; }
      const len = matchLen(w32, data, p, q, maxLen);
      if (len > bestLen || (len === bestLen && dist < bestDist)) {
        bestLen = len; bestDist = dist;
        if (len >= MAX_MATCH) break;
        if (len >= GOOD_LEN) steps = Math.min(steps, 32);
      }
      q = prev[q];
    }
    return bestLen >= MIN_MATCH ? { len: bestLen, dist: bestDist } : null;
  };
  let p = 0;
  while (p < n) {
    const b1 = findBest(p);
    insert(p);
    if (b1 === null) { toks[count++] = data[p++]; continue; }
    // lazy: is the match at p+1 better?
    let takeLiteral = false;
    if (b1.len <= GOOD_LEN && p + 1 < n) {
      const b2 = findBest(p + 1);
      if (b2 !== null && (b2.len > b1.len || (b2.len === b1.len && b2.dist < b1.dist))) takeLiteral = true;
    }
    if (takeLiteral) { toks[count++] = data[p++]; continue; }
    const L = b1.len;
    toks[count++] = 256 + (L - 3) * 32768 + (b1.dist - 1);
    for (let k = 1; k < L; k++) insert(p + k);
    p += L;
  }
  return { toks, count };
}

// --------------------------------------------------------------------------- block encoder
/** Frequencies of a token range [t0, t1): returns {litFreq(288), distFreq(30), inputBytes}. */
function blockFreq(toks, t0, t1) {
  const litFreq = new Array(288).fill(0);
  const distFreq = new Array(30).fill(0);
  let inputBytes = 0;
  for (let t = t0; t < t1; t++) {
    const v = toks[t];
    if (v < 256) { litFreq[v]++; inputBytes++; }
    else {
      const x = v - 256;
      const len = (x / 32768 | 0) + 3;
      const dist = (x % 32768) + 1;
      litFreq[LEN_SYM[len]]++;
      distFreq[DIST_SYM[dist]]++;
      inputBytes += len;
    }
  }
  litFreq[256]++; // end of block
  return { litFreq, distFreq, inputBytes };
}
/** Bit cost of a token range under given code lengths (no header). */
function dataBits(toks, t0, t1, litLen, distLen) {
  let bits = 0;
  for (let t = t0; t < t1; t++) {
    const v = toks[t];
    if (v < 256) { bits += litLen[v]; continue; }
    const x = v - 256;
    const len = (x / 32768 | 0) + 3;
    const dist = (x % 32768) + 1;
    bits += litLen[LEN_SYM[len]] + LEN_EXT[len] + distLen[DIST_SYM[dist]] + DIST_EXT[dist];
  }
  bits += litLen[256];
  return bits;
}
/** Dynamic-block header bit cost for given lit/dist lengths (incl. the 3-bit block header). */
function dynamicHeaderBits(litLen, distLen) {
  let hlit = 257;
  for (let i = 287; i >= 257; i--) if (litLen[i] > 0) { hlit = i + 1; break; }
  let hdist = 1;
  for (let i = 29; i >= 1; i--) if (distLen[i] > 0) { hdist = i + 1; break; }
  const seq = litLen.slice(0, hlit).concat(distLen.slice(0, hdist));
  const rle = rleLengths(seq);
  const clFreq = new Array(19).fill(0);
  for (const [s] of rle) clFreq[s]++;
  const clLen = huffmanLengths(clFreq, 7);
  let hclen = 4;
  for (let i = 18; i >= 4; i--) if (clLen[CL_ORDER[i]] > 0) { hclen = i + 1; break; }
  let bits = 3 + 5 + 5 + 4 + 3 * hclen;
  for (const [s, extra, ebits] of rle) bits += clLen[s] + ebits;
  return { bits, hlit, hdist, hclen, clLen, rle };
}
/** Write one dynamic block [t0,t1) of tokens; isLast marks BFINAL. */
function writeDynamicBlock(bw, toks, t0, t1, isLast) {
  const { litFreq, distFreq } = blockFreq(toks, t0, t1);
  const litLen = huffmanLengths(litFreq, 15);
  const hasMatch = distFreq.some((f) => f > 0);
  const distLen = hasMatch ? huffmanLengths(distFreq, 15) : new Array(30).fill(0);
  if (!hasMatch) distLen[0] = 1;
  const { bits, hlit, hdist, hclen, clLen, rle } = dynamicHeaderBits(litLen, distLen);
  const litCode = canonicalCodes(litLen);
  const distCode = canonicalCodes(distLen);
  const clCode = canonicalCodes(clLen);
  bw.writeBits(isLast ? 1 : 0, 1);
  bw.writeBits(2, 2); // dynamic
  bw.writeBits(hlit - 257, 5);
  bw.writeBits(hdist - 1, 5);
  bw.writeBits(hclen - 4, 4);
  for (let i = 0; i < hclen; i++) bw.writeBits(clLen[CL_ORDER[i]], 3);
  for (const [s, extra, ebits] of rle) {
    bw.writeCode(clCode.codes[s], clLen[s]);
    if (ebits > 0) bw.writeBits(extra, ebits);
  }
  for (let t = t0; t < t1; t++) {
    const v = toks[t];
    if (v < 256) { bw.writeCode(litCode.codes[v], litLen[v]); continue; }
    const x = v - 256;
    const len = (x / 32768 | 0) + 3;
    const dist = (x % 32768) + 1;
    const ls = LEN_SYM[len];
    bw.writeCode(litCode.codes[ls], litLen[ls]);
    if (LEN_EXT[len] > 0) bw.writeBits(len - LEN_BASE[ls - 257], LEN_EXT[len]);
    const ds = DIST_SYM[dist];
    bw.writeCode(distCode.codes[ds], distLen[ds]);
    if (DIST_EXT[dist] > 0) bw.writeBits(dist - DIST_BASE[ds], DIST_EXT[dist]);
  }
  bw.writeCode(litCode.codes[256], litLen[256]);
}
/** Write one fixed-Huffman block [t0,t1). */
function writeFixedBlock(bw, toks, t0, t1, isLast) {
  const litCode = canonicalCodes(FIXED_LIT_LEN);
  const distCode = canonicalCodes(FIXED_DIST_LEN);
  bw.writeBits(isLast ? 1 : 0, 1);
  bw.writeBits(1, 2); // fixed
  for (let t = t0; t < t1; t++) {
    const v = toks[t];
    if (v < 256) { bw.writeCode(litCode.codes[v], FIXED_LIT_LEN[v]); continue; }
    const x = v - 256;
    const len = (x / 32768 | 0) + 3;
    const dist = (x % 32768) + 1;
    const ls = LEN_SYM[len];
    bw.writeCode(litCode.codes[ls], FIXED_LIT_LEN[ls]);
    if (LEN_EXT[len] > 0) bw.writeBits(len - LEN_BASE[ls - 257], LEN_EXT[len]);
    const ds = DIST_SYM[dist];
    bw.writeCode(distCode.codes[ds], FIXED_DIST_LEN[ds]);
    if (DIST_EXT[dist] > 0) bw.writeBits(dist - DIST_BASE[ds], DIST_EXT[dist]);
  }
  bw.writeCode(litCode.codes[256], FIXED_LIT_LEN[256]);
}
/** Encode `data` as one zlib stream with our deflate; returns Uint8Array (raw deflate + wrapper). */
function deflateCustom(data) {
  const { toks, count } = tokenise(data);
  // split into blocks at token boundaries, each covering at most BLOCK_INPUT input bytes
  const bounds = [0];
  let inputBytes = 0, lastCut = 0;
  for (let t = 0; t < count; t++) {
    const v = toks[t];
    inputBytes += v < 256 ? 1 : ((v - 256) / 32768 | 0) + 3;
    if (inputBytes - lastCut >= BLOCK_INPUT && t + 1 < count) { bounds.push(t + 1); lastCut = inputBytes; }
  }
  bounds.push(count);
  const bw = new BitWriter(data.length);
  for (let b = 0; b < bounds.length - 1; b++) {
    const t0 = bounds[b], t1 = bounds[b + 1];
    const isLast = b === bounds.length - 2;
    const { litFreq, distFreq, inputBytes: rawLen } = blockFreq(toks, t0, t1);
    const litLen = huffmanLengths(litFreq, 15);
    const hasMatch = distFreq.some((f) => f > 0);
    const distLen = hasMatch ? huffmanLengths(distFreq, 15) : new Array(30).fill(0);
    if (!hasMatch) distLen[0] = 1;
    const dynBits = dynamicHeaderBits(litLen, distLen).bits + dataBits(toks, t0, t1, litLen, distLen);
    const fixBits = 3 + dataBits(toks, t0, t1, FIXED_LIT_LEN, FIXED_DIST_LEN);
    // stored blocks never win on this data (filtered screenshots compress well);
    // the choice is dynamic vs fixed only.
    void rawLen;
    if (dynBits <= fixBits) writeDynamicBlock(bw, toks, t0, t1, isLast);
    else writeFixedBlock(bw, toks, t0, t1, isLast);
  }
  const raw = bw.finish();
  const out = new Uint8Array(2 + raw.length + 4);
  out[0] = 0x78; out[1] = 0x9c;
  out.set(raw, 2);
  new DataView(out.buffer).setUint32(2 + raw.length, adler32(data));
  return out;
}

// --------------------------------------------------------------------------- main
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
// Per-instance deflate candidates (all deterministic): our custom deflate, plus zlib
// level 9 at memLevel 9, memLevel 8, and memLevel 8 with Z_FILTERED — the three
// zlib settings differ by library heuristics and each wins on some instances
// (measured on the practice set: memLevel 8 beats memLevel 9 on most, FILTERED
// wins on the rich i006). Take the smallest stream per instance.
const Z_CANDIDATES = [
  { level: 9, memLevel: 9 },
  { level: 9, memLevel: 8 },
  { level: 9, memLevel: 8, strategy: constants.Z_FILTERED },
];
for (const id of readdirSync(I).sort()) {
  const image = decodePng(new Uint8Array(readFileSync(join(I, id, 'image.png'))));
  if (image.width !== WIDTH || image.height !== HEIGHT) throw new Error(id + ': not 1280x720');
  const best = screen(image.rgba, image.width, image.height);
  let idat = null;
  for (const opts of Z_CANDIDATES) {
    const z = deflateSync(best.data, opts);
    if (idat === null || z.length < idat.length) idat = z;
  }
  try {
    const custom = deflateCustom(best.data);
    if (custom.length < idat.length) idat = custom;
  } catch (e) {
    // custom deflate must never sink the entry: fall back to the best zlib stream
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'image.png'), pngOfForm(image.width, image.height, best.form, new Uint8Array(idat)));
}
