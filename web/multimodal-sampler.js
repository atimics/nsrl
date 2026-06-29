const MODEL_MAGIC = "NSRLMOD1";
const MODEL_VERSION = 1;
const BOS = 1;
const PROMPT = 2;
const TEXT = 3;
const IMAGE = 4;
const EOS = 5;
const TEXT_BASE = 16;
const TEXT_COUNT = 128;
const IMAGE_BASE = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS = 16;
const SIGNATURE_GRID = 16;
const SIGNATURE_BINS = SIGNATURE_GRID * SIGNATURE_GRID;
const VOCAB_SIZE = IMAGE_BASE + IMAGE_BINS;
const MAX_CONTEXT_TOKENS = 64;
const CONTEXT_LENGTHS = [1, 2, 4, 8, 16, 32, 64];
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

export class SolomonMultimodalSampler {
  constructor(modelBytes) {
    const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
    const expectedHash = readU64AtEnd(bytes);
    const actualHash = hashBytes(bytes.subarray(0, bytes.length - 8));
    if (expectedHash !== actualHash) {
      throw new Error(
        `NSRLMOD1 hash mismatch: expected ${hex64(expectedHash)}, got ${hex64(actualHash)}`,
      );
    }
    const reader = new BinaryReader(bytes.subarray(0, bytes.length - 8));
    const magic = reader.readAscii(8);
    if (magic !== MODEL_MAGIC) {
      throw new Error("not an NSRLMOD1 model");
    }
    const version = reader.readU32();
    if (version !== MODEL_VERSION) {
      throw new Error(`unsupported NSRLMOD1 version ${version}`);
    }
    expectU32(reader, VOCAB_SIZE, "vocab size");
    expectU32(reader, TEXT_BASE, "text base");
    expectU32(reader, TEXT_COUNT, "text count");
    expectU32(reader, IMAGE_BASE, "image base");
    expectU32(reader, IMAGE_BINS, "image bins");
    expectU32(reader, SIGNATURE_GRID, "signature grid");
    this.tokenCount = Number(reader.readU64());
    this.tokenHash = reader.readU64();
    this.unigramTotal = reader.readU32();
    this.unigram = sortCounts(reader.readCountList());
    const maxContext = reader.readU32();
    if (maxContext !== MAX_CONTEXT_TOKENS) {
      throw new Error(`NSRLMOD1 max context mismatch: expected ${MAX_CONTEXT_TOKENS}, got ${maxContext}`);
    }
    const contextRows = reader.readU32();
    this.contexts = new Map();
    for (let row = 0; row < contextRows; row += 1) {
      const contextLength = reader.readU32();
      if (contextLength === 0 || contextLength > MAX_CONTEXT_TOKENS) {
        throw new Error("NSRLMOD1 context row has invalid length");
      }
      const context = [];
      for (let index = 0; index < contextLength; index += 1) {
        context.push(readToken(reader));
      }
      const total = reader.readU32();
      const next = sortCounts(reader.readCountList());
      this.contexts.set(contextKey(context), { total, next });
    }
    if (!reader.isDone()) {
      throw new Error("trailing bytes in NSRLMOD1 model");
    }
    this.modelHash = actualHash;
  }

  sample(prompt, options = {}) {
    const normalizedPrompt = normalizeText(prompt || "king solomon seal");
    const topK = Math.max(1, Number(options.topK || 1));
    const maxTextTokens = Math.max(1, Number(options.maxTextTokens || 320));
    const seed = (Number(options.seed || 1) >>> 0) ^ hashText32(normalizedPrompt);
    const tokens = [BOS, PROMPT, ...encodeTextTokens(normalizedPrompt), TEXT];
    const textTokens = [];

    for (let step = 0; step < maxTextTokens; step += 1) {
      const token = this.nextToken(tokens, "text", seed, step, topK);
      if (token === IMAGE || token === EOS) {
        break;
      }
      if (isTextToken(token)) {
        tokens.push(token);
        textTokens.push(token);
      }
    }
    tokens.push(IMAGE);

    const imageBins = new Uint8Array(SIGNATURE_BINS);
    for (let index = 0; index < imageBins.length; index += 1) {
      const token = this.nextToken(tokens, "image", seed, maxTextTokens + index, topK);
      const imageToken = isImageToken(token) ? token : this.defaultImageToken();
      imageBins[index] = imageToken - IMAGE_BASE;
      tokens.push(imageToken);
    }
    tokens.push(EOS);

    const width = 192;
    const height = 192;
    const rgba = renderImageBins(imageBins, width, height);
    return {
      width,
      height,
      rgba,
      text: decodeTextTokens(textTokens),
      metadata: {
        model_kind: "NSRLMOD1",
        model_hash: hex64(this.modelHash),
        token_hash: hex64(this.tokenHash),
        token_count: this.tokenCount,
        image_grid: SIGNATURE_GRID,
      },
    };
  }

  nextToken(history, phase, seed, step, topK) {
    const lengths = contextLengthsForPosition(history.length).sort((left, right) => right - left);
    for (const length of lengths) {
      const row = this.contexts.get(contextKey(history.slice(history.length - length)));
      if (!row) {
        continue;
      }
      const token = chooseFromCounts(row.next, phase, seed, step, topK);
      if (token != null) {
        return token;
      }
    }
    return chooseFromCounts(this.unigram, phase, seed, step, topK) ?? (phase === "text" ? IMAGE : IMAGE_BASE);
  }

  defaultImageToken() {
    return chooseFromCounts(this.unigram, "image", 0, 0, 1) ?? IMAGE_BASE;
  }
}

class BinaryReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.offset = 0;
  }

  isDone() {
    return this.offset === this.bytes.length;
  }

  readAscii(count) {
    const bytes = this.readBytes(count);
    return String.fromCharCode(...bytes);
  }

  readU32() {
    this.require(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readU64() {
    this.require(8);
    const lo = BigInt(this.view.getUint32(this.offset, true));
    const hi = BigInt(this.view.getUint32(this.offset + 4, true));
    this.offset += 8;
    return (hi << 32n) | lo;
  }

  readCountList() {
    const count = this.readU32();
    const out = [];
    for (let index = 0; index < count; index += 1) {
      out.push({ token: readToken(this), count: this.readU32() });
    }
    return out;
  }

  readBytes(count) {
    this.require(count);
    const bytes = this.bytes.subarray(this.offset, this.offset + count);
    this.offset += count;
    return bytes;
  }

  require(count) {
    if (this.offset + count > this.bytes.length) {
      throw new Error("truncated NSRLMOD1 model");
    }
  }
}

function readToken(reader) {
  const token = reader.readU32();
  if (token >= VOCAB_SIZE) {
    throw new Error(`token ${token} is outside NSRLMOD1 vocab`);
  }
  return token;
}

function expectU32(reader, expected, label) {
  const actual = reader.readU32();
  if (actual !== expected) {
    throw new Error(`NSRLMOD1 ${label} mismatch: expected ${expected}, got ${actual}`);
  }
}

function sortCounts(counts) {
  return counts.sort((left, right) => right.count - left.count || left.token - right.token);
}

function contextLengthsForPosition(position) {
  const lengths = [];
  if (position > 0 && position <= MAX_CONTEXT_TOKENS) {
    lengths.push(position);
  }
  for (const length of CONTEXT_LENGTHS) {
    if (length <= position && !lengths.includes(length)) {
      lengths.push(length);
    }
  }
  return lengths;
}

function contextKey(tokens) {
  return tokens.join(",");
}

function chooseFromCounts(counts, phase, seed, step, topK) {
  const candidates = counts.filter((entry) => allowedToken(entry.token, phase));
  if (candidates.length === 0) {
    return null;
  }
  const limit = Math.max(1, Math.min(topK || 1, candidates.length));
  if (limit === 1) {
    return candidates[0].token;
  }
  const total = candidates.slice(0, limit).reduce((sum, entry) => sum + entry.count, 0);
  if (total <= 0) {
    return candidates[0].token;
  }
  let draw = mix32((seed ^ step) >>> 0) % total;
  for (let index = 0; index < limit; index += 1) {
    const entry = candidates[index];
    if (draw < entry.count) {
      return entry.token;
    }
    draw -= entry.count;
  }
  return candidates[0].token;
}

function allowedToken(token, phase) {
  if (phase === "text") {
    return isTextToken(token) || token === IMAGE || token === EOS;
  }
  return isImageToken(token);
}

function isTextToken(token) {
  return token >= TEXT_BASE && token < TEXT_BASE + TEXT_COUNT;
}

function isImageToken(token) {
  return token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS;
}

function encodeTextTokens(text) {
  return Array.from(normalizeText(text), (ch) => TEXT_BASE + Math.min(127, ch.charCodeAt(0)));
}

function decodeTextTokens(tokens) {
  return normalizeText(
    tokens
      .filter(isTextToken)
      .map((token) => String.fromCharCode(token - TEXT_BASE))
      .join(""),
  );
}

function normalizeText(value) {
  return String(value || "")
    .replace(/[^\x20-\x7e]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function renderImageBins(imageBins, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const paper = [239, 232, 210];
  const ink = [18, 18, 15];
  const scaleX = Math.max(1, Math.floor(width / SIGNATURE_GRID));
  const scaleY = Math.max(1, Math.floor(height / SIGNATURE_GRID));
  for (let y = 0; y < height; y += 1) {
    const gy = Math.min(SIGNATURE_GRID - 1, Math.floor(y / scaleY));
    for (let x = 0; x < width; x += 1) {
      const gx = Math.min(SIGNATURE_GRID - 1, Math.floor(x / scaleX));
      const alpha = imageBins[gy * SIGNATURE_GRID + gx] * 17;
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        rgba[offset + channel] = Math.floor((paper[channel] * (255 - alpha) + ink[channel] * alpha) / 255);
      }
      rgba[offset + 3] = 255;
    }
  }
  return rgba;
}

function readU64AtEnd(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const offset = bytes.byteLength - 8;
  const lo = BigInt(view.getUint32(offset, true));
  const hi = BigInt(view.getUint32(offset + 4, true));
  return (hi << 32n) | lo;
}

function hashBytes(bytes) {
  let hash = FNV_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = (hash * FNV_PRIME) & FNV_MASK;
  }
  return hash;
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function hashText32(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mix32(value) {
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}
