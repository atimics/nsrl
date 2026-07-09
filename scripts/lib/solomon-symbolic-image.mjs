const DEFAULT_GRID = 16;
const DEFAULT_IMAGE_BASE = 144;
const DEFAULT_IMAGE_BINS = 16;
const DEFAULT_INK_THRESHOLD = 64;
const DEFAULT_CHANNEL_TOKENS = {
  ink: 11,
  edge: 12,
  component: 13,
  radial: 14,
  direction: 15,
};
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

function imageConfig(options = {}) {
  return {
    grid: Number(options.grid || DEFAULT_GRID),
    imageBase: Number(options.imageBase || DEFAULT_IMAGE_BASE),
    imageBins: Number(options.imageBins || DEFAULT_IMAGE_BINS),
    inkThreshold: Number(options.inkThreshold || DEFAULT_INK_THRESHOLD),
    channelTokens: {
      ...DEFAULT_CHANNEL_TOKENS,
      ...(options.channelTokens || {}),
    },
  };
}

export function imageTokenChannels(profile) {
  if (profile === "ink16") {
    return ["ink"];
  }
  if (profile === "ink-edge16") {
    return ["ink", "edge"];
  }
  if (profile === "symbolic16") {
    return ["ink", "edge", "component", "radial", "direction"];
  }
  throw new Error(`unknown image token profile: ${profile}`);
}

export function imageTaskTokens(signature, profile = "symbolic16", options = {}) {
  const config = imageConfig(options);
  const channels = imageTokenChannels(profile);
  if (channels.length === 1 && channels[0] === "ink") {
    return imageTokens(signature, config);
  }
  return channels.flatMap((channel) => [
    config.channelTokens[channel],
    ...imageChannelTokens(signature, channel, config),
  ]);
}

export function symbolicImageTokens(signature, options = {}) {
  return imageTaskTokens(signature, "symbolic16", options);
}

export function imageTokenChannelStats(signatures, profile = "symbolic16", options = {}) {
  const config = imageConfig(options);
  const stats = {};
  for (const channel of imageTokenChannels(profile)) {
    let records = 0;
    let totalTokens = 0;
    let nonzeroTokens = 0;
    let activeRecords = 0;
    let multiBinRecords = 0;
    let minBin = config.imageBins;
    let maxBin = 0;
    const tokenLengths = new Set();
    const bins = new Set();
    const recordHashes = new Set();
    for (const signature of signatures) {
      const tokens = imageChannelTokens(signature, channel, config);
      records += 1;
      totalTokens += tokens.length;
      tokenLengths.add(tokens.length);
      let rowNonzeroTokens = 0;
      const rowBins = new Set();
      for (const token of tokens) {
        const bin = token - config.imageBase;
        if (bin < 0 || bin >= config.imageBins) {
          throw new Error(`image channel ${channel} produced out-of-range token ${token}`);
        }
        bins.add(bin);
        rowBins.add(bin);
        minBin = Math.min(minBin, bin);
        maxBin = Math.max(maxBin, bin);
        if (bin > 0) {
          nonzeroTokens += 1;
          rowNonzeroTokens += 1;
        }
      }
      if (rowNonzeroTokens > 0) {
        activeRecords += 1;
      }
      if (rowBins.size > 1) {
        multiBinRecords += 1;
      }
      recordHashes.add(fnv64Hex(tokens));
    }
    const lengths = [...tokenLengths].sort((left, right) => left - right);
    stats[channel] = {
      records,
      tokens_per_record: lengths.length === 1 ? lengths[0] : null,
      token_lengths: lengths,
      total_tokens: totalTokens,
      nonzero_tokens: nonzeroTokens,
      active_records: activeRecords,
      multi_bin_records: multiBinRecords,
      distinct_bins: bins.size,
      min_bin: bins.size > 0 ? minBin : 0,
      max_bin: bins.size > 0 ? maxBin : 0,
      unique_record_hashes: recordHashes.size,
      duplicate_record_hashes: records - recordHashes.size,
    };
  }
  return stats;
}

export function imageChannelTokens(signature, channel, options = {}) {
  const config = imageConfig(options);
  if (channel === "ink") {
    return imageTokens(signature, config);
  }
  if (channel === "edge") {
    return edgeImageTokens(signature, config);
  }
  if (channel === "component") {
    return componentImageTokens(signature, config);
  }
  if (channel === "radial") {
    return radialImageTokens(signature, config);
  }
  if (channel === "direction") {
    return directionImageTokens(signature, config);
  }
  throw new Error(`unknown image channel: ${channel}`);
}

export function imageTokens(signature, options = {}) {
  const config = imageConfig(options);
  return signature.map((value) =>
    config.imageBase + Math.min(config.imageBins - 1, Math.floor((Number(value) * config.imageBins) / 256)),
  );
}

function edgeImageTokens(signature, config) {
  const out = [];
  for (let y = 0; y < config.grid; y += 1) {
    for (let x = 0; x < config.grid; x += 1) {
      const center = signature[y * config.grid + x];
      const right = signature[y * config.grid + Math.min(config.grid - 1, x + 1)];
      const down = signature[Math.min(config.grid - 1, y + 1) * config.grid + x];
      const edge = Math.min(255, Math.abs(center - right) + Math.abs(center - down));
      out.push(config.imageBase + Math.min(config.imageBins - 1, Math.floor((edge * config.imageBins) / 256)));
    }
  }
  return out;
}

function componentImageTokens(signature, config) {
  const components = connectedComponents(signature, config);
  const out = [];
  for (let y = 0; y < config.grid; y += 1) {
    for (let x = 0; x < config.grid; x += 1) {
      const index = y * config.grid + x;
      const component = components[index];
      if (component < 0) {
        out.push(config.imageBase);
        continue;
      }
      const size = components.sizes[component] || 1;
      const sizeBucket = size >= 32 ? 3 : size >= 12 ? 2 : size >= 4 ? 1 : 0;
      const crossingBucket = localCrossingBucket(signature, x, y, config);
      const bin = 1 + Math.min(14, sizeBucket * 4 + crossingBucket);
      out.push(config.imageBase + bin);
    }
  }
  return out;
}

function connectedComponents(signature, config) {
  const labels = new Array(config.grid * config.grid).fill(-1);
  const sizes = [];
  for (let y = 0; y < config.grid; y += 1) {
    for (let x = 0; x < config.grid; x += 1) {
      const start = y * config.grid + x;
      if (!isInkCell(signature[start], config) || labels[start] >= 0) {
        continue;
      }
      const label = sizes.length;
      const stack = [start];
      labels[start] = label;
      let size = 0;
      while (stack.length > 0) {
        const current = stack.pop();
        size += 1;
        const cx = current % config.grid;
        const cy = Math.floor(current / config.grid);
        for (const [nx, ny] of fourNeighbors(cx, cy, config)) {
          const next = ny * config.grid + nx;
          if (labels[next] < 0 && isInkCell(signature[next], config)) {
            labels[next] = label;
            stack.push(next);
          }
        }
      }
      sizes.push(size);
    }
  }
  labels.sizes = sizes;
  return labels;
}

function fourNeighbors(x, y, config) {
  const out = [];
  if (x > 0) {
    out.push([x - 1, y]);
  }
  if (x + 1 < config.grid) {
    out.push([x + 1, y]);
  }
  if (y > 0) {
    out.push([x, y - 1]);
  }
  if (y + 1 < config.grid) {
    out.push([x, y + 1]);
  }
  return out;
}

function radialImageTokens(signature, config) {
  const out = [];
  const center = (config.grid - 1) / 2;
  const maxDistance = Math.sqrt(center * center * 2);
  for (let y = 0; y < config.grid; y += 1) {
    for (let x = 0; x < config.grid; x += 1) {
      const value = signature[y * config.grid + x];
      if (!isInkCell(value, config)) {
        out.push(config.imageBase);
        continue;
      }
      const distance = Math.sqrt((x - center) ** 2 + (y - center) ** 2);
      const radialBin = 1 + Math.min(14, Math.floor((distance * 15) / maxDistance));
      out.push(config.imageBase + radialBin);
    }
  }
  return out;
}

function directionImageTokens(signature, config) {
  const out = [];
  for (let y = 0; y < config.grid; y += 1) {
    for (let x = 0; x < config.grid; x += 1) {
      out.push(config.imageBase + directionBin(signature, x, y, config));
    }
  }
  return out;
}

function directionBin(signature, x, y, config) {
  const center = signatureAt(signature, x, y, config);
  const neighbors = [
    isInkCell(signatureAt(signature, x, y - 1, config), config),
    isInkCell(signatureAt(signature, x + 1, y - 1, config), config),
    isInkCell(signatureAt(signature, x + 1, y, config), config),
    isInkCell(signatureAt(signature, x + 1, y + 1, config), config),
    isInkCell(signatureAt(signature, x, y + 1, config), config),
    isInkCell(signatureAt(signature, x - 1, y + 1, config), config),
    isInkCell(signatureAt(signature, x - 1, y, config), config),
    isInkCell(signatureAt(signature, x - 1, y - 1, config), config),
  ];
  const degree = neighbors.filter(Boolean).length;
  if (!isInkCell(center, config) && degree < 2) {
    return 0;
  }
  if (degree === 0) {
    return 1;
  }
  if (degree === 1) {
    return 2;
  }
  if (neighborTransitions(neighbors) >= 4 || degree >= 5) {
    return 15;
  }
  const scores = [
    signatureAt(signature, x - 1, y, config) + signatureAt(signature, x + 1, y, config),
    signatureAt(signature, x, y - 1, config) + signatureAt(signature, x, y + 1, config),
    signatureAt(signature, x - 1, y - 1, config) + signatureAt(signature, x + 1, y + 1, config),
    signatureAt(signature, x + 1, y - 1, config) + signatureAt(signature, x - 1, y + 1, config),
  ];
  let best = 0;
  for (let index = 1; index < scores.length; index += 1) {
    if (scores[index] > scores[best]) {
      best = index;
    }
  }
  const strength = scores[best] >= 384 ? 2 : scores[best] >= 160 ? 1 : 0;
  return 3 + best * 3 + strength;
}

function localCrossingBucket(signature, x, y, config) {
  const degree = [
    isInkCell(signatureAt(signature, x, y - 1, config), config),
    isInkCell(signatureAt(signature, x + 1, y, config), config),
    isInkCell(signatureAt(signature, x, y + 1, config), config),
    isInkCell(signatureAt(signature, x - 1, y, config), config),
  ].filter(Boolean).length;
  if (degree >= 4) return 3;
  if (degree >= 3) return 2;
  if (degree >= 2) return 1;
  return 0;
}

function neighborTransitions(neighbors) {
  let transitions = 0;
  for (let index = 0; index < neighbors.length; index += 1) {
    if (neighbors[index] !== neighbors[(index + 1) % neighbors.length]) {
      transitions += 1;
    }
  }
  return transitions;
}

function signatureAt(signature, x, y, config) {
  if (x < 0 || y < 0 || x >= config.grid || y >= config.grid) return 0;
  return Number(signature[y * config.grid + x] || 0);
}

function isInkCell(value, config) {
  return Number(value || 0) >= config.inkThreshold;
}

function fnv64Hex(tokens) {
  let hash = FNV64_OFFSET;
  for (const token of tokens) {
    hash ^= BigInt(Number(token) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}
