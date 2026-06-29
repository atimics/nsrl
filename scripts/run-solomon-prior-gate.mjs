#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const DEFAULT_TEXT_INDEX =
  "data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv";
const DEFAULT_PROMPTS =
  "data/processed/key-solomon-goetia-latent-v1/scaling-curve/prompts-1425.jsonl";
const DEFAULT_OUT_DIR =
  "data/processed/key-solomon-goetia-latent-v1/prior-gate-hard-classifier";
const GRID = 16;
const BINS = GRID * GRID;
const FEATURE_COUNT = 8192;
const INK_MIDPOINT = 54;
const CONTENT_WINDOW = 16;
const STOPWORDS = new Set([
  "a",
  "about",
  "after",
  "again",
  "all",
  "also",
  "an",
  "and",
  "any",
  "are",
  "as",
  "at",
  "be",
  "before",
  "both",
  "but",
  "by",
  "can",
  "etc",
  "for",
  "from",
  "great",
  "have",
  "he",
  "her",
  "him",
  "his",
  "in",
  "is",
  "it",
  "man",
  "many",
  "men",
  "must",
  "of",
  "or",
  "order",
  "seal",
  "shall",
  "she",
  "spirit",
  "spirits",
  "the",
  "this",
  "thou",
  "to",
  "unto",
  "upon",
  "which",
  "who",
  "will",
  "with",
]);
const CONCEPT_TRIGGERS = new Set([
  "answer",
  "bring",
  "cause",
  "change",
  "discover",
  "give",
  "know",
  "make",
  "produce",
  "recover",
  "return",
  "show",
  "speak",
  "teach",
]);
const PANEL_PROMPTS = [
  "Crocell",
  "Stolas",
  "Bael",
  "hidden geometry and rushing waters",
  "astronomy and herbs teacher",
];
let crcTable = null;

const args = parseArgs(process.argv.slice(2));
const textIndexPath = args["text-index"] ?? DEFAULT_TEXT_INDEX;
const promptsPath = args.prompts ?? DEFAULT_PROMPTS;
const outDir = args["out-dir"] ?? DEFAULT_OUT_DIR;
const epochs = Number.parseInt(args.epochs ?? "16", 10);
const classifierKind = args.classifier ?? "centroid";
const seeds = (args.seeds ?? "solomon-gate-a,solomon-gate-b,solomon-gate-c")
  .split(",")
  .map((seed) => seed.trim())
  .filter(Boolean);

fs.mkdirSync(outDir, { recursive: true });

const spirits = readTextIndex(textIndexPath);
const classByNumber = new Map(spirits.map((spirit, index) => [spirit.number, index]));
const canonicalRows = spirits.map((spirit) => ({
  label: classByNumber.get(spirit.number),
  text: `${spirit.name} ${spirit.aliases.join(" ")} ${spirit.text}`,
  source: "canonical",
}));
const aliasRows = spirits.flatMap((spirit) => {
  const label = classByNumber.get(spirit.number);
  const names = [spirit.name, ...spirit.aliases].filter(Boolean);
  return names.flatMap((name) => [
    { label, text: name, source: "alias" },
    { label, text: `${name} seal`, source: "alias" },
    { label, text: `${name} spirit`, source: "alias" },
    { label, text: `${name} character`, source: "alias" },
    { label, text: `${name} ${spirit.name} ${spirit.aliases.join(" ")}`, source: "alias" },
  ]);
});
const promptRows = readPromptRows(promptsPath, classByNumber);
const derivedRows = derivedConceptRows(spirits, classByNumber);
const trainingRows = [...canonicalRows, ...aliasRows, ...promptRows, ...derivedRows];
const sourceCounts = countBy(trainingRows, (row) => row.source);

const runs = seeds.map((seed) =>
  trainClassifier({ spirits, trainingRows, epochs, seed, classifierKind }),
);
const gateRows = [];
for (const prompt of PANEL_PROMPTS) {
  for (const run of runs) {
    const prediction = predict(run, prompt);
    const spirit = spirits[prediction.label];
    gateRows.push({
      seed: run.seed,
      prompt,
      predictedNumber: spirit.number,
      predictedName: spirit.name,
      score: prediction.score,
      margin: prediction.margin,
      top: predictTop(run, prompt, spirits, 5),
    });
  }
}

const primaryRun = runs[0];
const panelItems = PANEL_PROMPTS.map((prompt) => {
  const prediction = predict(primaryRun, prompt);
  const spirit = spirits[prediction.label];
  return {
    prompt,
    prediction,
    spirit,
    signature: spirit.signature,
  };
});
const distances = pairwiseDistances(panelItems);

const panelPath = path.join(outDir, "decoded-layout-panel.png");
writePng(panelPath, renderPanel(panelItems));

const summaryPath = path.join(outDir, "summary.tsv");
fs.writeFileSync(
  summaryPath,
  [
    "seed\tprompt\tpredicted_number\tpredicted_name\tscore\tmargin",
    ...gateRows.map((row) =>
      [
        row.seed,
        row.prompt,
        row.predictedNumber,
        row.predictedName,
        row.score,
        row.margin,
      ].join("\t"),
    ),
    "",
    "left\tright\tdistance",
    ...distances.map((row) => [row.left, row.right, row.distance].join("\t")),
    "",
  ].join("\n"),
);

const trace = {
  schema: "nsrl.solomon_prior_gate.v1",
  objective: "hard-layout-72-way-text-classifier",
  text_index: textIndexPath,
  prompts: promptsPath,
  out_dir: outDir,
  epochs,
  classifier: classifierKind,
  feature_count: FEATURE_COUNT,
  spirits: spirits.length,
  training_rows: trainingRows.length,
  seeds,
  panel: panelItems.map((item) => ({
    prompt: item.prompt,
    predicted_number: item.spirit.number,
    predicted_name: item.spirit.name,
    score: item.prediction.score,
    margin: item.prediction.margin,
    top5: predictTop(primaryRun, item.prompt, spirits, 5),
  })),
  distances,
  training_sources: sourceCounts,
  notes: [
    "generation uses classifier prediction plus per-spirit hard layout code",
    "no prompt-time bitmap lookup or nearest-signature retrieval",
    "panel is prior-only 16x16 layout, not denoised final seal",
  ],
};
const tracePath = path.join(outDir, "trace.json");
fs.writeFileSync(tracePath, `${JSON.stringify(trace, null, 2)}\n`);

console.log(
  JSON.stringify({
    schema: trace.schema,
    panel: panelPath,
    summary: summaryPath,
    trace: tracePath,
    panel_predictions: trace.panel,
  }),
);

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected argument: ${arg}`);
    }
    const key = arg.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`${arg} requires a value`);
    }
    parsed[key] = value;
    index += 1;
  }
  return parsed;
}

function readTextIndex(filePath) {
  const rows = fs.readFileSync(filePath, "utf8").trimEnd().split("\n");
  return rows.slice(1).map((line, index) => {
    const fields = line.split("\t");
    if (fields.length < 9) {
      throw new Error(`${filePath}:${index + 2} expected at least 9 fields`);
    }
    const signature = fields[7].split(",").map((part) => {
      const value = Number.parseInt(part, 10);
      return value >= INK_MIDPOINT ? 255 : 0;
    });
    if (signature.length !== BINS) {
      throw new Error(`${filePath}:${index + 2} expected ${BINS} signature bins`);
    }
    return {
      number: Number.parseInt(fields[0], 10),
      name: fields[1],
      aliases: fields[2].split("|").filter(Boolean),
      sliceId: fields[3],
      signature,
      text: fields[8],
    };
  });
}

function readPromptRows(filePath, classByNumber) {
  return fs
    .readFileSync(filePath, "utf8")
    .trimEnd()
    .split("\n")
    .filter(Boolean)
    .map((line, index) => {
      const row = JSON.parse(line);
      const label = classByNumber.get(row.spirit_id);
      if (label === undefined) {
        throw new Error(`${filePath}:${index + 1} unknown spirit_id ${row.spirit_id}`);
      }
      return {
        label,
        text: row.text,
        source: row.source ?? "prompt",
      };
    });
}

function derivedConceptRows(spirits, classByNumber) {
  const rows = [];
  for (const spirit of spirits) {
    const label = classByNumber.get(spirit.number);
    const prefix = `${spirit.name} ${spirit.aliases.join(" ")}`.trim();
    const fullContent = unique(contentTokens(tokenize(spirit.text)));
    if (fullContent.length >= 4) {
      rows.push({
        label,
        text: `${prefix} ${fullContent.slice(0, 28).join(" ")}`,
        source: "derived-keywords",
      });
      rows.push({
        label,
        text: fullContent.slice(0, 28).join(" "),
        source: "derived-keywords-bare",
      });
    }
    for (const segment of spirit.text.split(/[.;:—]+/g)) {
      const content = unique(contentTokens(tokenize(segment)));
      if (content.length < 3) continue;
      const hasTrigger = content.some((token) => CONCEPT_TRIGGERS.has(token));
      if (!hasTrigger && content.length < 5) continue;
      const compact = content.slice(0, 18).join(" ");
      rows.push({ label, text: `${prefix} ${compact}`, source: "derived-clause" });
      rows.push({ label, text: compact, source: "derived-clause-bare" });
      if (content.includes("teach")) {
        const taught = content.filter((token) => token !== "teach").slice(0, 14);
        if (taught.length >= 2) {
          rows.push({
            label,
            text: `teach teacher ${taught.join(" ")}`,
            source: "derived-teaching",
          });
          rows.push({
            label,
            text: `${prefix} teach teacher ${taught.join(" ")}`,
            source: "derived-teaching",
          });
        }
      }
    }
  }
  return rows;
}

function trainClassifier({ spirits, trainingRows, epochs, seed, classifierKind }) {
  if (classifierKind === "centroid") {
    return trainCentroidClassifier({ spirits, trainingRows, seed });
  }
  if (classifierKind !== "perceptron") {
    throw new Error(`unknown classifier: ${classifierKind}`);
  }
  const weights = Array.from({ length: spirits.length }, () => new Int32Array(FEATURE_COUNT));
  const biases = new Int32Array(spirits.length);
  const examples = trainingRows.map((row) => ({
    label: row.label,
    features: textFeatures(row.text),
    weight: rowWeight(row),
  }));
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const order = shuffledIndices(examples.length, `${seed}:${epoch}`);
    for (const exampleIndex of order) {
      const example = examples[exampleIndex];
      const prediction = predictFromFeatures({ weights, biases }, example.features);
      if (prediction.label === example.label) {
        biases[example.label] += 1;
        continue;
      }
      updateClass(weights[example.label], example.features, 1);
      updateClass(weights[prediction.label], example.features, -1);
      biases[example.label] += 4;
      biases[prediction.label] -= 4;
    }
  }
  return { seed, classifierKind, weights, biases };
}

function trainCentroidClassifier({ spirits, trainingRows, seed }) {
  const examples = trainingRows.map((row) => ({
    label: row.label,
    features: textFeatures(row.text),
    weight: rowWeight(row),
  }));
  const documentFrequency = new Uint32Array(FEATURE_COUNT);
  for (const example of examples) {
    for (const feature of new Set(example.features.map(([index]) => index))) {
      documentFrequency[feature] += 1;
    }
  }
  const idf = new Float64Array(FEATURE_COUNT);
  for (let index = 0; index < FEATURE_COUNT; index += 1) {
    idf[index] = Math.log((examples.length + 1) / (documentFrequency[index] + 1)) + 1;
  }
  const centroids = Array.from({ length: spirits.length }, () => new Float64Array(FEATURE_COUNT));
  const counts = new Uint32Array(spirits.length);
  for (const example of examples) {
    counts[example.label] += example.weight;
    const centroid = centroids[example.label];
    for (const [feature, value] of example.features) {
      centroid[feature] += value * idf[feature] * example.weight;
    }
  }
  const norms = new Float64Array(spirits.length);
  for (let label = 0; label < centroids.length; label += 1) {
    const centroid = centroids[label];
    const count = Math.max(1, counts[label]);
    let norm = 0;
    for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
      centroid[feature] /= count;
      norm += centroid[feature] * centroid[feature];
    }
    norms[label] = Math.sqrt(norm) || 1;
  }
  return { seed, classifierKind: "centroid", centroids, norms, idf };
}

function predict(model, text) {
  return predictFromFeatures(model, textFeatures(text));
}

function predictFromFeatures(model, features) {
  const ranked = rankFromFeatures(model, features, 2);
  return {
    label: ranked[0].label,
    score: ranked[0].score,
    margin: ranked[0].score - (ranked[1]?.score ?? 0),
  };
}

function predictTop(model, text, spirits, count) {
  return rankFromFeatures(model, textFeatures(text), count).map((prediction) => {
    const spirit = spirits[prediction.label];
    return {
      number: spirit.number,
      name: spirit.name,
      score: prediction.score,
    };
  });
}

function rankFromFeatures(model, features, count) {
  if (model.classifierKind === "centroid") {
    return rankFromCentroids(model, features, count);
  }
  const ranked = [];
  for (let label = 0; label < model.weights.length; label += 1) {
    let score = model.biases[label];
    const weights = model.weights[label];
    for (const [feature, value] of features) {
      score += weights[feature] * value;
    }
    ranked.push({ label, score });
  }
  ranked.sort((left, right) => right.score - left.score || left.label - right.label);
  return ranked.slice(0, count);
}

function predictFromCentroids(model, features) {
  const ranked = rankFromCentroids(model, features, 2);
  return {
    label: ranked[0].label,
    score: ranked[0].score,
    margin: ranked[0].score - (ranked[1]?.score ?? 0),
  };
}

function rankFromCentroids(model, features, count) {
  let queryNorm = 0;
  const weighted = features.map(([feature, value]) => {
    const next = value * model.idf[feature];
    queryNorm += next * next;
    return [feature, next];
  });
  queryNorm = Math.sqrt(queryNorm) || 1;
  const ranked = [];
  for (let label = 0; label < model.centroids.length; label += 1) {
    const centroid = model.centroids[label];
    let score = 0;
    for (const [feature, value] of weighted) {
      score += centroid[feature] * value;
    }
    score = Math.round((score * 1000000) / (model.norms[label] * queryNorm));
    ranked.push({ label, score });
  }
  ranked.sort((left, right) => right.score - left.score || left.label - right.label);
  return ranked.slice(0, count);
}

function updateClass(weights, features, direction) {
  for (const [feature, value] of features) {
    weights[feature] += direction * value;
  }
}

function rowWeight(row) {
  if (row.source === "alias") return 8;
  if (row.source === "derived-teaching") return 2;
  if (row.source === "derived-clause-bare" || row.source === "derived-keywords-bare") return 2;
  return 1;
}

function textFeatures(text) {
  const tokens = tokenize(text);
  const accum = new Map();
  if (tokens.length > 0 && tokens.length <= 4) {
    addFeature(accum, "whole", tokens.join(" "), 0, 40);
  }
  for (let index = 0; index < tokens.length; index += 1) {
    addFeature(accum, "tok", tokens[index], index, 3);
    if (tokens[index + 1]) {
      addFeature(accum, "bi", `${tokens[index]} ${tokens[index + 1]}`, index, 4);
    }
    if (tokens[index + 1] && tokens[index + 2]) {
      addFeature(
        accum,
        "tri",
        `${tokens[index]} ${tokens[index + 1]} ${tokens[index + 2]}`,
        index,
        5,
      );
    }
  }
  const content = contentTokens(tokens);
  if (content.length > 0 && content.length <= 5) {
    addFeature(accum, "cwhole", content.join(" "), 0, 44);
    addFeature(accum, "cset", sortedKey(content), 0, 44);
  }
  for (let index = 0; index < content.length; index += 1) {
    addFeature(accum, "ctok", content[index], index, 5);
    if (content[index + 1]) {
      addFeature(accum, "cbi", `${content[index]} ${content[index + 1]}`, index, 6);
    }
    if (content[index + 1] && content[index + 2]) {
      addFeature(
        accum,
        "ctri",
        `${content[index]} ${content[index + 1]} ${content[index + 2]}`,
        index,
        7,
      );
    }
    const windowEnd = Math.min(content.length, index + CONTENT_WINDOW);
    for (let right = index + 1; right < windowEnd; right += 1) {
      addFeature(accum, "skip2", `${content[index]} ${content[right]}`, index, 7);
      addFeature(accum, "pair", sortedKey([content[index], content[right]]), index, 8);
      for (let third = right + 1; third < windowEnd; third += 1) {
        addFeature(
          accum,
          "triple",
          sortedKey([content[index], content[right], content[third]]),
          index,
          8,
        );
      }
    }
  }
  return [...accum.entries()];
}

function addFeature(accum, namespace, text, position, base) {
  if (text.length < 2) return;
  const hash = fnv(`${namespace}\xff${text}`);
  const feature = hash % FEATURE_COUNT;
  const sign = hash & 0x80000000 ? -1 : 1;
  const value = sign * Math.min(31, base + Math.min(20, text.length) + (position % 7));
  accum.set(feature, Math.max(-127, Math.min(127, (accum.get(feature) ?? 0) + value)));
}

function tokenize(text) {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .map(normalizeToken)
    .filter((token) => token.length >= 2);
}

function contentTokens(tokens) {
  return tokens.filter((token) => token.length >= 3 && !STOPWORDS.has(token));
}

function unique(tokens) {
  const seen = new Set();
  const out = [];
  for (const token of tokens) {
    if (seen.has(token)) continue;
    seen.add(token);
    out.push(token);
  }
  return out;
}

function sortedKey(tokens) {
  return [...tokens].sort().join("\x00");
}

function countBy(items, keyFn) {
  const counts = {};
  for (const item of items) {
    const key = keyFn(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function normalizeToken(token) {
  if (["teach", "teacher", "teaches", "teacheth", "teaching"].includes(token)) {
    return "teach";
  }
  if (["know", "knows", "known", "knowing", "knoweth", "knowledge"].includes(token)) {
    return "know";
  }
  if (["make", "makes", "maketh", "making"].includes(token)) {
    return "make";
  }
  if (["discover", "discovers", "discovereth", "discovering"].includes(token)) {
    return "discover";
  }
  if (["produce", "produces", "produceth", "producing"].includes(token)) {
    return "produce";
  }
  if (["answer", "answers", "answereth", "answering"].includes(token)) {
    return "answer";
  }
  if (["virtue", "virtues"].includes(token)) {
    return "virtue";
  }
  if (["water", "waters"].includes(token)) {
    return "water";
  }
  if (["rush", "rushing", "rushings"].includes(token)) {
    return "rush";
  }
  if (["herb", "herbs"].includes(token)) {
    return "herb";
  }
  if (["stone", "stones"].includes(token)) {
    return "stone";
  }
  if (["science", "sciences"].includes(token)) {
    return "science";
  }
  if (token.length > 5 && token.endsWith("eth")) {
    return token.slice(0, -3);
  }
  if (token.length > 5 && token.endsWith("ing")) {
    return token.slice(0, -3);
  }
  if (token.length > 4 && token.endsWith("es")) {
    return token.slice(0, -2);
  }
  if (token.length > 3 && token.endsWith("s")) {
    return token.slice(0, -1);
  }
  return token;
}

function shuffledIndices(count, seed) {
  const indices = Array.from({ length: count }, (_, index) => index);
  let state = fnv(seed);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    state = xorshift32(state);
    const swapIndex = state % (index + 1);
    [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
  }
  return indices;
}

function xorshift32(value) {
  let x = value >>> 0;
  x ^= (x << 13) >>> 0;
  x ^= x >>> 17;
  x ^= (x << 5) >>> 0;
  return x >>> 0;
}

function fnv(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function pairwiseDistances(items) {
  const rows = [];
  for (let left = 0; left < items.length; left += 1) {
    for (let right = left + 1; right < items.length; right += 1) {
      rows.push({
        left: items[left].prompt,
        right: items[right].prompt,
        distance: signatureDistance(items[left].signature, items[right].signature),
      });
    }
  }
  return rows;
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < left.length; index += 1) {
    distance += Math.abs(left[index] - right[index]);
  }
  return distance;
}

function renderPanel(items) {
  const scale = 8;
  const gap = 8;
  const tile = GRID * scale;
  const width = items.length * tile + (items.length - 1) * gap;
  const height = tile;
  const pixels = Buffer.alloc(width * height * 4, 255);
  for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
    const xOffset = itemIndex * (tile + gap);
    const signature = items[itemIndex].signature;
    for (let y = 0; y < GRID; y += 1) {
      for (let x = 0; x < GRID; x += 1) {
        const value = 255 - signature[y * GRID + x];
        for (let sy = 0; sy < scale; sy += 1) {
          for (let sx = 0; sx < scale; sx += 1) {
            const px = xOffset + x * scale + sx;
            const py = y * scale + sy;
            const offset = (py * width + px) * 4;
            pixels[offset] = value;
            pixels[offset + 1] = value;
            pixels[offset + 2] = value;
            pixels[offset + 3] = 255;
          }
        }
      }
    }
  }
  return { width, height, pixels };
}

function writePng(filePath, image) {
  const raw = Buffer.alloc((image.width * 4 + 1) * image.height);
  for (let y = 0; y < image.height; y += 1) {
    const rawOffset = y * (image.width * 4 + 1);
    raw[rawOffset] = 0;
    image.pixels.copy(
      raw,
      rawOffset + 1,
      y * image.width * 4,
      (y + 1) * image.width * 4,
    );
  }
  const chunks = [
    pngChunk("IHDR", ihdr(image.width, image.height)),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ];
  fs.writeFileSync(filePath, Buffer.concat([Buffer.from("\x89PNG\r\n\x1a\n", "binary"), ...chunks]));
}

function ihdr(width, height) {
  const out = Buffer.alloc(13);
  out.writeUInt32BE(width, 0);
  out.writeUInt32BE(height, 4);
  out[8] = 8;
  out[9] = 6;
  out[10] = 0;
  out[11] = 0;
  out[12] = 0;
  return out;
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function crc32(buffer) {
  if (!crcTable) {
    crcTable = Array.from({ length: 256 }, (_, index) => {
      let crc = index;
      for (let bit = 0; bit < 8; bit += 1) {
        crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
      }
      return crc >>> 0;
    });
  }
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
