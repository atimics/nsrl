const MODEL_MAGIC = "NSRLLMM1";
const MODEL_VERSION = 5;
const TRANSFORMER_MAGICS = new Set(["NSRLMT4\n", "NSRLMT5\n"]);
const PAD = 0;
const BOS = 1;
const PROMPT = 2;
const TEXT = 3;
const IMAGE = 4;
const EOS = 5;
const TASK_TEXT_TO_IMAGE = 6;
const TASK_IMAGE_TO_TEXT = 7;
const TASK_MATCH = 8;
const TASK_EXPLAIN = 9;
const TASK_IDENTIFY = 10;
const TEXT_BASE = 16;
const TEXT_COUNT = 128;
const IMAGE_BASE = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS = 16;
const TEXT_CHUNK_BASE = 160;
const TEXT_CHUNK_NAME_START = 24;
const TEXT_TOKEN_PROFILE_CHAR = 0;
const TEXT_TOKEN_PROFILE_CHUNKED = 1;
const TEXT_CHUNKS = [
  "Solomon selects ",
  ": ",
  "He ",
  "is ",
  "appeareth ",
  "maketh ",
  "teacheth ",
  "giveth ",
  "causeth ",
  "knoweth ",
  "healeth ",
  "teaches ",
  "and ",
  "the ",
  "of ",
  "to ",
  "in ",
  "a ",
  "his ",
  "with ",
  "upon ",
  "unto ",
  "This ",
  "His ",
  "Bael",
  "Agares",
  "Vassago",
  "Samigina",
  "Marbas",
  "Valefor",
  "Amon",
  "Barbatos",
  "Paimon",
  "Buer",
  "Gusion",
  "Sitri",
  "Beleth",
  "Leraje",
  "Eligos",
  "Zepar",
  "Botis",
  "Bathin",
  "Sallos",
  "Purson",
  "Marax",
  "Ipos",
  "Aim",
  "Naberius",
  "Glasya-Labolas",
  "Bune",
  "Ronove",
  "Berith",
  "Astaroth",
  "Forneus",
  "Foras",
  "Asmoday",
  "Gaap",
  "Furfur",
  "Marchosias",
  "Stolas",
  "Phenex",
  "Halphas",
  "Malphas",
  "Raum",
  "Focalor",
  "Vepar",
  "Sabnock",
  "Shax",
  "Vine",
  "Bifrons",
  "Uvall",
  "Haagenti",
  "Crocell",
  "Furcas",
  "Balam",
  "Alloces",
  "Camio",
  "Murmur",
  "Orobas",
  "Gremory",
  "Ose",
  "Amy",
  "Oriax",
  "Vapula",
  "Zagan",
  "Volac",
  "Andras",
  "Haures",
  "Andrealphus",
  "Cimejes",
  "Amdusias",
  "Belial",
  "Decarabia",
  "Seere",
  "Dantalion",
  "Andromalius",
];
const SIGNATURE_GRID = 16;
const SIGNATURE_BINS = SIGNATURE_GRID * SIGNATURE_GRID;
const VOCAB_SIZE = 256;
const OUTPUT_SHIFT = 8;
const Q15_SHIFT = 15;
const RMS_NORM_EPSILON = 1;
const INV_SQRT_2_Q15 = 23170;
const RSQRT_LUT_8BIT = [
  32767, 32641, 32515, 32391, 32268, 32146, 32026, 31907, 31790, 31673, 31558, 31445, 31332,
  31221, 31111, 31002, 30894, 30787, 30682, 30577, 30474, 30371, 30270, 30169, 30070, 29972,
  29874, 29778, 29682, 29587, 29494, 29401, 29309, 29217, 29127, 29038, 28949, 28861, 28774,
  28688, 28602, 28518, 28434, 28350, 28268, 28186, 28105, 28024, 27945, 27866, 27787, 27709,
  27632, 27556, 27480, 27405, 27330, 27256, 27183, 27110, 27038, 26966, 26895, 26825, 26755,
  26686, 26617, 26548, 26481, 26413, 26346, 26280, 26214, 26149, 26084, 26020, 25956, 25893,
  25830, 25767, 25705, 25644, 25583, 25522, 25462, 25402, 25342, 25283, 25225, 25167, 25109,
  25051, 24994, 24938, 24882, 24826, 24770, 24715, 24660, 24606, 24552, 24498, 24445, 24392,
  24339, 24287, 24235, 24184, 24132, 24081, 24031, 23980, 23930, 23881, 23831, 23782, 23733,
  23685, 23637, 23589, 23541, 23494, 23447, 23400, 23354, 23307, 23262, 23216,
];
const DEFAULT_EMBEDDED_TEXT_LM_ORDER = 12;
const DEFAULT_EMBEDDED_TEXT_LM_MIN_ORDER = 3;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

export class SolomonAttentionSampler {
  constructor(modelBytes) {
    const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
    const expectedHash = readU64AtEnd(bytes);
    const actualHash = hashBytes(bytes.subarray(0, bytes.length - 8));
    if (expectedHash !== actualHash) {
      throw new Error(
        `NSRLLMM1 hash mismatch: expected ${hex64(expectedHash)}, got ${hex64(actualHash)}`,
      );
    }

    const reader = new BinaryReader(bytes.subarray(0, bytes.length - 8));
    const magic = reader.readAscii(8);
    if (magic !== MODEL_MAGIC) {
      throw new Error("not an NSRLLMM1 model");
    }
    const version = reader.readU32();
    if (version === 0 || version > MODEL_VERSION) {
      throw new Error(`unsupported NSRLLMM1 version ${version}`);
    }
    expectU32(reader, PAD, "pad token");
    expectU32(reader, VOCAB_SIZE, "vocab size");
    expectU32(reader, TEXT_BASE, "text base");
    expectU32(reader, TEXT_COUNT, "text count");
    expectU32(reader, IMAGE_BASE, "image base");
    expectU32(reader, IMAGE_BINS, "image bins");
    expectU32(reader, SIGNATURE_GRID, "signature grid");
    this.attentionKind = reader.readU32();
    this.positionPolicy = reader.readU32();
    this.textTokenProfile = version >= 2 ? reader.readU32() : TEXT_TOKEN_PROFILE_CHAR;
    if (this.attentionKind !== 0) {
      throw new Error(`web sampler supports base2 attention only, got ${this.attentionKind}`);
    }
    if (
      this.textTokenProfile !== TEXT_TOKEN_PROFILE_CHAR &&
      this.textTokenProfile !== TEXT_TOKEN_PROFILE_CHUNKED
    ) {
      throw new Error(`unsupported NSRLLMM1 text token profile ${this.textTokenProfile}`);
    }
    this.textMemory = version >= 3 ? reader.readTextMemory(version) : null;
    this.contextSeqLen = reader.readU32();
    this.tokenCount = Number(reader.readU64());
    this.tokenHash = reader.readU64();
    this.innerModelHash = reader.readU64();
    const transformerLength = Number(reader.readU64());
    const transformerBytes = reader.readBytes(transformerLength);
    if (!reader.isDone()) {
      throw new Error("trailing bytes in NSRLLMM1 body");
    }
    this.transformer = new MiniTransformerModel(transformerBytes);
    if (this.transformer.contextSeqLen !== this.contextSeqLen) {
      throw new Error("NSRLLMM1 context length mismatch");
    }
    this.modelHash = actualHash;
  }

  sample(prompt, options = {}) {
    const normalizedPrompt = normalizeText(prompt || "king solomon seal");
    const seed = (Number(options.seed || 1) >>> 0) ^ hashText32(normalizedPrompt);
    const topK = Math.max(1, Number(options.topK || 1));
    const maxTextTokens = Math.max(1, Number(options.maxTextTokens || 220));
    const suppressNameChunksAfterOpening = options.suppressNameChunksAfterOpening === true;

    const baseTokens = [
      BOS,
      PROMPT,
      ...encodeTextTokens(normalizedPrompt, this.textTokenProfile),
      TEXT,
    ];
    let tokens = baseTokens.slice();
    let textTokens = null;
    let textSource = "raw_attention";
    let textLmfallback = false;
    const memoryExample = this.selectTextExample(normalizedPrompt, seed);
    if (memoryExample) {
      const textPrior = buildEmbeddedTextPrior(
        memoryExample,
        this.textTokenProfile,
        DEFAULT_EMBEDDED_TEXT_LM_ORDER,
        DEFAULT_EMBEDDED_TEXT_LM_MIN_ORDER,
      );
      const generatedTextTokens = this.generateTextTokens(tokens, seed, topK, maxTextTokens, textPrior, {
        suppressNameChunksAfterOpening,
      });
      const memoryTextTokens = embeddedTextTokens(memoryExample, this.textTokenProfile);
      if (shouldUseMemoryTextFallback(generatedTextTokens, memoryTextTokens, maxTextTokens)) {
        textTokens = memoryTextTokens;
        tokens = baseTokens.concat(textTokens);
        textSource = "embedded_text_memory_guard";
        textLmfallback = true;
      } else {
        textTokens = generatedTextTokens;
        textSource = "embedded_text_lm_strict";
      }
    } else {
      textTokens = this.generateTextTokens(tokens, seed, topK, maxTextTokens, null, {
        suppressNameChunksAfterOpening,
      });
    }

    tokens.push(IMAGE);
    const imageBins = new Uint8Array(SIGNATURE_BINS);
    if (memoryExample?.imageTokens?.length === SIGNATURE_BINS) {
      for (let index = 0; index < imageBins.length; index += 1) {
        const imageToken = isImageToken(memoryExample.imageTokens[index])
          ? memoryExample.imageTokens[index]
          : IMAGE_BASE;
        imageBins[index] = imageToken - IMAGE_BASE;
        tokens.push(imageToken);
      }
    } else {
      for (let index = 0; index < imageBins.length; index += 1) {
        const token = this.nextToken(tokens, "image", seed, maxTextTokens + index, topK, false);
        const imageToken = isImageToken(token) ? token : IMAGE_BASE;
        imageBins[index] = imageToken - IMAGE_BASE;
        tokens.push(imageToken);
      }
    }
    tokens.push(EOS);

    const width = 192;
    const height = 192;
    const rgba = renderImageBins(imageBins, width, height);
    return {
      width,
      height,
      rgba,
      text: decodeTextTokens(textTokens, this.textTokenProfile),
      metadata: {
        model_kind: "NSRLLMM1",
        model_hash: hex64(this.modelHash),
        inner_model_hash: hex64(this.innerModelHash),
        token_hash: hex64(this.tokenHash),
        token_count: this.tokenCount,
        text_source: textSource,
        text_lm_fallback: textLmfallback,
        image_source:
          memoryExample?.imageTokens?.length === SIGNATURE_BINS
            ? "embedded_image_memory_strict"
            : "raw_attention",
        text_memory_order: this.textMemory?.order || 0,
        text_lm_order: memoryExample ? DEFAULT_EMBEDDED_TEXT_LM_ORDER : 0,
        text_lm_min_order: memoryExample ? DEFAULT_EMBEDDED_TEXT_LM_MIN_ORDER : 0,
        text_memory_examples: this.textMemory?.examples.length || 0,
        image_grid: SIGNATURE_GRID,
      },
    };
  }

  diagnoseMemoryContinuation(prompt, options = {}) {
    const normalizedPrompt = normalizeText(prompt || "king solomon seal");
    const seed = (Number(options.seed || 1) >>> 0) ^ hashText32(normalizedPrompt);
    const textPrefix = options.textPrefix ?? "Solomon selects ";
    const topN = Math.max(1, Number(options.topN || 10));
    const memoryExample = this.selectTextExample(normalizedPrompt, seed);
    if (!memoryExample) {
      return {
        prompt: normalizedPrompt,
        textPrefix,
        memoryFound: false,
        candidates: [],
      };
    }
    const prefixTokens = encodeTextPrefixTokens(textPrefix, this.textTokenProfile);
    const memoryTextTokens = embeddedTextTokens(memoryExample, this.textTokenProfile);
    const prefixMatchesMemory = prefixTokens.every(
      (token, index) => memoryTextTokens[index] === token,
    );
    const targetToken = prefixMatchesMemory ? memoryTextTokens[prefixTokens.length] : undefined;
    const history = [
      BOS,
      PROMPT,
      ...encodeTextTokens(normalizedPrompt, this.textTokenProfile),
      TEXT,
      ...prefixTokens,
    ];
    const row = this.transformer.forward(paddedContext(history, this.contextSeqLen), this.positionPolicy);
    const candidates = sortedAllowedCandidates(row, "text", false, this.textTokenProfile);
    const targetIndex = candidates.findIndex((candidate) => candidate.token === targetToken);
    const targetCandidate =
      targetIndex >= 0
        ? candidates[targetIndex]
        : {
            token: targetToken ?? 0,
            logit: targetToken === undefined ? 0 : row.logits[targetToken],
            probability: targetToken === undefined ? 0 : row.probabilities[targetToken],
          };
    const bestCompetitor = candidates.find((candidate) => candidate.token !== targetToken);
    return {
      prompt: normalizedPrompt,
      primaryName: memoryExample.primaryName,
      memoryPrompt: memoryExample.prompt,
      textPrefix,
      memoryFound: true,
      prefixMatchesMemory,
      expectedToken: formatToken(targetToken),
      expectedRank:
        targetToken === undefined ? null : targetIndex >= 0 ? targetIndex + 1 : candidates.length + 1,
      expectedMarginQ8:
        targetToken === undefined
          ? null
          : bestCompetitor
            ? targetCandidate.logit - bestCompetitor.logit
            : targetCandidate.logit,
      candidates: candidates.slice(0, topN).map((candidate) => ({
        token: candidate.token,
        text: formatToken(candidate.token),
        logitQ8: candidate.logit,
        probabilityQ15: candidate.probability,
      })),
    };
  }

  scoreRawContinuations(prompt, continuations, options = {}) {
    const normalizedPrompt = normalizeText(prompt);
    const candidates = requireRawContinuationCandidates(continuations, this.textTokenProfile);
    const taskToken = rawScoringTaskToken(options.task || "text");
    const imageTokens = rawScoringImageTokens(options);
    const prefix = [BOS];
    if (taskToken !== null) {
      prefix.push(taskToken);
    }
    prefix.push(PROMPT, ...encodeTextTokens(normalizedPrompt, this.textTokenProfile));
    if (imageTokens) {
      prefix.push(IMAGE, ...imageTokens);
    }
    prefix.push(TEXT);

    const scores = candidates.map((candidate) =>
      scoreRawContinuationCandidate(
        this.transformer,
        this.positionPolicy,
        this.contextSeqLen,
        prefix,
        candidate,
      ));
    scores.sort((left, right) =>
      right.mean_log_probability_microunits - left.mean_log_probability_microunits ||
      left.candidate_id.localeCompare(right.candidate_id),
    );
    const winner = scores[0];
    const runnerUp = scores[1] || null;
    const visiblePrefix = prefix.slice(-this.contextSeqLen);
    const promptToken = encodeTextTokens(normalizedPrompt, this.textTokenProfile);

    return {
      schema: "nsrl.solomon_raw_continuation_score.v0",
      selected_candidate_id: winner.candidate_id,
      selected_text: winner.text,
      margin_microunits: runnerUp
        ? winner.mean_log_probability_microunits - runnerUp.mean_log_probability_microunits
        : null,
      scores,
      conditioning: {
        task: options.task || "text",
        task_token: taskToken,
        normalized_prompt: normalizedPrompt,
        prompt_token_count: promptToken.length,
        image_token_count: imageTokens?.length || 0,
        prefix_token_count: prefix.length,
        prefix_token_hash: fnv64Hex(prefix),
        context_seq_len: this.contextSeqLen,
        truncated_prefix_tokens: Math.max(0, prefix.length - this.contextSeqLen),
        visible_prefix_tokens: visiblePrefix,
        prompt_marker_visible: visiblePrefix.includes(PROMPT),
        image_marker_visible: imageTokens ? visiblePrefix.includes(IMAGE) : null,
      },
      provenance: {
        model_kind: MODEL_MAGIC,
        model_hash: hex64(this.modelHash),
        inner_model_hash: hex64(this.innerModelHash),
        token_hash: hex64(this.tokenHash),
        token_count: this.tokenCount,
        raw_transformer_only: true,
        embedded_text_memory_used: false,
        retrieval_used: false,
        oracle_or_target_lookup_used: false,
      },
    };
  }

  generateTextTokens(tokens, seed, topK, maxTextTokens, textPrior = null, options = {}) {
    const textTokens = [];
    for (let step = 0; step < maxTextTokens; step += 1) {
      const allowStop = step >= 16;
      const token = this.nextToken(tokens, "text", seed, step, topK, allowStop, textPrior, options);
      if (allowStop && (token === IMAGE || token === EOS)) {
        break;
      }
      if (isTextToken(token, this.textTokenProfile)) {
        tokens.push(token);
        textTokens.push(token);
      }
    }
    return textTokens;
  }

  nextToken(history, phase, seed, step, topK, allowStop, textPrior = null, options = {}) {
    const context = paddedContext(history, this.contextSeqLen);
    const row = this.transformer.forward(context, this.positionPolicy);
    const candidates = sortedAllowedCandidates(row, phase, allowStop, this.textTokenProfile);
    if (
      phase === "text" &&
      this.textTokenProfile === TEXT_TOKEN_PROFILE_CHUNKED &&
      options.suppressNameChunksAfterOpening === true &&
      generatedTextIsAfterOpening(history, this.textTokenProfile)
    ) {
      suppressNameChunkCandidates(candidates);
    }
    if (phase === "text" && textPrior) {
      applyEmbeddedTextPrior(
        candidates,
        generatedTextContext(history, this.textTokenProfile),
        textPrior,
      );
    }
    return chooseCandidate(candidates, seed, step, topK).token;
  }

  selectTextExample(prompt, seed) {
    const examples = this.textMemory?.examples || [];
    if (examples.length === 0) {
      return null;
    }
    const promptKey = normalizeKey(prompt);
    const scored = [];
    for (const example of examples) {
      const score = textMemoryPromptScopeScore(promptKey, example);
      if (score > 0) {
        scored.push({ example, score });
      }
    }
    const pool = scored.length > 0 ? scored : examples.map((example) => ({ example, score: 1 }));
    pool.sort((left, right) => right.score - left.score || left.example.primaryName.localeCompare(right.example.primaryName));
    const bestScore = pool[0].score;
    const best = pool.filter((entry) => entry.score === bestScore);
    return best[mix32(seed) % best.length]?.example || null;
  }
}

function requireRawContinuationCandidates(continuations, textTokenProfile) {
  if (!Array.isArray(continuations) || continuations.length < 2) {
    throw new Error("raw continuation scoring requires at least two candidates");
  }
  const ids = new Set();
  return continuations.map((candidate, index) => {
    const value = typeof candidate === "string"
      ? {candidate_id: candidate, text: candidate}
      : candidate;
    const candidateId = String(value?.candidate_id || "").trim();
    const text = normalizeTextPreservingEdgeSpace(value?.text || "");
    if (!candidateId || ids.has(candidateId)) {
      throw new Error(`raw continuation candidate ${index} has a missing or duplicate id`);
    }
    ids.add(candidateId);
    const tokens = encodeNormalizedTextTokens(text, textTokenProfile);
    if (tokens.length === 0) {
      throw new Error(`raw continuation candidate ${candidateId} has no encodable text`);
    }
    return {candidate_id: candidateId, text, tokens};
  });
}

function rawScoringTaskToken(task) {
  const tasks = new Map([
    ["text", null],
    ["match", TASK_MATCH],
    ["explain", TASK_EXPLAIN],
    ["identify", TASK_IDENTIFY],
    ["image-to-text", TASK_IMAGE_TO_TEXT],
    ["text-to-image", TASK_TEXT_TO_IMAGE],
  ]);
  if (!tasks.has(task)) {
    throw new Error(`unknown raw continuation scoring task ${task}`);
  }
  return tasks.get(task);
}

function rawScoringImageTokens(options) {
  if (options.imageTokens !== undefined && options.imageBins !== undefined) {
    throw new Error("provide raw imageTokens or imageBins, not both");
  }
  if (options.imageTokens !== undefined) {
    if (!Array.isArray(options.imageTokens) && !ArrayBuffer.isView(options.imageTokens)) {
      throw new Error("raw imageTokens must be an array");
    }
    const tokens = Array.from(options.imageTokens, Number);
    if (tokens.length === 0 || tokens.some((token) =>
      !Number.isInteger(token) ||
      !(isImageToken(token) || (token >= TASK_IDENTIFY + 1 && token < TEXT_BASE)))) {
      throw new Error("raw imageTokens contain an invalid token");
    }
    return tokens;
  }
  if (options.imageBins !== undefined) {
    if (!Array.isArray(options.imageBins) && !ArrayBuffer.isView(options.imageBins)) {
      throw new Error("raw imageBins must be an array");
    }
    const bins = Array.from(options.imageBins, Number);
    if (bins.length === 0 || bins.some((bin) =>
      !Number.isInteger(bin) || bin < 0 || bin >= IMAGE_BINS)) {
      throw new Error(`raw imageBins must be integers in [0, ${IMAGE_BINS - 1}]`);
    }
    return bins.map((bin) => IMAGE_BASE + bin);
  }
  return null;
}

function scoreRawContinuationCandidate(
  transformer,
  positionPolicy,
  contextSeqLen,
  prefix,
  candidate,
) {
  const history = prefix.slice();
  let logProbability = 0;
  let logitSum = 0;
  const tokenRows = [];
  for (const token of candidate.tokens) {
    const context = paddedContext(history, contextSeqLen);
    const row = transformer.forward(context, positionPolicy);
    const tokenLogProbability = logSoftmaxForToken(row.logits, token);
    logProbability += tokenLogProbability;
    logitSum += row.logits[token];
    tokenRows.push({
      token,
      logit_q8: row.logits[token],
      probability_q15: row.probabilities[token],
      context_hash: fnv64Hex(context),
    });
    history.push(token);
  }
  const meanLogProbability = logProbability / candidate.tokens.length;
  return {
    candidate_id: candidate.candidate_id,
    text: candidate.text,
    token_count: candidate.tokens.length,
    token_hash: fnv64Hex(candidate.tokens),
    logit_sum_q8: logitSum,
    log_probability_microunits: Math.round(logProbability * 1_000_000),
    mean_log_probability_microunits: Math.round(meanLogProbability * 1_000_000),
    zero_probability_tokens_q15: tokenRows.filter((row) => row.probability_q15 === 0).length,
    token_rows: tokenRows,
  };
}

function logSoftmaxForToken(logits, token) {
  let maxLogit = -Infinity;
  for (const logit of logits) {
    maxLogit = Math.max(maxLogit, logit);
  }
  let normalizer = 0;
  for (const logit of logits) {
    normalizer += Math.pow(2, (logit - maxLogit) / 256);
  }
  return ((logits[token] - maxLogit) / 256) * Math.LN2 - Math.log(normalizer);
}

export class MiniTransformerModel {
  constructor(bytes) {
    const reader = new BinaryReader(bytes);
    const magic = reader.readAscii(8);
    if (!TRANSFORMER_MAGICS.has(magic)) {
      throw new Error("bad mini-transformer magic");
    }
    expectU32(reader, VOCAB_SIZE, "mini-transformer vocab");
    this.dModel = reader.readU32();
    this.heads = reader.readU32();
    this.hiddenDim = reader.readU32();
    if (
      this.dModel <= 0 ||
      this.heads <= 0 ||
      this.hiddenDim <= 0 ||
      this.dModel % this.heads !== 0
    ) {
      throw new Error("invalid mini-transformer geometry");
    }
    this.headDim = this.dModel / this.heads;
    this.attentionScaleShift = powerOfFourSqrtShift(this.headDim);
    this.contextSeqLen = reader.readU32();
    const embeddingCount = Number(reader.readU64());
    const positionEmbeddingCount = Number(reader.readU64());
    const qCount = Number(reader.readU64());
    const kCount = Number(reader.readU64());
    const vCount = Number(reader.readU64());
    const oCount = Number(reader.readU64());
    const upCount = Number(reader.readU64());
    const gateCount = Number(reader.readU64());
    const downCount = Number(reader.readU64());
    const outputCount = Number(reader.readU64());
    this.embeddingHash = reader.readU64();
    this.attentionQHash = reader.readU64();
    this.attentionKHash = reader.readU64();
    this.attentionVHash = reader.readU64();
    this.attentionOHash = reader.readU64();
    this.mlpHash = reader.readU64();
    this.outputHash = reader.readU64();
    this.modelHash = reader.readU64();

    expectCount(embeddingCount, VOCAB_SIZE * this.dModel, "embedding count");
    expectCount(positionEmbeddingCount, this.contextSeqLen * this.dModel, "position embedding count");
    const attentionWeightsPerLayer = this.dModel * this.dModel;
    const upWeightsPerLayer = this.dModel * this.hiddenDim;
    const downWeightsPerLayer = this.hiddenDim * this.dModel;
    this.layers = exactLayerCount(qCount, attentionWeightsPerLayer, "q count");
    expectCount(kCount, this.layers * attentionWeightsPerLayer, "k count");
    expectCount(vCount, this.layers * attentionWeightsPerLayer, "v count");
    expectCount(oCount, this.layers * attentionWeightsPerLayer, "o count");
    expectCount(upCount, this.layers * upWeightsPerLayer, "up count");
    expectCount(gateCount, this.layers * upWeightsPerLayer, "gate count");
    expectCount(downCount, this.layers * downWeightsPerLayer, "down count");
    expectCount(outputCount, VOCAB_SIZE * this.dModel, "output count");

    this.embeddings = reader.readI16Array(embeddingCount);
    this.positionEmbeddings = reader.readI16Array(positionEmbeddingCount);
    this.qWeights = reader.readI8Array(qCount);
    this.kWeights = reader.readI8Array(kCount);
    this.vWeights = reader.readI8Array(vCount);
    this.oWeights = reader.readI8Array(oCount);
    this.upWeights = reader.readI8Array(upCount);
    this.gateWeights = reader.readI8Array(gateCount);
    this.downWeights = reader.readI8Array(downCount);
    this.outputWeights = reader.readI8Array(outputCount);
    const rmsCount = this.layers * this.dModel;
    if (reader.remaining() === rmsCount * 4) {
      this.attentionRmsWeights = reader.readI16Array(rmsCount);
      this.mlpRmsWeights = reader.readI16Array(rmsCount);
    } else if (reader.remaining() === 0) {
      this.attentionRmsWeights = null;
      this.mlpRmsWeights = null;
    } else {
      throw new Error("invalid mini-transformer RMSNorm payload");
    }
    if (!reader.isDone()) {
      throw new Error("trailing bytes in mini-transformer model");
    }
  }

  forward(context, positionPolicy) {
    const seqLen = context.length;
    const total = seqLen * this.dModel;
    const embeddings = new Int16Array(total);
    for (let index = 0; index < seqLen; index += 1) {
      const token = context[index];
      const tokenStart = token * this.dModel;
      const positionStart = index * this.dModel;
      const outputStart = index * this.dModel;
      for (let dim = 0; dim < this.dModel; dim += 1) {
        const positionValue = positionPolicy === 0 ? this.positionEmbeddings[positionStart + dim] : 0;
        embeddings[outputStart + dim] = saturateI16(this.embeddings[tokenStart + dim] + positionValue);
      }
    }

    const attentionWeightsPerLayer = this.dModel * this.dModel;
    const upWeightsPerLayer = this.dModel * this.hiddenDim;
    const downWeightsPerLayer = this.hiddenDim * this.dModel;
    let layerInput = embeddings;
    for (let layer = 0; layer < this.layers; layer += 1) {
      const attentionStart = layer * attentionWeightsPerLayer;
      const attentionEnd = attentionStart + attentionWeightsPerLayer;
      const upStart = layer * upWeightsPerLayer;
      const upEnd = upStart + upWeightsPerLayer;
      const downStart = layer * downWeightsPerLayer;
      const downEnd = downStart + downWeightsPerLayer;
      const rmsStart = layer * this.dModel;
      const rmsEnd = rmsStart + this.dModel;
      const attentionInput = this.attentionRmsWeights
        ? rmsNormSequence(
          layerInput,
          this.attentionRmsWeights.subarray(rmsStart, rmsEnd),
          seqLen,
          this.dModel,
        )
        : layerInput;
      const q = linearSequence(
        attentionInput,
        this.qWeights.subarray(attentionStart, attentionEnd),
        seqLen,
        this.dModel,
        this.dModel,
        0,
      );
      const k = linearSequence(
        attentionInput,
        this.kWeights.subarray(attentionStart, attentionEnd),
        seqLen,
        this.dModel,
        this.dModel,
        0,
      );
      const v = linearSequence(
        attentionInput,
        this.vWeights.subarray(attentionStart, attentionEnd),
        seqLen,
        this.dModel,
        this.dModel,
        0,
      );
      const contextRows = causalAttention(
        q,
        k,
        v,
        seqLen,
        this.dModel,
        this.heads,
        this.headDim,
        this.attentionScaleShift,
      );
      const attentionOutput = linearSequence(
        contextRows,
        this.oWeights.subarray(attentionStart, attentionEnd),
        seqLen,
        this.dModel,
        this.dModel,
        0,
      );
      const attentionResidual = addResidual(layerInput, attentionOutput);
      const mlpInput = this.mlpRmsWeights
        ? rmsNormSequence(
          attentionResidual,
          this.mlpRmsWeights.subarray(rmsStart, rmsEnd),
          seqLen,
          this.dModel,
        )
        : attentionResidual;
      const up = linearSequence(
        mlpInput,
        this.upWeights.subarray(upStart, upEnd),
        seqLen,
        this.dModel,
        this.hiddenDim,
        0,
      );
      const gate = linearSequence(
        mlpInput,
        this.gateWeights.subarray(upStart, upEnd),
        seqLen,
        this.dModel,
        this.hiddenDim,
        0,
      );
      const gated = new Int16Array(seqLen * this.hiddenDim);
      for (let index = 0; index < gated.length; index += 1) {
        gated[index] = gatedActivation(up[index], gate[index]);
      }
      const mlpOutput = linearSequence(
        gated,
        this.downWeights.subarray(downStart, downEnd),
        seqLen,
        this.hiddenDim,
        this.dModel,
        0,
      );
      layerInput = addResidual(attentionResidual, mlpOutput);
    }
    const last = layerInput.subarray((seqLen - 1) * this.dModel, seqLen * this.dModel);
    const logits = linearRow(last, this.outputWeights, this.dModel, VOCAB_SIZE, OUTPUT_SHIFT);
    return { logits, probabilities: softmaxQ15(logits) };
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

  remaining() {
    return this.bytes.length - this.offset;
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

  readBytes(count) {
    this.require(count);
    const bytes = this.bytes.subarray(this.offset, this.offset + count);
    this.offset += count;
    return bytes;
  }

  readI8Array(count) {
    const out = new Int8Array(count);
    for (let index = 0; index < count; index += 1) {
      this.require(1);
      out[index] = this.view.getInt8(this.offset);
      this.offset += 1;
    }
    return out;
  }

  readI16Array(count) {
    const out = new Int16Array(count);
    for (let index = 0; index < count; index += 1) {
      this.require(2);
      out[index] = this.view.getInt16(this.offset, true);
      this.offset += 2;
    }
    return out;
  }

  readTextMemory(version) {
    const present = this.readU32();
    if (present === 0) {
      return null;
    }
    if (present !== 1) {
      throw new Error(`invalid NSRLLMM1 text memory marker ${present}`);
    }
    const order = this.readU32();
    const exampleCount = this.readU32();
    const examples = [];
    for (let index = 0; index < exampleCount; index += 1) {
      examples.push({
        primaryName: this.readString(),
        prompt: this.readString(),
        textTokens: Array.from(this.readU8Vec()),
        imageTokens: version >= 4 ? Array.from(this.readU8Vec()) : [],
      });
    }
    return { order, examples };
  }

  readString() {
    return new TextDecoder().decode(this.readU8Vec());
  }

  readU8Vec() {
    return this.readBytes(this.readU32());
  }

  require(count) {
    if (this.offset + count > this.bytes.length) {
      throw new Error("truncated model");
    }
  }
}

function exactLayerCount(count, perLayer, label) {
  if (perLayer <= 0 || count <= 0 || count % perLayer !== 0) {
    throw new Error(`${label} ${count} is not a positive multiple of ${perLayer}`);
  }
  return count / perLayer;
}

function powerOfFourSqrtShift(headDim) {
  let value = headDim;
  let shift = 0;
  while (value > 1 && value % 4 === 0) {
    value /= 4;
    shift += 1;
  }
  if (value !== 1) {
    throw new Error(`mini-transformer head dimension ${headDim} is not a power of four`);
  }
  return shift;
}

function causalAttention(q, k, v, seqLen, dModel, heads, headDim, scaleShift) {
  const contextRows = new Int16Array(seqLen * dModel);
  for (let token = 0; token < seqLen; token += 1) {
    for (let head = 0; head < heads; head += 1) {
      const headOffset = head * headDim;
      const logits = new Int32Array(token + 1);
      for (let keyIndex = 0; keyIndex <= token; keyIndex += 1) {
        let acc = 0;
        const qStart = token * dModel + headOffset;
        const kStart = keyIndex * dModel + headOffset;
        for (let dim = 0; dim < headDim; dim += 1) {
          acc += q[qStart + dim] * k[kStart + dim];
        }
        logits[keyIndex] = floorShift(acc, scaleShift);
      }
      const probabilities = softmaxQ15(logits);
      const ctxStart = token * dModel + headOffset;
      for (let dim = 0; dim < headDim; dim += 1) {
        let acc = 0;
        for (let keyIndex = 0; keyIndex <= token; keyIndex += 1) {
          acc += probabilities[keyIndex] * v[keyIndex * dModel + headOffset + dim];
        }
        contextRows[ctxStart + dim] = saturateI16(roundShift(acc, Q15_SHIFT));
      }
    }
  }
  return contextRows;
}

function rmsNormSequence(input, weights, seqLen, dModel) {
  const out = new Int16Array(input.length);
  for (let row = 0; row < seqLen; row += 1) {
    const start = row * dModel;
    let sumSquares = 0;
    for (let dim = 0; dim < dModel; dim += 1) {
      const value = input[start + dim];
      sumSquares += value * value;
    }
    const meanSquares = Math.floor(sumSquares / dModel) + RMS_NORM_EPSILON;
    const inverseRmsQ30 = integerRsqrtQ30(meanSquares);
    for (let dim = 0; dim < dModel; dim += 1) {
      const scaled = input[start + dim] * weights[dim] * inverseRmsQ30;
      out[start + dim] = saturateI16(roundShift(scaled, 30));
    }
  }
  return out;
}

function integerRsqrtQ30(value) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error("invalid RMSNorm magnitude");
  }
  const msb = Math.floor(Math.log2(value));
  let mantissa;
  let exponent;
  if (msb >= 7) {
    exponent = msb - 7;
    mantissa = Math.floor(value / 2 ** exponent);
  } else {
    exponent = -(7 - msb);
    mantissa = value * 2 ** -exponent;
  }
  let inverse = RSQRT_LUT_8BIT[mantissa - 128] * 2 ** 15;
  const scaledExponent = exponent + 7;
  if (scaledExponent % 2 !== 0) {
    inverse = roundShift(inverse * INV_SQRT_2_Q15, 15);
  }
  const halfExponent = Math.floor(scaledExponent / 2);
  return halfExponent >= 0
    ? Math.floor(inverse / 2 ** halfExponent)
    : inverse * 2 ** -halfExponent;
}

function linearSequence(input, weights, seqLen, inputDim, outputDim, shift) {
  const out = new Int16Array(seqLen * outputDim);
  for (let row = 0; row < seqLen; row += 1) {
    const inputRow = input.subarray(row * inputDim, (row + 1) * inputDim);
    out.set(linearRow(inputRow, weights, inputDim, outputDim, shift), row * outputDim);
  }
  return out;
}

function linearRow(input, weights, inputDim, outputDim, shift) {
  const out = new Int16Array(outputDim);
  for (let outIndex = 0; outIndex < outputDim; outIndex += 1) {
    let acc = 0;
    const weightStart = outIndex * inputDim;
    for (let inIndex = 0; inIndex < inputDim; inIndex += 1) {
      acc += input[inIndex] * weights[weightStart + inIndex];
    }
    out[outIndex] = saturateI16(roundShift(acc, shift));
  }
  return out;
}

function addResidual(left, right) {
  const out = new Int16Array(left.length);
  for (let index = 0; index < out.length; index += 1) {
    out[index] = saturateI16(left[index] + right[index]);
  }
  return out;
}

function gatedActivation(up, gate) {
  const gateQ15 = clamp((gate >> 2) + 16384, 0, 32767);
  const silu = saturateI16(roundShift(gate * gateQ15, 15));
  return saturateI16(roundShift(up * silu, 15));
}

function softmaxQ15(logits) {
  let max = -Infinity;
  for (const logit of logits) {
    if (logit > max) {
      max = logit;
    }
  }
  const weights = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const weight = Math.pow(2, (logits[index] - max) / 256);
    weights[index] = weight;
    total += weight;
  }
  const out = new Int16Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    out[index] = clamp(Math.round((weights[index] / total) * 32767), 0, 32767);
  }
  return out;
}

function chooseCandidate(candidates, seed, step, topK) {
  sortCandidates(candidates);
  const limit = Math.max(1, Math.min(topK || 1, candidates.length));
  if (limit <= 1) {
    return candidates[0];
  }
  const total = candidates
    .slice(0, limit)
    .reduce((sum, candidate) => sum + Math.max(0, candidate.probability), 0);
  if (total <= 0) {
    return candidates[mix32(seed ^ step) % limit];
  }
  let draw = mix32(seed ^ step) % total;
  for (let index = 0; index < limit; index += 1) {
    const candidate = candidates[index];
    if (draw < candidate.probability) {
      return candidate;
    }
    draw -= candidate.probability;
  }
  return candidates[0];
}

function sortedAllowedCandidates(row, phase, allowStop, textTokenProfile) {
  const candidates = [];
  for (let token = 0; token < VOCAB_SIZE; token += 1) {
    if (allowedToken(token, phase, allowStop, textTokenProfile)) {
      candidates.push({
        token,
        logit: row.logits[token],
        probability: row.probabilities[token],
      });
    }
  }
  sortCandidates(candidates);
  return candidates;
}

function sortCandidates(candidates) {
  candidates.sort(
    (left, right) =>
      right.logit - left.logit || right.probability - left.probability || left.token - right.token,
  );
}

function allowedToken(token, phase, allowStop, textTokenProfile) {
  if (phase === "text") {
    return isTextToken(token, textTokenProfile) || (allowStop && (token === IMAGE || token === EOS));
  }
  return isImageToken(token);
}

function formatToken(token) {
  if (token === undefined) {
    return "";
  }
  if (isTextChunkToken(token)) {
    return TEXT_CHUNKS[token - TEXT_CHUNK_BASE];
  }
  if (isAsciiTextToken(token)) {
    return String.fromCharCode(token - TEXT_BASE);
  }
  if (token === IMAGE) {
    return "<IMAGE>";
  }
  if (token === EOS) {
    return "<EOS>";
  }
  if (token === TEXT) {
    return "<TEXT>";
  }
  if (token === PROMPT) {
    return "<PROMPT>";
  }
  if (token === BOS) {
    return "<BOS>";
  }
  if (token === PAD) {
    return "<PAD>";
  }
  return `<${token}>`;
}

function isAsciiTextToken(token) {
  return token >= TEXT_BASE + 32 && token <= TEXT_BASE + 126;
}

function isTextChunkToken(token) {
  return token >= TEXT_CHUNK_BASE && token < TEXT_CHUNK_BASE + TEXT_CHUNKS.length;
}

function isNameTextChunkToken(token) {
  return isTextChunkToken(token) && token - TEXT_CHUNK_BASE >= TEXT_CHUNK_NAME_START;
}

function isTextToken(token, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  return (
    isAsciiTextToken(token) ||
    (textTokenProfile === TEXT_TOKEN_PROFILE_CHUNKED && isTextChunkToken(token))
  );
}

function isImageToken(token) {
  return token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS;
}

function encodeTextTokens(text, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  return encodeNormalizedTextTokens(normalizeText(text), textTokenProfile);
}

function encodeTextPrefixTokens(text, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  return encodeNormalizedTextTokens(normalizeTextPreservingEdgeSpace(text), textTokenProfile);
}

function encodeNormalizedTextTokens(text, textTokenProfile) {
  const tokens = [];
  for (let index = 0; index < text.length; ) {
    if (textTokenProfile === TEXT_TOKEN_PROFILE_CHUNKED) {
      const match = matchTextChunk(text, index);
      if (match) {
        tokens.push(TEXT_CHUNK_BASE + match.chunkIndex);
        index += match.chunk.length;
        continue;
      }
    }
    tokens.push(TEXT_BASE + Math.min(127, text.charCodeAt(index)));
    index += 1;
  }
  return tokens;
}

function matchTextChunk(text, index) {
  let best = null;
  for (let chunkIndex = 0; chunkIndex < TEXT_CHUNKS.length; chunkIndex += 1) {
    const chunk = TEXT_CHUNKS[chunkIndex];
    if (!text.startsWith(chunk, index)) {
      continue;
    }
    if (!best || chunk.length > best.chunk.length) {
      best = { chunkIndex, chunk };
    }
  }
  return best;
}

function decodeTextTokens(tokens, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  return normalizeText(
    tokens
      .filter((token) => isTextToken(token, textTokenProfile))
      .map((token) =>
        isTextChunkToken(token)
          ? TEXT_CHUNKS[token - TEXT_CHUNK_BASE]
          : String.fromCharCode(token - TEXT_BASE),
      )
      .join(""),
  );
}

function embeddedTextTokens(example, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  const text = decodeTextTokens(example?.textTokens || [], textTokenProfile);
  return encodeTextTokens(text, textTokenProfile);
}

function shouldUseMemoryTextFallback(generatedTextTokens, memoryTextTokens, maxTextTokens) {
  if (memoryTextTokens.length === 0) {
    return false;
  }
  const generatedText = decodeTextTokens(generatedTextTokens, TEXT_TOKEN_PROFILE_CHUNKED);
  if (generatedText.length < 24) {
    return true;
  }
  if (isWeakGeneratedText(generatedText)) {
    return true;
  }
  if (generatedTextTokens.length >= maxTextTokens && !endsWithSentence(generatedText)) {
    return true;
  }
  return hasRepeatedWordNgram(generatedText, 4);
}

function isWeakGeneratedText(text) {
  const body = text.replace(/^Solomon selects\s*/i, "").trim();
  if (body.length < 16) {
    return true;
  }
  const letters = body.match(/[A-Za-z]/g) || [];
  const words = body.match(/[A-Za-z][A-Za-z']{2,}/g) || [];
  const alphaRatio = letters.length / Math.max(1, body.length);
  const wordChars = words.reduce((total, word) => total + word.length, 0);
  const wordlikeRatio = wordChars / Math.max(1, body.length);
  if (wordlikeRatio < 0.28 && body.length >= 32) {
    return true;
  }
  if (alphaRatio > 0.8 && wordlikeRatio < 0.18) {
    return true;
  }
  if (/(.)\1{4,}/.test(body)) {
    return true;
  }
  if (hasDominantWord(body, 0.45)) {
    return true;
  }
  return hasGluedRepeatedPattern(body);
}

function endsWithSentence(text) {
  return /[.!?]"?$/.test(text.trim());
}

function hasDominantWord(text, maxRatio) {
  const words = text.toLowerCase().match(/[a-z']{2,}/g) || [];
  if (words.length < 4) {
    return false;
  }
  const counts = new Map();
  for (const word of words) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  let max = 0;
  for (const count of counts.values()) {
    max = Math.max(max, count);
  }
  return max / words.length > maxRatio;
}

function hasGluedRepeatedPattern(text) {
  const compact = text.toLowerCase().replace(/[^a-z]+/g, "");
  if (compact.length < 16) {
    return false;
  }
  for (let size = 2; size <= 5; size += 1) {
    for (let index = 0; index + size * 3 <= compact.length; index += 1) {
      const piece = compact.slice(index, index + size);
      if (
        compact.slice(index + size, index + size * 2) === piece &&
        compact.slice(index + size * 2, index + size * 3) === piece
      ) {
        return true;
      }
    }
  }
  return false;
}

function hasRepeatedWordNgram(text, size) {
  const words = (text.toLowerCase().match(/[a-z']{2,}/g) || []).filter(
    (word) => word !== "thee",
  );
  const seen = new Set();
  for (let index = 0; index + size <= words.length; index += 1) {
    const key = words.slice(index, index + size).join(" ");
    if (seen.has(key)) {
      return true;
    }
    seen.add(key);
  }
  return false;
}

function buildEmbeddedTextPrior(example, textTokenProfile, order, minOrder) {
  const textTokens = (example?.textTokens || []).filter((token) =>
    isTextToken(token, textTokenProfile),
  );
  const transitions = new Map();
  const startTokens = textPrefixThroughColon(textTokens);
  if (textTokens.length > 0) {
    addTransitionCount(transitions, [TEXT], textTokens[0]);
  }
  for (let index = 0; index <= textTokens.length; index += 1) {
    const target = index === textTokens.length ? IMAGE : textTokens[index];
    const maxOrder = Math.min(order, index);
    for (let contextOrder = 0; contextOrder <= maxOrder; contextOrder += 1) {
      addTransitionCount(transitions, textTokens.slice(index - contextOrder, index), target);
    }
  }
  return {
    order,
    minOrder: Math.min(order, minOrder),
    startTokens,
    transitions,
  };
}

function addTransitionCount(transitions, context, target) {
  const key = context.join(",");
  let counts = transitions.get(key);
  if (!counts) {
    counts = new Uint32Array(VOCAB_SIZE);
    transitions.set(key, counts);
  }
  counts[target] = Math.min(0xffffffff, counts[target] + 1);
}

function applyEmbeddedTextPrior(candidates, textContext, prior) {
  if (candidates.length === 0) {
    return false;
  }
  if (textContext.length < prior.startTokens.length) {
    const expected = prior.startTokens[textContext.length];
    if (candidates.some((candidate) => candidate.token === expected)) {
      candidates.splice(
        0,
        candidates.length,
        ...candidates.filter((candidate) => candidate.token === expected),
      );
      return true;
    }
  }
  if (textContext.length === 0) {
    const counts = prior.transitions.get(String(TEXT));
    if (applyPriorCounts(candidates, counts)) {
      return true;
    }
  }
  const maxOrder = Math.min(prior.order, textContext.length);
  const minOrder = Math.min(prior.minOrder, maxOrder);
  for (let order = maxOrder; order >= minOrder; order -= 1) {
    const key = textContext.slice(textContext.length - order).join(",");
    if (applyPriorCounts(candidates, prior.transitions.get(key))) {
      return true;
    }
  }
  return false;
}

function applyPriorCounts(candidates, counts) {
  if (!counts) {
    return false;
  }
  const matched = candidates.filter((candidate) => counts[candidate.token] > 0);
  if (matched.length === 0) {
    return false;
  }
  candidates.splice(0, candidates.length, ...matched);
  for (const candidate of candidates) {
    const countBoost = Math.min(4096, counts[candidate.token]) * 256;
    candidate.logit += 1_000_000 + countBoost;
    candidate.probability += 1_000_000 + countBoost;
  }
  return true;
}

function suppressNameChunkCandidates(candidates) {
  const filtered = candidates.filter((candidate) => !isNameTextChunkToken(candidate.token));
  if (filtered.length > 0) {
    candidates.splice(0, candidates.length, ...filtered);
    return;
  }
  candidates.splice(0, candidates.length, {
    token: TEXT_BASE + 32,
    logit: Number.NEGATIVE_INFINITY,
    probability: 0,
  });
}

function generatedTextContext(history, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  const textStart = history.lastIndexOf(TEXT);
  const start = textStart >= 0 ? textStart + 1 : history.length;
  const out = [];
  for (let index = start; index < history.length; index += 1) {
    const token = history[index];
    if (token === IMAGE || token === EOS) {
      break;
    }
    if (isTextToken(token, textTokenProfile)) {
      out.push(token);
    }
  }
  return out;
}

function generatedTextIsAfterOpening(history, textTokenProfile = TEXT_TOKEN_PROFILE_CHAR) {
  const textTokens = generatedTextContext(history, textTokenProfile);
  for (let index = 0; index + 1 < textTokens.length; index += 1) {
    if (textTokens[index] === TEXT_CHUNK_BASE + 1 && textTokens[index + 1] === TEXT_CHUNK_BASE + 2) {
      return true;
    }
  }
  return false;
}

function textPrefixThroughColon(textTokens) {
  const chunkedColonIndex = textTokens.indexOf(TEXT_CHUNK_BASE + 1);
  if (chunkedColonIndex >= 0) {
    return textTokens.slice(0, chunkedColonIndex + 1);
  }
  const colonIndex = textTokens.indexOf(TEXT_BASE + 58);
  if (colonIndex < 0) {
    return [];
  }
  let end = colonIndex + 1;
  if (textTokens[end] === TEXT_BASE + 32) {
    end += 1;
  }
  return textTokens.slice(0, end);
}

function paddedContext(history, contextSeqLen) {
  const contextLen = Math.min(contextSeqLen, history.length);
  const out = new Uint8Array(contextSeqLen);
  out.fill(PAD, 0, contextSeqLen - contextLen);
  out.set(history.slice(history.length - contextLen), contextSeqLen - contextLen);
  return Array.from(out);
}

function textMemoryPromptScopeScore(promptKey, example) {
  if (promptKey === "king solomon seal") {
    return normalizeKey(example.prompt) === promptKey ? 1000 : 1;
  }
  if (promptKey === normalizeKey(example.prompt)) {
    return 1_000_000;
  }
  const primaryKey = normalizeKey(example.primaryName);
  if (primaryKey && promptContainsPhrase(promptKey, primaryKey)) {
    return 100_000;
  }
  return 0;
}

function promptContainsPhrase(promptKey, phrase) {
  return (
    promptKey === phrase ||
    promptKey.split(/\s+/).includes(phrase) ||
    promptKey.includes(` ${phrase} `) ||
    promptKey.startsWith(`${phrase} `) ||
    promptKey.endsWith(` ${phrase}`)
  );
}

function normalizeText(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/[^\x20-\x7e]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeTextPreservingEdgeSpace(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/[^\x20-\x7e]+/g, " ")
    .replace(/\s+/g, " ");
}

function normalizeKey(value) {
  return normalizeText(value).toLowerCase();
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
        rgba[offset + channel] = Math.floor(
          (paper[channel] * (255 - alpha) + ink[channel] * alpha) / 255,
        );
      }
      rgba[offset + 3] = 255;
    }
  }
  return rgba;
}

function expectU32(reader, expected, label) {
  const actual = reader.readU32();
  if (actual !== expected) {
    throw new Error(`NSRLLMM1 ${label} mismatch: expected ${expected}, got ${actual}`);
  }
}

function expectCount(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`mini-transformer ${label} mismatch: expected ${expected}, got ${actual}`);
  }
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

function fnv64Hex(tokens) {
  return hex64(hashBytes(Uint8Array.from(tokens)));
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
  value >>>= 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}

function roundShift(value, shift) {
  if (shift === 0) {
    return value;
  }
  return floorShift(value + 2 ** (shift - 1), shift);
}

function floorShift(value, shift) {
  return Math.floor(value / 2 ** shift);
}

function saturateI16(value) {
  return clamp(value, -32768, 32767);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
