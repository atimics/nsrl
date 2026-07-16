import {MiniTransformerModel} from "../../web/attention-sampler.js";

const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;

export const NATIVE_JUDGMENT_CANDIDATES = Object.freeze([
  Object.freeze({candidate_id: "accept", text: "accept"}),
  Object.freeze({candidate_id: "reject", text: "reject"}),
]);

export class NativeJudgmentScorer {
  constructor(modelBytes) {
    this.model = new MiniTransformerModel(modelBytes);
    this.positionPolicy = 1;
    this.contextIndependent = isContextIndependentModel(this.model);
    this.constantRow = this.contextIndependent
      ? this.model.forward(Array(this.model.contextSeqLen).fill(0), this.positionPolicy)
      : null;
  }

  score(prompt, candidates = NATIVE_JUDGMENT_CANDIDATES) {
    const promptBytes = Array.from(new TextEncoder().encode(String(prompt)));
    const normalized = requireCandidates(candidates);
    const scores = normalized.map((candidate) => scoreCandidate(
      this.model,
      this.positionPolicy,
      promptBytes,
      candidate,
      this.constantRow,
    ));
    scores.sort((left, right) =>
      right.mean_log_probability_microunits - left.mean_log_probability_microunits
      || left.candidate_id.localeCompare(right.candidate_id));
    const winner = scores[0];
    const runnerUp = scores[1];
    const margin = winner.mean_log_probability_microunits
      - runnerUp.mean_log_probability_microunits;
    return {
      schema: "nsrl.solomon_native_judgment_score.v0",
      selected_candidate_id: winner.candidate_id,
      selected_text: winner.text,
      confidence_milli: confidenceFromMargin(margin),
      margin_microunits: margin,
      scores,
      conditioning: {
        tokenizer: "byte_identity_u8_v1",
        prompt_bytes: promptBytes.length,
        prompt_sha256_unavailable_in_model_runner: true,
        context_seq_len: this.model.contextSeqLen,
        truncated_prompt_bytes: Math.max(0, promptBytes.length - this.model.contextSeqLen),
        visible_context_hash: fnv64Hex(paddedContext(promptBytes, this.model.contextSeqLen)),
      },
      provenance: {
        model_kind: "NSRLMT5",
        model_hash: hex64(this.model.modelHash),
        raw_transformer_only: true,
        context_independence_proven_from_weights: this.contextIndependent,
        context_independent_forward_cache_used: this.contextIndependent,
        suffix_memory_present: false,
        hidden_memory_used: false,
        retrieval_used: false,
        routing_oracle_used: false,
        oracle_or_target_lookup_used: false,
      },
    };
  }
}

function requireCandidates(candidates) {
  if (!Array.isArray(candidates) || candidates.length < 2) {
    throw new Error("native judgment scoring requires at least two candidates");
  }
  const ids = new Set();
  return candidates.map((candidate, index) => {
    const candidateId = String(candidate?.candidate_id || "").trim();
    const text = String(candidate?.text || "");
    const tokens = Array.from(new TextEncoder().encode(text));
    if (!candidateId || ids.has(candidateId)) {
      throw new Error(`native judgment candidate ${index} has a missing or duplicate id`);
    }
    if (tokens.length === 0) {
      throw new Error(`native judgment candidate ${candidateId} has no bytes`);
    }
    ids.add(candidateId);
    return {candidate_id: candidateId, text, tokens};
  });
}

function scoreCandidate(model, positionPolicy, promptBytes, candidate, constantRow) {
  const history = promptBytes.slice();
  let logProbability = 0;
  let logitSum = 0;
  const tokenRows = [];
  for (const token of candidate.tokens) {
    const context = paddedContext(history, model.contextSeqLen);
    const row = constantRow || model.forward(context, positionPolicy);
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
  return {
    candidate_id: candidate.candidate_id,
    text: candidate.text,
    token_count: candidate.tokens.length,
    token_hash: fnv64Hex(candidate.tokens),
    logit_sum_q8: logitSum,
    log_probability_microunits: Math.round(logProbability * 1_000_000),
    mean_log_probability_microunits: Math.round(
      logProbability * 1_000_000 / candidate.tokens.length,
    ),
    zero_probability_tokens_q15: tokenRows.filter((row) => row.probability_q15 === 0).length,
    token_rows: tokenRows,
  };
}

function isContextIndependentModel(model) {
  const allZero = (values) => values.every((value) => value === 0);
  if (!allZero(model.positionEmbeddings)
      || !allZero(model.qWeights)
      || !allZero(model.kWeights)
      || !allZero(model.vWeights)
      || !allZero(model.oWeights)
      || !allZero(model.upWeights)
      || !allZero(model.gateWeights)
      || !allZero(model.downWeights)
      || model.attentionRmsWeights !== null
      || model.mlpRmsWeights !== null) {
    return false;
  }
  const first = model.embeddings.subarray(0, model.dModel);
  for (let token = 1; token < 256; token += 1) {
    const row = model.embeddings.subarray(token * model.dModel, (token + 1) * model.dModel);
    for (let index = 0; index < model.dModel; index += 1) {
      if (row[index] !== first[index]) return false;
    }
  }
  return true;
}

function paddedContext(history, contextSeqLen) {
  const visible = history.slice(-contextSeqLen);
  return Array(contextSeqLen - visible.length).fill(0).concat(visible);
}

function logSoftmaxForToken(logits, token) {
  let maxLogit = -Infinity;
  for (const logit of logits) maxLogit = Math.max(maxLogit, logit);
  let normalizer = 0;
  for (const logit of logits) normalizer += Math.pow(2, (logit - maxLogit) / 256);
  return ((logits[token] - maxLogit) / 256) * Math.LN2 - Math.log(normalizer);
}

function confidenceFromMargin(marginMicrounits) {
  const margin = Math.abs(marginMicrounits) / 1_000_000;
  return Math.max(500, Math.min(1000, Math.round(1000 / (1 + Math.exp(-margin)))));
}

function fnv64Hex(bytes) {
  let hash = fnvOffset;
  for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  return hex64(hash);
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}
