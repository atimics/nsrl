#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {sha256Bytes, sha256Json} from "./lib/solomon-council-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelPath = "benchmarks/integer-transformer-proof-v1/successor-v2-candidate.nsrlmt";
const gutenbergFramePath =
  "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const unfamiliarFramePath =
  "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-source-frame.json";
const sealSourcePath = "web/assets/solomon-spirit-text-signatures.tsv";
const defaultEvidenceDir = "benchmarks/solomon-council-v0/production-v0/evidence";
const dimensions = [
  "source_grounded_correctness",
  "calibration",
  "hard_negative_rejection",
  "contradiction_detection",
  "decision_regret",
  "appropriate_abstention",
  "cross_modal_agreement",
  "unfamiliar_source_transfer",
];
const decisionIds = ["accept", "reject", "abstain"];

const config = parseArgs(process.argv.slice(2));
const readJson = (relative) => JSON.parse(fs.readFileSync(path.join(root, relative)));
const artifact = (relative) => ({
  path: relative,
  sha256: sha256Bytes(fs.readFileSync(path.join(root, relative))),
});
const gutenbergBinding = artifact(gutenbergFramePath);
const unfamiliarBinding = artifact(unfamiliarFramePath);
const sealBinding = artifact(sealSourcePath);
const gutenberg = readJson(gutenbergFramePath).sources;
const unfamiliar = readJson(unfamiliarFramePath).sources;
const seals = readSeals(path.join(root, sealSourcePath));
if (gutenberg.length !== 71 || unfamiliar.length !== 6 || seals.length !== 72) {
  throw new Error("wisdom casebook source populations changed");
}

const recordsByDimension = Object.fromEntries(dimensions.map((dimension) => [dimension, []]));
const draftCases = [];
for (const dimension of dimensions) {
  for (let index = 0; index < 72; index += 1) {
    const prepared = prepareCase(dimension, index);
    recordsByDimension[dimension].push(prepared.evidence);
    draftCases.push({...prepared.publicCase, nonce: crypto.randomBytes(32).toString("hex"), gold: prepared.gold});
  }
}

const evidenceDir = path.resolve(root, config.evidenceDir);
fs.mkdirSync(evidenceDir, {recursive: true});
const evidenceBindings = new Map();
for (const dimension of dimensions) {
  const records = recordsByDimension[dimension].map((record) => ({
    schema: "nsrl.solomon_wisdom_evidence.v0",
    analysis_role: "public_evidence_no_gold",
    dimension,
    verification_contract: "deterministic-public-evidence-v0",
    ...record,
  }));
  records.forEach(rejectGoldFields);
  const relative = path.join(config.evidenceDir, `${dimension}.json`);
  const absolute = path.join(root, relative);
  fs.writeFileSync(absolute, records.map((value) => JSON.stringify({
    artifact_id: value.episode_id,
    value,
  })).join("\n") + "\n");
  const bundleBinding = artifact(relative);
  evidenceBindings.set(dimension, {
    ...bundleBinding,
    records: new Map(records.map((value) => [value.episode_id, sha256Json(value)])),
  });
}
for (const entry of draftCases) {
  const bundle = evidenceBindings.get(entry.dimension);
  entry.evidence = [{
    path: bundle.path,
    sha256: bundle.sha256,
    record_id: entry.episode_id,
    record_sha256: bundle.records.get(entry.episode_id),
  }];
}

const draft = {
  schema: "nsrl.solomon_wisdom_casebook_draft.v0",
  analysis_role: "frozen_same_model_comparison",
  ceremony_id: "solomon-wisdom-production-v0",
  minimum_cases_per_dimension: 72,
  underlying_model: {
    model_id: "native-successor-v2-canonical-nll",
    artifact: artifact(modelPath),
  },
  integrity_policy: {
    no_oracle_target_lookup: true,
    no_hidden_memory: true,
    no_retrieval_target_leakage: true,
    gold_commitment_algorithm: "sha256-canonical-json-v0",
  },
  cases: draftCases,
};
fs.mkdirSync(path.dirname(config.draft), {recursive: true});
fs.writeFileSync(config.draft, `${JSON.stringify(draft, null, 2)}\n`, {mode: 0o600});
fs.chmodSync(config.draft, 0o600);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_casebook_preparation.v0",
  draft_path: config.draft,
  draft_sha256: sha256Bytes(fs.readFileSync(config.draft)),
  evidence_directory: path.relative(root, evidenceDir),
  evidence_bundle_sha256: Object.fromEntries([...evidenceBindings].map(
    ([dimension, binding]) => [dimension, binding.sha256])),
  cases: draftCases.length,
  cases_per_dimension: 72,
  private_mode: (fs.statSync(config.draft).mode & 0o777).toString(8),
  contains_public_gold: false,
}, null, 2)}\n`);

function parseArgs(args) {
  let draft = "";
  let evidenceDir = defaultEvidenceDir;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--draft") draft = path.resolve(args[++index] || "");
    else if (args[index] === "--evidence-dir") evidenceDir = args[++index] || "";
    else throw new Error(`unknown argument ${args[index]}`);
  }
  if (!draft) {
    throw new Error("Usage: node scripts/prepare-solomon-wisdom-casebook-v0.mjs --draft PRIVATE.json [--evidence-dir DIR]");
  }
  if (!evidenceDir || path.isAbsolute(evidenceDir) || evidenceDir.split(/[\\/]/).includes("..")) {
    throw new Error("--evidence-dir must be a repository-relative path");
  }
  return {draft, evidenceDir};
}

function prepareCase(dimension, index) {
  const number = String(index + 1).padStart(3, "0");
  const episodeId = `wisdom-v0-${dimension}-${number}`;
  let evidence;
  let expectedLabel;
  let shouldAbstain = false;
  let sourceFamily = "gutenberg";
  let question;
  let costs;
  if (["source_grounded_correctness", "calibration", "hard_negative_rejection"].includes(dimension)) {
    const source = gutenberg[index % gutenberg.length];
    const other = gutenberg[(index + 1) % gutenberg.length];
    expectedLabel = dimension === "hard_negative_rejection" ? false : index % 2 === 0;
    const claimedValue = expectedLabel ? source.title : other.title;
    evidence = metadataEvidence(episodeId, source, "title", claimedValue, gutenbergBinding);
    question = `Does the sealed source record support that ${source.source_id} has title ${JSON.stringify(claimedValue)}?`;
    costs = labelCosts(expectedLabel);
  } else if (dimension === "contradiction_detection") {
    const source = gutenberg[index % gutenberg.length];
    const other = gutenberg[(index + 1) % gutenberg.length];
    expectedLabel = false;
    evidence = {
      episode_id: episodeId,
      kind: "claim_set",
      parent_source: gutenbergBinding,
      claims: [
        {claim_id: "claim-a", subject: source.source_id, predicate: "title", value: source.title},
        {claim_id: "claim-b", subject: source.source_id, predicate: "title", value: other.title},
      ],
      verification_rule: "same-subject-predicate-requires-single-value-v0",
    };
    question = `Can both sealed title claims about ${source.source_id} be jointly true?`;
    costs = labelCosts(false);
  } else if (dimension === "decision_regret") {
    const preferred = decisionIds[index % decisionIds.length];
    const publicCosts = preferred === "accept"
      ? {accept: 100, reject: 700, abstain: 500}
      : preferred === "reject"
        ? {accept: 800, reject: 100, abstain: 500}
        : {accept: 800, reject: 700, abstain: 100};
    evidence = {
      episode_id: episodeId,
      kind: "consequence_ledger",
      actions: decisionIds.map((actionId) => ({
        action_id: actionId,
        fixed_cost_milli: publicCosts[actionId],
        event_probability_milli: 0,
        event_impact_milli: 0,
      })),
      verification_rule: "fixed-plus-floor-probability-times-impact-v0",
    };
    expectedLabel = preferred === "accept";
    shouldAbstain = preferred === "abstain";
    question = "Which sealed action minimizes expected cost under the public consequence ledger?";
    costs = publicCosts;
  } else if (dimension === "appropriate_abstention") {
    const source = gutenberg[index % gutenberg.length];
    evidence = {
      episode_id: episodeId,
      kind: "incomplete_source_record",
      parent_source: gutenbergBinding,
      source_id: source.source_id,
      requested_field: "publication_date",
      present_fields: {title: source.title, author: source.author},
      verification_rule: "requested-field-must-be-present-v0",
    };
    expectedLabel = false;
    shouldAbstain = true;
    question = `What publication date is supported for ${source.source_id}? Abstain if the sealed evidence does not contain it.`;
    costs = {accept: 1000, reject: 1000, abstain: 0};
  } else if (dimension === "cross_modal_agreement") {
    const claimed = seals[index];
    const observed = index % 2 === 0 ? claimed : seals[(index + 1) % seals.length];
    const claimedHash = sha256Bytes(Buffer.from(claimed.signature));
    const observedHash = sha256Bytes(Buffer.from(observed.signature));
    expectedLabel = claimedHash === observedHash;
    sourceFamily = "goetia_seal";
    evidence = {
      episode_id: episodeId,
      kind: "text_image_binding",
      parent_source: sealBinding,
      claimed_name: claimed.name,
      claimed_signature_sha256: claimedHash,
      observed_signature_sha256: observedHash,
      observed_signature_u8_16x16: observed.signature,
      verification_rule: "sha256-exact-signature-agreement-v0",
    };
    question = `Does the observed 16x16 seal agree with the canonical sealed signature for ${claimed.name}?`;
    costs = labelCosts(expectedLabel);
  } else if (dimension === "unfamiliar_source_transfer") {
    const source = unfamiliar[index % unfamiliar.length];
    const other = unfamiliar[(index + 1) % unfamiliar.length];
    expectedLabel = index % 2 === 0;
    const claimedValue = expectedLabel ? source.creator : other.creator;
    sourceFamily = source.family;
    evidence = metadataEvidence(episodeId, source, "creator", claimedValue, unfamiliarBinding);
    question = `Does the sealed unfamiliar-source record support creator ${JSON.stringify(claimedValue)} for ${source.source_id}?`;
    costs = labelCosts(expectedLabel);
  } else {
    throw new Error(`unknown wisdom dimension ${dimension}`);
  }
  return {
    evidence,
    publicCase: {
      episode_id: episodeId,
      dimension,
      source_family: sourceFamily,
      unfamiliar_source: dimension === "unfamiliar_source_transfer",
      question,
      evidence: [],
      decision_ids: decisionIds,
    },
    gold: {expected_label: expectedLabel, should_abstain: shouldAbstain, decision_costs_milli: costs},
  };
}

function metadataEvidence(episodeId, source, field, claimedValue, parentSource) {
  return {
    episode_id: episodeId,
    kind: "sealed_metadata_claim",
    parent_source: parentSource,
    source_id: source.source_id,
    source_family: source.family || "gutenberg",
    field,
    observed_value: source[field],
    claimed_value: claimedValue,
    verification_rule: "canonical-json-field-equality-v0",
  };
}

function labelCosts(expected) {
  return expected
    ? {accept: 0, reject: 1000, abstain: 600}
    : {accept: 1000, reject: 0, abstain: 600};
}

function readSeals(absolute) {
  return fs.readFileSync(absolute, "utf8").trimEnd().split("\n").slice(1).map((line) => {
    const columns = line.split("\t");
    return {name: columns[1], signature: columns[7].split(",").map(Number)};
  });
}

function rejectGoldFields(value) {
  const forbidden = new Set([
    "gold", "expected_label", "should_abstain", "decision_costs_milli", "nonce",
  ]);
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    for (const [key, child] of Object.entries(node)) {
      if (forbidden.has(key)) throw new Error(`public evidence exposed forbidden field ${key}`);
      visit(child);
    }
  };
  visit(value);
}
