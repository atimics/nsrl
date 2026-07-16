#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicationPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-publication.json";
const resolve = (value) => path.isAbsolute(value) ? value : path.join(root, value);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(`adaptive publication check: ${message}`);
};
const publicationBytes = fs.readFileSync(resolve(publicationPath));
const publication = JSON.parse(publicationBytes);
assert(publication.schema === "nsrl.adaptive_composition_publication.v1",
  "wrong publication schema");
const allowed = ["supported", "falsified", "inconclusive"];
assert(JSON.stringify(publication.publication_contract.allowed_statuses) === JSON.stringify(allowed)
  && publication.publication_contract.fail_closed_on_unknown_status === true,
"publication status contract changed");
assert(allowed.includes(publication.verdict.status)
  && Number(publication.verdict.supported) + Number(publication.verdict.falsified)
    + Number(publication.verdict.inconclusive) === 1
  && publication.verdict[publication.verdict.status] === true,
"publication verdict is not exclusive and trivalent");
for (const [label, binding] of Object.entries(publication.sources)) {
  const bytes = fs.readFileSync(resolve(binding.path));
  assert(bytes.length === binding.bytes && sha256(bytes) === binding.sha256,
    `${label} source binding changed`);
}
const result = JSON.parse(fs.readFileSync(resolve(publication.sources.result.path)));
const receipt = JSON.parse(fs.readFileSync(resolve(publication.sources.replay_receipt.path)));
assert(result.verdict === "falsified" && publication.verdict.status === result.verdict,
  "publication does not preserve the frozen result verdict");
assert(result.adaptive_trajectory.accepted_actions === 0
  && Object.values(result.endpoints).every(
    (endpoint) => endpoint.total_nll_millibits === "5930001" && endpoint.final_state === "empty"),
"frozen zero-fire endpoint evidence changed");
assert(receipt.verdict === result.verdict
  && receipt.replay_artifacts.length === 9
  && receipt.replay_artifacts.every((artifact) => artifact.identical === true)
  && Object.values(receipt.guarantees).filter(Boolean).length === 4
  && receipt.guarantees.post_outcome_threshold_change === false,
"tracked replay receipt changed");
assert(publication.claims.find((claim) =>
  claim.id === "nonvacuous_persistent_adaptive_composition_improves_canonical_nll")
  ?.status === "falsified",
"nonvacuous improvement claim did not receive its falsifier");
assert(publication.claims.find((claim) =>
  claim.id === "zero_positive_regret_for_fired_actions")?.status === "inconclusive",
"vacuous zero-regret claim was overstated");
assert(Object.values(publication.interpretation).filter(
  (value) => typeof value === "boolean").every((value) => value === false),
"publication escaped its authorization boundary");
execFileSync(process.execPath, [resolve(publication.sources.publisher.path), "--check",
  publication.sources.contract.path, publication.sources.result.path,
  publication.sources.replay_receipt.path, publicationPath], {cwd: root, stdio: "pipe"});
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.adaptive_composition_publication_check.v1",
  verdict: publication.verdict.status,
  adaptive_fires: result.adaptive_trajectory.accepted_actions,
  total_nll_millibits: result.endpoints.adaptive.total_nll_millibits,
  correction_range_q32: [publication.evidence.minimum_correction_q32,
    publication.evidence.maximum_correction_q32],
  source_bindings: Object.keys(publication.sources).length,
  byte_replay: true,
  fail_closed: true,
  ok: true,
}, null, 2)}\n`);
