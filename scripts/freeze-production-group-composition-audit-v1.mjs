#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let contractPath = "";
let auditPath = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--audit") auditPath = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !auditPath || !outPath) {
  throw new Error("--contract, --audit, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);

const contract = await json(contractPath);
assert(
  contract.schema === "nsrl.production_group_composition_audit_contract.v1",
  "unexpected group composition contract schema",
);
for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.candidate.model_path, sha256: contract.candidate.model_sha256 },
  { path: contract.bindings.tokenizer_path, sha256: contract.bindings.tokenizer_sha256 },
  { path: contract.bindings.dev_tokens_path, sha256: contract.bindings.dev_tokens_sha256 },
  { path: contract.derivation.artifact_path, sha256: contract.derivation.artifact_sha256 },
]) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const audit = await json(auditPath);
assert(audit.schema === "nsrl.production_group_composition_audit.v1", "unexpected audit schema");
assert(
  audit.bindings.source_model_hash === contract.source.model_hash
    && audit.bindings.candidate_model_hash === contract.candidate.model_hash
    && audit.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
    && audit.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
  "composition audit binding mismatch",
);
assert(
  audit.evaluation.partition === "development"
    && audit.evaluation.context_tokens === contract.bindings.context_tokens
    && audit.evaluation.windows === contract.bindings.windows,
  "composition evaluation geometry mismatch",
);
assert(audit.candidate_diff_isolated_to_groups === true, "candidate group isolation failed");
assert(audit.rows.length === contract.compositions.length, "composition row count mismatch");
for (let index = 0; index < contract.compositions.length; index += 1) {
  assert(
    audit.rows[index].id === contract.compositions[index].id
      && same(audit.rows[index].candidate_groups, contract.compositions[index].candidate_groups),
    `composition row mismatch at index ${index}`,
  );
  assert(
    audit.rows[index].residual_saturation_count <= contract.gates.residual_saturation_max,
    `composition row saturated: ${audit.rows[index].id}`,
  );
}

const byId = Object.fromEntries(audit.rows.map((row) => [row.id, row]));
assert(
  byId.source.model_hash === contract.source.model_hash
    && byId.source.total_nll_millibits === contract.source.development_total_nll_millibits
    && byId.candidate.model_hash === contract.candidate.model_hash
    && byId.candidate.total_nll_millibits
      === contract.candidate.development_total_nll_millibits,
  "source or candidate endpoint did not reproduce",
);

const individual = {};
const marginal = {};
for (const group of ["embeddings", "k", "v", "o"]) {
  individual[group] = byId[`${group}_only`].delta_from_source_millibits;
  marginal[group] = byId.candidate.total_nll_millibits
    - byId[`without_${group}`].total_nll_millibits;
}
const individuallyBeneficialGroups = Object.entries(individual)
  .filter(([, delta]) => delta < 0)
  .map(([group]) => group);
const marginallyBeneficialGroups = Object.entries(marginal)
  .filter(([, delta]) => delta < 0)
  .map(([group]) => group);
const interactionDelta = byId.candidate.delta_from_source_millibits
  - Object.values(individual).reduce((total, delta) => total + delta, 0);
const outcome = individuallyBeneficialGroups.length > 0
  ? "composition_attribution_found_individually_beneficial_group"
  : marginallyBeneficialGroups.length > 0
    ? "composition_attribution_found_only_marginally_beneficial_group"
    : "composition_attribution_found_no_beneficial_group";

const result = {
  schema: "nsrl.production_group_composition_attribution.v1",
  checked: check,
  objective: contract.objective,
  outcome,
  source_total_nll_millibits: byId.source.total_nll_millibits,
  candidate_total_nll_millibits: byId.candidate.total_nll_millibits,
  candidate_delta_millibits: byId.candidate.delta_from_source_millibits,
  individual_group_delta_millibits: individual,
  candidate_marginal_group_delta_millibits: marginal,
  interaction_delta_millibits: interactionDelta,
  individually_beneficial_groups: individuallyBeneficialGroups,
  marginally_beneficial_groups: marginallyBeneficialGroups,
  rows: audit.rows,
  gates: {
    candidate_difference_isolated_to_composed_groups: true,
    source_and_candidate_endpoints_reproduced: true,
    all_compositions_residual_saturation_zero: true,
    source_and_candidate_artifacts_unchanged: true,
    test_partition_not_read: true,
  },
  authorization: {
    read_only_development_attribution: true,
    training: false,
    candidate_checkpoint: false,
    test_evaluation: false,
    quality_promotion: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "composition attribution differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  individually_beneficial_groups: result.individually_beneficial_groups,
  marginally_beneficial_groups: result.marginally_beneficial_groups,
  out: outPath,
})}\n`);
