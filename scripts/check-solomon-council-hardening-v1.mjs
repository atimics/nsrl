#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {sha256Bytes} from "./lib/solomon-council-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultPath = "benchmarks/solomon-council-v1/hardening-result.json";
const result = JSON.parse(fs.readFileSync(path.join(root, resultPath), "utf8"));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const replay = childProcess.spawnSync(process.execPath, [
  "scripts/audit-solomon-council-hardening-v1.mjs", "--check",
], {cwd: root, encoding: "utf8", timeout: 30000});
assert(replay.status === 0,
  `Council hardening result did not replay: ${replay.stderr || replay.stdout}`);
assert(result.schema === "nsrl.solomon_council_hardening_result.v1",
  "wrong Council hardening result schema");
assert(result.historical_ceremony?.cases === 576
  && result.historical_ceremony?.v0_promotion_gate_passed === true
  && result.historical_ceremony?.v0_result_preserved_as_historical_record === true,
"historical v0 boundary changed");
for (const [name, source] of Object.entries(result.sources ?? {})) {
  assert(typeof source.path === "string" && /^[0-9a-f]{64}$/.test(source.sha256),
    `invalid source binding ${name}`);
  const bytes = fs.readFileSync(path.join(root, source.path));
  assert(bytes.length === source.bytes && sha256Bytes(bytes) === source.sha256,
    `source binding changed: ${name}`);
}
assert(result.baseline_fairness?.actual_same_model === true
  && result.baseline_fairness?.actual_same_public_casebook === true,
"same-model or same-casebook historical binding changed");
assert(result.baseline_fairness?.actual_solo_tool_observations === 0
  && result.baseline_fairness?.actual_council_tool_observations === 2880
  && result.baseline_fairness?.actual_solo_permission_budget_declarations === 0
  && result.baseline_fairness?.actual_council_permission_budget_declarations === 3456
  && result.baseline_fairness?.actual_equivalent_tool_observations === false
  && result.baseline_fairness?.actual_equivalent_tool_permissions === false
  && result.baseline_fairness?.actual_equivalent_tool_budgets === false
  && result.baseline_fairness?.actual_equivalent_tool_access === false,
"historical tool-access asymmetry changed");
assert(result.baseline_fairness?.counterfactual_tool_parity_cases === 576
  && /diagnostic_only/.test(result.baseline_fairness?.counterfactual_role ?? ""),
"post-outcome parity diagnostic boundary changed");
const dimensions = Object.values(result.dimensions ?? {});
assert(dimensions.length === 8
  && dimensions.every((dimension) => dimension.cases === 72
    && dimension.exact_tie === true
    && dimension.council_strictly_outperforms_tool_parity_solo === false),
"tool-parity baseline no longer ties Council on all eight dimensions");
assert(result.coverage?.adversarial_evidence?.stale === 0
  && result.coverage?.tool_boundaries?.tool_failures === 0
  && result.coverage?.tool_boundaries?.permission_denials === 0
  && result.coverage?.human_ambiguity?.human_authored_ambiguous_cases === 0
  && result.coverage?.outcomes?.production_receipts_with_observed_outcomes === 0
  && result.coverage?.outcomes?.production_receipts_with_calibration_revisions === 0,
"expected hardening coverage deficits changed");
assert(result.coverage?.transfer?.unfamiliar_source_cases === 72
  && result.coverage?.transfer?.unfamiliar_source_families === 3
  && result.coverage?.transfer?.cross_modal_cases === 72
  && result.coverage?.transfer?.cross_modal_source_families === 1,
"historical transfer coverage changed");
assert(result.integrity?.exact_v0_ceremony_replay === true
  && result.integrity?.generation_integrity_green === true
  && result.integrity?.provenance_green === true
  && result.integrity?.all_receipts_shadow_only === true,
"historical integrity or shadow gate changed");
assert(result.gates?.actual_tool_parity_baseline === false
  && result.gates?.strict_council_outperformance === false
  && result.gates?.all_passed === false
  && result.verdict?.status === "falsified",
"Council hardening verdict is not fail-closed");
assert(result.authorization?.historical_v0_result_rewritten === false
  && result.authorization?.effective_council_promotion_authorized === false
  && result.authorization?.operational_action_execution_authorized === false
  && result.authorization?.product_release_authorized === false
  && result.authorization?.remain_shadow_only === true,
"Council hardening authorization escaped its boundary");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_council_hardening_check.v1",
  ok: true,
  verdict: result.verdict.status,
  historical_cases: result.historical_ceremony.cases,
  actual_tool_observations: {
    solo: result.baseline_fairness.actual_solo_tool_observations,
    council: result.baseline_fairness.actual_council_tool_observations,
  },
  tool_parity_dimensions_tied: dimensions.length,
  missing_coverage: [
    "stale_evidence", "tool_failures", "permission_denials", "human_ambiguity",
    "production_outcomes", "calibration_revisions", "long_transfer",
  ],
  remain_shadow_only: true,
  byte_replay: true,
}, null, 2)}\n`);
