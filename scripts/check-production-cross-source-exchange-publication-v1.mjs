#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const publicationPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-publication.json";
const resultPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json";
const contractPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const rate = (numerator, denominator) => ({
  numerator,
  denominator,
  exact: `${numerator}/${denominator}`,
});
const classify = ({vacuous, uncovered, rejection, supportMaximum, unsafe, fired,
  signedRegret}) => vacuous ? "inconclusive"
    : uncovered < rejection && uncovered > supportMaximum ? "inconclusive"
      : uncovered >= rejection || unsafe > 0 || fired < 1 || signedRegret >= 0n
        ? "falsified" : "supported";

const publicationBytes = fs.readFileSync(publicationPath);
const resultBytes = fs.readFileSync(resultPath);
const contractBytes = fs.readFileSync(contractPath);
const publication = JSON.parse(publicationBytes);
const result = JSON.parse(resultBytes);
const contract = JSON.parse(contractBytes);
assert(publication.schema === "nsrl.production_cross_source_exchange_publication.v1"
  && publication.analysis_role
    === "deterministic_post_outcome_reporting_of_frozen_prospective_experiment",
"wrong publication schema");
assert(result.schema === "nsrl.production_cross_source_exchange_result.v1"
  && contract.schema === "nsrl.production_cross_source_exchange_contract.v1",
"wrong publication inputs");
assert(publication.source_sha256.frozen_prospective_contract === sha256(contractBytes)
  && publication.source_sha256.checked_prospective_result === sha256(resultBytes)
  && publication.source_sha256.publisher === sha256(fs.readFileSync(
    new URL("./publish-production-cross-source-exchange-v1.mjs", import.meta.url))),
"publication replay binding changed");
assert(result.source_sha256.contract === sha256(contractBytes)
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"result is not bound to the prospective contract");
assert(contract.bindings.analyzer.sha256 === sha256(fs.readFileSync(
  contract.bindings.analyzer.path))
  && contract.bindings.checker.sha256 === sha256(fs.readFileSync(
    contract.bindings.checker.path)),
"a pre-outcome bound analysis program changed");

const frame = JSON.parse(fs.readFileSync(contract.bindings.source_frame.path));
const predictor = JSON.parse(fs.readFileSync(contract.bindings.predictor.path));
const roleSets = Object.fromEntries(["fitting", "calibration", "evaluation"].map((role) => [
  role, new Set(frame.sources.filter((source) => source.role === role)
    .map((source) => source.source_id)),
]));
assert(roleSets.fitting.size === 16 && roleSets.calibration.size === 39
  && roleSets.evaluation.size === 16 && roleSets.calibration.size >= 19,
"fitting/calibration/evaluation source-panel counts changed");
assert([...roleSets.fitting].every((id) => !roleSets.calibration.has(id)
  && !roleSets.evaluation.has(id))
  && [...roleSets.calibration].every((id) => !roleSets.evaluation.has(id)),
"source roles are not disjoint");
for (const key of ["source_id", "ebook_id", "author_key", "raw_sha256", "panel_sha256"]) {
  assert(new Set(frame.sources.map((source) => source[key])).size === frame.sources.length,
    `independent source-panel ${key} is not unique`);
}
assert(frame.role_partition.entire_sources_disjoint_across_roles === true
  && frame.panel_sampling.panel_documents_per_source === 1
  && frame.outcome_firewall.action_cube_outcomes_read === false,
"source-panel definition or pre-outcome firewall changed");
assert(new Set(predictor.fitted_rows.map((row) => row.source_id)).size === 16
  && predictor.fitted_rows.every((row) => roleSets.fitting.has(row.source_id))
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"residual predictor is not fitting-only");
assert(contract.exchange_set.length === 1
  && contract.probe_features.candidate_multi_atom_outcomes_excluded === true
  && contract.router.candidate_multi_atom_outcomes_hidden_until_after_action === true
  && contract.conformal.alpha_numerator === 1
  && contract.conformal.alpha_denominator === 20
  && contract.authorization.alter_exchange_features_predictor_score_or_falsifiers_after_outcomes
    === false,
"a required frozen design element changed");

const evaluation = result.untouched_evaluation.rows;
assert(evaluation.length === roleSets.evaluation.size
  && new Set(evaluation.map((row) => row.source_id)).size === evaluation.length
  && evaluation.every((row) => roleSets.evaluation.has(row.source_id)),
"evaluation rows are not exactly the untouched evaluation source panels");
const covered = evaluation.filter((row) => row.covered).length;
const fired = evaluation.filter((row) => row.fires);
const unsafe = fired.filter((row) => BigInt(row.exchange_contrast_q32) >= 0n);
const signedRegret = fired.reduce(
  (sum, row) => sum + BigInt(row.exchange_contrast_q32), 0n);
const positiveRegret = fired.reduce((sum, row) => {
  const contrast = BigInt(row.exchange_contrast_q32);
  return sum + (contrast > 0n ? contrast : 0n);
}, 0n);
const metrics = publication.untouched_evaluation_metrics;
assert(same(metrics.unsafe_action_rate, rate(unsafe.length, evaluation.length))
  && same(metrics.unsafe_given_firing_diagnostic, rate(unsafe.length, fired.length))
  && same(metrics.firing_rate, rate(fired.length, evaluation.length))
  && same(metrics.source_panel_coverage, rate(covered, evaluation.length)),
"a published rate changed");
assert(BigInt(metrics.regret_relative_to_abstention.aggregate_signed_q32) === signedRegret
  && metrics.regret_relative_to_abstention.mean_signed_q32.numerator
    === signedRegret.toString()
  && metrics.regret_relative_to_abstention.mean_signed_q32.denominator === evaluation.length
  && BigInt(metrics.regret_relative_to_abstention.aggregate_positive_part_q32)
    === positiveRegret,
"published regret relative to abstention changed");
for (const [index, row] of evaluation.entries()) {
  const reported = metrics.regret_relative_to_abstention.per_source_panel_q32[index];
  const expected = row.fires ? BigInt(row.exchange_contrast_q32) : 0n;
  assert(reported.source_id === row.source_id
    && BigInt(reported.signed_regret_q32) === expected
    && BigInt(reported.positive_part_q32) === (expected > 0n ? expected : 0n),
  "per-panel regret relative to abstention changed");
}

const vacuous = result.conformal.correction_q32 === "positive_infinity"
  || result.conformal.order_statistic_rank > result.conformal.calibration_source_panels;
const expectedVerdict = classify({
  vacuous,
  uncovered: evaluation.length - covered,
  rejection: contract.falsifiers.coverage.exact_binomial_rejection_failures,
  supportMaximum: contract.falsifiers.coverage.maximum_failures_for_support,
  unsafe: unsafe.length,
  fired: fired.length,
  signedRegret,
});
assert(publication.verdict.status === expectedVerdict
  && same(publication.verdict.allowed_statuses,
    ["supported", "falsified", "inconclusive"])
  && publication.verdict.vacuous_envelope_status === "inconclusive",
"three-way publication verdict changed");
assert(classify({vacuous: true, uncovered: 0, rejection: 3, supportMaximum: 1,
  unsafe: 0, fired: 5, signedRegret: -1n}) === "inconclusive"
  && classify({vacuous: false, uncovered: 2, rejection: 3, supportMaximum: 1,
    unsafe: 0, fired: 5, signedRegret: -1n}) === "inconclusive"
  && classify({vacuous: false, uncovered: 3, rejection: 3, supportMaximum: 1,
    unsafe: 0, fired: 5, signedRegret: -1n}) === "falsified"
  && classify({vacuous: false, uncovered: 0, rejection: 3, supportMaximum: 1,
    unsafe: 1, fired: 5, signedRegret: -1n}) === "falsified"
  && classify({vacuous: false, uncovered: 0, rejection: 3, supportMaximum: 1,
    unsafe: 0, fired: 0, signedRegret: 0n}) === "falsified",
"publication verdict edge cases changed");
assert(publication.sealed_material.documents === "200--212"
  && publication.sealed_material.read === false
  && publication.sealed_material.independent_source_panels === false
  && publication.sealed_material.cross_source_transfer_evidence === false
  && contract.authorization.read_documents_200_212 === false
  && result.decision.documents_200_212_read === false,
"sealed same-source documents entered the experiment");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_cross_source_exchange_publication_check.v1",
  independent_source_panel_definition_verified: true,
  fitting_source_panels: roleSets.fitting.size,
  calibration_source_panels: roleSets.calibration.size,
  untouched_evaluation_source_panels: roleSets.evaluation.size,
  rates: {
    unsafe_action: metrics.unsafe_action_rate.exact,
    firing: metrics.firing_rate.exact,
    coverage: metrics.source_panel_coverage.exact,
  },
  aggregate_signed_regret_relative_to_abstention_q32: signedRegret.toString(),
  aggregate_positive_regret_relative_to_abstention_q32: positiveRegret.toString(),
  verdict: expectedVerdict,
  vacuous_envelope_verdict_verified: "inconclusive",
  documents_200_212_read: false,
}, null, 2)}\n`);
