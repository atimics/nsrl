#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const resultPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-publication.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const rate = (numerator, denominator) => ({
  numerator,
  denominator,
  exact: `${numerator}/${denominator}`,
});
const classify = ({vacuous, uncovered, coverageRejectionFailures,
  maximumCoverageFailuresForSupport, unsafe, maximumUnsafeForSupport, fired,
  minimumFiredForSupport, signedRegret}) => {
  if (vacuous) return {
    status: "inconclusive",
    reason: "vacuous_conformal_envelope",
  };
  if (uncovered < coverageRejectionFailures
    && uncovered > maximumCoverageFailuresForSupport) {
    return {
      status: "inconclusive",
      reason: "coverage_between_support_and_rejection_boundaries",
    };
  }
  if (uncovered >= coverageRejectionFailures) return {
    status: "falsified",
    reason: "source_envelope_failure_falsifier",
  };
  if (unsafe > maximumUnsafeForSupport) return {
    status: "falsified",
    reason: "unsafe_action_falsifier",
  };
  if (fired < minimumFiredForSupport) return {
    status: "falsified",
    reason: "nonvacuity_falsifier",
  };
  if (signedRegret >= 0n) return {
    status: "falsified",
    reason: "nonnegative_regret_relative_to_abstention_falsifier",
  };
  return {status: "supported", reason: "all_preregistered_support_gates_pass"};
};

const resultBytes = fs.readFileSync(resultPath);
const contractBytes = fs.readFileSync(contractPath);
const result = JSON.parse(resultBytes);
const contract = JSON.parse(contractBytes);
assert(result.schema === "nsrl.production_cross_source_exchange_result.v1"
  && result.analysis_role === "prospective_untouched_evaluation",
"wrong prospective result");
assert(contract.schema === "nsrl.production_cross_source_exchange_contract.v1"
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"wrong frozen prospective contract");
assert(result.source_sha256.contract === sha256(contractBytes),
  "result is not bound to the frozen contract");

const evaluation = result.untouched_evaluation.rows;
const denominator = evaluation.length;
assert(denominator === contract.population.evaluation_source_panels,
  "evaluation source-panel count changed");
const covered = evaluation.filter((row) => row.covered).length;
const fired = evaluation.filter((row) => row.fires);
const unsafe = fired.filter((row) => row.unsafe);
const signedRegret = fired.reduce(
  (sum, row) => sum + BigInt(row.exchange_contrast_q32), 0n);
const positiveRegret = fired.reduce((sum, row) => {
  const contrast = BigInt(row.exchange_contrast_q32);
  return sum + (contrast > 0n ? contrast : 0n);
}, 0n);
assert(covered === result.untouched_evaluation.envelope_covered
  && fired.length === result.untouched_evaluation.fired_source_panels
  && unsafe.length === result.untouched_evaluation.unsafe_firings
  && signedRegret === BigInt(
    result.untouched_evaluation.aggregate_fired_exchange_contrast_q32),
"published metric substrate changed");

const calibrationPanels = result.conformal.calibration_source_panels;
const vacuous = result.conformal.correction_q32 === "positive_infinity"
  || result.conformal.order_statistic_rank > calibrationPanels;
const verdict = classify({
  vacuous,
  uncovered: denominator - covered,
  coverageRejectionFailures:
    contract.falsifiers.coverage.exact_binomial_rejection_failures,
  maximumCoverageFailuresForSupport:
    contract.falsifiers.coverage.maximum_failures_for_support,
  unsafe: unsafe.length,
  maximumUnsafeForSupport:
    contract.falsifiers.unsafe_action.maximum_unsafe_firings_for_support,
  fired: fired.length,
  minimumFiredForSupport:
    contract.falsifiers.nonvacuity.minimum_fired_source_panels,
  signedRegret,
});
const publisherBytes = fs.readFileSync(new URL(import.meta.url));
const publication = {
  schema: "nsrl.production_cross_source_exchange_publication.v1",
  analysis_role: "deterministic_post_outcome_reporting_of_frozen_prospective_experiment",
  source_sha256: {
    frozen_prospective_contract: sha256(contractBytes),
    checked_prospective_result: sha256(resultBytes),
    publisher: sha256(publisherBytes),
  },
  prospective_integrity: {
    reporting_only: true,
    exchange_features_predictor_abstention_alpha_and_falsifiers_changed: false,
    calibration_or_evaluation_outcomes_used_to_tune_rule: false,
    source_roles_changed: false,
  },
  independent_source_panel: {
    experimental_unit: contract.population.source_unit,
    required_distinctness:
      "unique Gutenberg ebook ID, raw SHA-256, panel SHA-256, and normalized author key across all fitting, calibration, and evaluation roles",
    role_rule: "the complete ebook source is assigned to exactly one role",
    panel_contents: contract.panel_sampling,
    stochastic_independence_claimed: false,
    exchangeability_assumption: contract.population.exchangeability_assumption,
    documents_200_212_excluded_as_same_source_documents: true,
  },
  frozen_design: {
    fitting_source_panels: contract.population.fitting_source_panels,
    calibration_source_panels: contract.population.calibration_source_panels,
    evaluation_source_panels: contract.population.evaluation_source_panels,
    exchange_set: contract.exchange_set,
    probe_features: contract.probe_features,
    residual_predictor: contract.predictor,
    abstention_rule: contract.router.abstention_rule,
    strict_firing_rule: contract.router.strict_firing_rule,
    error_level: `${contract.conformal.alpha_numerator}/${contract.conformal.alpha_denominator}`,
    falsifiers: contract.falsifiers,
  },
  conformal_envelope: {
    status: vacuous ? "vacuous" : "finite",
    calibration_source_panels: calibrationPanels,
    order_statistic_rank: result.conformal.order_statistic_rank,
    correction_q32: result.conformal.correction_q32,
    minimum_calibration_source_panels_for_finite_95_percent_envelope: 19,
  },
  untouched_evaluation_metrics: {
    evaluation_source_panels: denominator,
    unsafe_action_rate: rate(unsafe.length, denominator),
    unsafe_given_firing_diagnostic: rate(unsafe.length, fired.length),
    firing_rate: rate(fired.length, denominator),
    source_panel_coverage: rate(covered, denominator),
    regret_relative_to_abstention: {
      definition:
        "candidate loss minus control loss when the router fires; zero when it abstains",
      comparator: "always retain control mask 47",
      sign: "negative is improvement; positive is harm relative to abstention",
      aggregate_signed_q32: signedRegret.toString(),
      mean_signed_q32: {
        numerator: signedRegret.toString(),
        denominator,
      },
      aggregate_positive_part_q32: positiveRegret.toString(),
      mean_positive_part_q32: {
        numerator: positiveRegret.toString(),
        denominator,
      },
      per_source_panel_q32: evaluation.map((row) => ({
        source_id: row.source_id,
        action: row.fires ? "fire" : "abstain",
        signed_regret_q32: row.fires ? row.exchange_contrast_q32 : "0",
        positive_part_q32: row.fires && BigInt(row.exchange_contrast_q32) > 0n
          ? row.exchange_contrast_q32 : "0",
      })),
    },
  },
  verdict: {
    status: verdict.status,
    reason: verdict.reason,
    allowed_statuses: ["supported", "falsified", "inconclusive"],
    vacuous_envelope_status: "inconclusive",
    scope: contract.decision_rule.scope_if_supported,
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
  },
  sealed_material: {
    documents: "200--212",
    read: false,
    independent_source_panels: false,
    cross_source_transfer_evidence: false,
  },
};
const bytes = `${JSON.stringify(publication, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: publication.schema,
  output: outputPath,
  publication_sha256: sha256(Buffer.from(bytes)),
  rates: {
    unsafe_action: publication.untouched_evaluation_metrics.unsafe_action_rate.exact,
    firing: publication.untouched_evaluation_metrics.firing_rate.exact,
    coverage: publication.untouched_evaluation_metrics.source_panel_coverage.exact,
  },
  aggregate_signed_regret_relative_to_abstention_q32: signedRegret.toString(),
  aggregate_positive_regret_relative_to_abstention_q32: positiveRegret.toString(),
  verdict: publication.verdict.status,
  documents_200_212_read: false,
}, null, 2)}\n`);
