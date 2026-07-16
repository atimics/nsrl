#!/usr/bin/env node

import fs from "node:fs";

const contractPath = process.argv[2]
  ?? "protocol/examples/p10m-adaptive-composition-v1-preregistration.json";
const fail = (message) => {
  throw new Error(`adaptive composition theory v1: ${message}`);
};
const assert = (condition, message) => {
  if (!condition) fail(message);
};
const gcd = (left, right) => {
  let a = left < 0n ? -left : left;
  let b = right < 0n ? -right : right;
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};
const reduce = (numerator, denominator) => {
  const divisor = gcd(numerator, denominator);
  return {numerator: numerator / divisor, denominator: denominator / divisor};
};
const add = (left, right) => reduce(
  left.numerator * right.denominator + right.numerator * left.denominator,
  left.denominator * right.denominator,
);
const ceilDiv = (numerator, denominator) =>
  (numerator + denominator - 1n) / denominator;

const contract = JSON.parse(fs.readFileSync(contractPath));
const canonicalEvaluatorSource = fs.readFileSync(
  contractPath.startsWith("/")
    ? new URL("../../crates/nsrl-train/src/production.rs", `file://${contractPath}`)
    : "crates/nsrl-train/src/production.rs",
  "utf8",
);
assert(contract.schema === "nsrl.adaptive_composition_preregistration.v1",
  "wrong preregistration schema");
assert(contract.analysis_role === "prospective_pre_source_acquisition"
  && contract.status === "preregistered_not_execution_ready",
"preregistration role or state changed");

// Exact counterexample on MJ-19's six-panel horizon: all-zero has probability
// 7/10 and each of the six one-hot unsafe vectors has probability 1/20.
const counterexampleSize = 6;
const allSafeMass = {numerator: 7n, denominator: 10n};
const oneHotMass = {numerator: 1n, denominator: 20n};
let totalCounterexampleMass = allSafeMass;
for (let unsafePosition = 0; unsafePosition < counterexampleSize; unsafePosition += 1) {
  totalCounterexampleMass = add(totalCounterexampleMass, oneHotMass);
}
assert(totalCounterexampleMass.numerator === totalCounterexampleMass.denominator,
  "six-panel exchangeable counterexample mass does not sum to one");
const firstFiveSafeMass = add(allSafeMass, oneHotMass);
assert(firstFiveSafeMass.numerator === 3n && firstFiveSafeMass.denominator === 4n,
  "first-five-safe history mass changed");
const conditionalHazard = reduce(
  oneHotMass.numerator * firstFiveSafeMass.denominator,
  oneHotMass.denominator * firstFiveSafeMass.numerator,
);
const marginalHazard = {numerator: 1n, denominator: 20n};
const unsafeLikelihoodRatioMultiplier = 5n;
const safeLikelihoodRatioMultiplier = {numerator: 15n, denominator: 19n};
const expectedMultiplier = add(
  {numerator: conditionalHazard.numerator * unsafeLikelihoodRatioMultiplier,
    denominator: conditionalHazard.denominator},
  {numerator: (conditionalHazard.denominator - conditionalHazard.numerator)
      * safeLikelihoodRatioMultiplier.numerator,
    denominator: conditionalHazard.denominator
      * safeLikelihoodRatioMultiplier.denominator},
);
assert(conditionalHazard.numerator * marginalHazard.denominator
    > marginalHazard.numerator * conditionalHazard.denominator
  && expectedMultiplier.numerator > expectedMultiplier.denominator
  && expectedMultiplier.numerator === 61n && expectedMultiplier.denominator === 57n,
"counterexample does not violate the conditional supermartingale premise");

// Verify the finite-horizon alpha-spending arithmetic exactly.
const rounds = BigInt(contract.error_control.adaptive_source_rounds);
const globalAlpha = {numerator: 1n, denominator: 20n};
const perRound = {numerator: 1n, denominator: 120n};
let spend = {numerator: 0n, denominator: 1n};
for (let round = 0n; round < rounds; round += 1n) spend = add(spend, perRound);
assert(spend.numerator * globalAlpha.denominator
    === globalAlpha.numerator * spend.denominator,
"per-round spends do not sum to global alpha");

// Split-conformal rank k=ceil((n+1)(1-epsilon)). At epsilon=1/120,
// n=119 is the first finite correction; n=118 is vacuous.
const rank = (calibration) => ceilDiv(
  BigInt(calibration + 1) * (perRound.denominator - perRound.numerator),
  perRound.denominator,
);
assert(rank(119) === 119n && rank(119) <= 119n,
  "119-panel conformal correction is not finite rank 119");
assert(rank(118) === 119n && rank(118) > 118n,
  "118-panel conformal correction should be vacuous");
const freshRanks = Array.from({length: 120}, (_, index) => BigInt(index + 1));
assert(freshRanks.filter((freshRank) => freshRank > rank(119)).length === 1,
  "rank-119 correction does not have exact 1/120 worst-case exceedance");
assert(contract.source_design.calibration_sources_per_family === 119
  && contract.error_control.calibration_order_statistic_rank === 119,
"preregistration does not pay the finite-sample calibration price");

// The time-uniform event is pointwise: if any unsafe event occurs its
// indicator is bounded by the sum of round indicators. This identity is the
// dependence-free core of the union-bound theorem.
for (let mask = 0; mask < (1 << Number(rounds)); mask += 1) {
  const eventCount = Array.from({length: Number(rounds)}, (_, index) =>
    (mask >> index) & 1).reduce((sum, value) => sum + value, 0);
  const anyUnsafe = Number(mask !== 0);
  assert(anyUnsafe <= eventCount, `union indicator failed at mask ${mask}`);
}

// The transition theorem permits genuinely noncommuting state maps. Labels
// are chronological, so HT means T(H(0))=2 while TH means H(T(0))=1.
const head = (state) => state + 1;
const trunk = (state) => state * 2;
const ht = trunk(head(0));
const th = head(trunk(0));
assert(ht === 2 && th === 1 && ht !== th,
  "noncommuting transition witness changed");

const physicalActions = contract.actions.filter((action) =>
  action.kind === "state_specific_physical_move");
const abstentions = contract.actions.filter((action) => action.kind === "abstention");
assert(physicalActions.length === 2
  && physicalActions.map((action) => action.id).join(",")
    === "head_lattice_exit,trunk_lattice_exit"
  && abstentions.length === 1 && abstentions[0].id === "abstain",
"preregistration must bind two physical action families");
assert(physicalActions.every((action) => action.maximum_coordinate_writes === 2
    && action.manifest_source === "proper_fitting_only")
  && abstentions[0].maximum_coordinate_writes === 0,
"physical actions are not fitting-only two-write moves plus abstention");
const groupFingerprints = physicalActions.map((action) =>
  [...action.allowed_groups].sort().join("\0"));
assert(groupFingerprints[0] !== groupFingerprints[1],
  "physical action group sets are not distinct");
assert(contract.reachable_state_contract.labelled_paths.join(",")
    === "empty,H,T,HH,HT,TH,TT"
  && contract.reachable_state_contract.path_label_order
    === "chronological_left_to_right"
  && contract.reachable_state_contract.maximum_labelled_states === 7
  && contract.reachable_state_contract.state_escape_is_hard_falsifier
  && contract.reachable_state_contract.calibration_score_covers_every_state_action_passage,
"bounded state-action calibration surface changed");
assert(contract.noncommutativity_gate.model_hashes_must_differ
  && contract.noncommutativity_gate.function_hashes_must_differ
  && contract.noncommutativity_gate.checked_before_calibration_outcomes,
"noncommutativity is not a pre-outcome hard gate");
assert(contract.noncommutativity_gate.left_path === "HT"
  && contract.noncommutativity_gate.right_path === "TH"
  && contract.noncommutativity_gate.left_transition
    === "T_trunk(T_head(theta_0))"
  && contract.noncommutativity_gate.right_transition
    === "T_head(T_trunk(theta_0))",
"noncommuting path order is ambiguous");

assert(contract.model.persistent_updates && !contract.model.rollback_between_rounds,
  "experiment is not a persistent model composition");
assert(contract.controller.error_control_unit === "whole_source_panel"
  && contract.controller.source_panel_rounds === 6
  && contract.controller.ordered_passage_decisions === 24
  && contract.controller.maximum_physical_actions_per_passage === 1
  && contract.controller.maximum_accepted_actions_globally === 2
  && contract.controller.persist_accepted_action_before_next_passage
  && contract.controller.after_acceptance_limit === "always_abstain"
  && !contract.controller.current_passage_exact_action_outcomes_are_controller_inputs,
"source-panel error control and within-panel persistent decisions are ambiguous");
assert(contract.source_design.families.join(",")
    === "federal_register,rfc,science"
  && contract.source_design.independent_unit === "whole_publication_source_panel"
  && contract.source_design.passages_per_source === 4
  && contract.source_design.windows_per_passage === 2
  && contract.source_design.context_tokens === 64
  && contract.source_design.proper_fitting_sources_per_family === 12
  && contract.source_design.adaptive_sources_per_family === 2
  && contract.source_design.sealed_endpoint_sources_per_family === 19
  && contract.source_design.exclude_all_m18_m19_source_ids_and_independence_keys
  && contract.source_design.source_frame_outcome_independent,
"fresh source-panel design changed");
assert(contract.error_control.global_alpha === "1/20"
  && contract.error_control.per_round_spend === "1/120"
  && contract.error_control.sum_of_spends === "1/20"
  && contract.error_control.unsafe_positive_regret_boundary_q32 === "0"
  && contract.error_control.scope
    === "simultaneous_over_all_prefixes_through_round_6"
  && !contract.error_control.independence_between_evaluation_rounds_required,
"error-control contract changed");
assert(contract.selection.strict_rule
    === "predicted_contrast_plus_family_round_correction_lt_zero"
  && contract.selection.current_candidate_outcomes_forbidden
  && contract.selection.sealed_endpoint_inputs_forbidden
  && contract.selection.state_action_manifest_must_be_frozen,
"selection is not outcome-blind and fitting-fixed");
assert(contract.objectives.primary_retained_model_endpoint
    === "integer_base2_softmax_nll_millibits"
  && contract.objectives.primary_aggregation === "total_nll_millibits"
  && contract.objectives.canonical_evaluator
    === "evaluate_production_model_canonical_nll"
  && contract.objectives.canonical_evaluator_path
    === "crates/nsrl-train/src/production.rs"
  && contract.objectives.normalization_independent,
"retained-model endpoint is not canonical NLL");
assert(contract.objectives.certificate_observation === "q47_weight_base2_nll_q32"
  && contract.objectives.zero_probability_floor_millibits === 32000,
"certificate or endpoint numeric objective changed");
assert(canonicalEvaluatorSource.includes("pub fn evaluate_production_model_canonical_nll(")
  && canonicalEvaluatorSource.includes("total_nll_millibits")
  && canonicalEvaluatorSource.includes("zero_probability_floor_millibits"),
"canonical retained-model evaluator binding is missing");
assert(contract.comparators.includes("always_abstain")
  && contract.comparators.includes("head_lattice_exit_only_policy")
  && contract.comparators.includes("trunk_lattice_exit_only_policy"),
"required abstention and fixed-action comparators are missing");
assert(contract.support_requires.adaptive_endpoint_strictly_beats_always_abstain
  && contract.support_requires.adaptive_endpoint_strictly_beats_best_fixed_action_policy
  && contract.support_requires.observed_cumulative_positive_regret_q32_equals === "0"
  && contract.support_requires.zero_probability_windows_do_not_increase
  && contract.support_requires.persistent_final_model_artifact_present
  && contract.support_requires.byte_identical_replay,
"promotion rules do not bind both comparators and positive regret");
for (const falsifier of [
  "physical_action_delta_fingerprints_equal",
  "physical_action_group_sets_equal",
  "ht_th_model_hashes_equal",
  "ht_th_function_hashes_equal",
  "executed_state_outside_bound_reachable_set",
  "current_candidate_outcome_used_for_selection",
  "any_fired_exact_contrast_nonnegative",
  "adaptive_endpoint_not_strictly_better_than_abstain",
  "adaptive_endpoint_not_strictly_better_than_best_fixed_policy",
  "zero_probability_windows_increase",
  "replay_hash_mismatch",
]) {
  assert(contract.hard_falsifiers.includes(falsifier),
    `missing hard falsifier ${falsifier}`);
}
assert(contract.authorization.evaluation_outcomes === false
  && contract.authorization.optimizer_promotion === false
  && contract.authorization.paid_scaling === false,
"pre-execution authorization boundary changed");

// Complete the omitted MJ-19 numeric branch. A nonzero Q47 target weight gives
// at most 60 bits; an annihilated target receives the declared 32-bit floor.
const maximumNonzeroTargetNllBits = 60n;
const annihilatedTargetFloorBits = 32n;
const maximumWindowNllBits = maximumNonzeroTargetNllBits > annihilatedTargetFloorBits
  ? maximumNonzeroTargetNllBits : annihilatedTargetFloorBits;
const maximumPassageContrastBits = 2n * maximumWindowNllBits;
assert(maximumWindowNllBits === 60n && maximumPassageContrastBits === 120n,
  "corrected MJ-19 numeric bound changed");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.adaptive_composition_theory_check.v1",
  marginal_to_conditional_bridge: {
    status: "falsified",
    exchangeable_coordinates: counterexampleSize,
    marginal_unsafe_probability: "1/20",
    terminal_conditional_unsafe_probability_after_5_safe: "1/15",
    mj19_unsafe_multiplier_on_that_history: unsafeLikelihoodRatioMultiplier.toString(),
    mj19_conditional_expected_multiplier: "61/57",
  },
  valid_replacement: {
    method: "simultaneous_state_action_split_conformal_plus_alpha_spending",
    horizon: Number(rounds),
    global_alpha: "1/20",
    per_round_spend: "1/120",
    calibration_sources_per_family: 119,
    calibration_rank: 119,
    independence_between_evaluation_rounds_required: false,
    anytime_positive_regret_bound_q32: "0",
  },
  persistent_optimizer_contract: {
    physical_action_families: physicalActions.map((action) => action.id),
    labelled_reachable_paths: contract.reachable_state_contract.labelled_paths,
    source_panel_rounds: contract.controller.source_panel_rounds,
    ordered_passage_decisions: contract.controller.ordered_passage_decisions,
    maximum_accepted_actions: contract.controller.maximum_accepted_actions_globally,
    noncommuting_transition_witness: {HT: ht, TH: th},
    comparators: contract.comparators,
    primary_endpoint: contract.objectives.primary_retained_model_endpoint,
    execution_ready: false,
  },
  mj19_numeric_correction: {
    nonzero_target_maximum_nll_bits: maximumNonzeroTargetNllBits.toString(),
    annihilated_target_floor_bits: annihilatedTargetFloorBits.toString(),
    two_window_maximum_absolute_contrast_bits: maximumPassageContrastBits.toString(),
  },
}, null, 2)}\n`);
