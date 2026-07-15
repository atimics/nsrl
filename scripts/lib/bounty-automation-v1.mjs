import {
  canonicalJson,
  sha256Canonical,
  validateModelLaunchRecipe,
} from "./model-launch-v1.mjs";

export const BOUNTY_AUTOMATION_POLICY_SCHEMA = "nsrl.bounty_automation_policy.v1";

const ACCOUNT_PATTERN = /^nsrl:[a-z0-9][a-z0-9:_-]{2,127}$/;
const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const UINT_PATTERN = /^(0|[1-9][0-9]*)$/;

function fail(message) {
  throw new Error(`bounty automation v1: ${message}`);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function string(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function account(value, label) {
  const parsed = string(value, label);
  if (!ACCOUNT_PATTERN.test(parsed)) {
    fail(`${label} must use the nsrl: account namespace`);
  }
  return parsed;
}

function slug(value, label) {
  const parsed = string(value, label);
  if (!SLUG_PATTERN.test(parsed)) {
    fail(`${label} must be a lowercase kebab-case slug`);
  }
  return parsed;
}

function uintString(value, label, { positive = false } = {}) {
  if (typeof value !== "string" || !UINT_PATTERN.test(value)) {
    fail(`${label} must be an unsigned base-10 integer string`);
  }
  if (positive && value === "0") {
    fail(`${label} must be positive`);
  }
  return value;
}

function uint(value, label, { positive = false, maximum = Number.MAX_SAFE_INTEGER } = {}) {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0) || value > maximum) {
    fail(`${label} must be ${positive ? "a positive" : "a nonnegative"} safe integer`);
  }
  return value;
}

function sourceEvidence(state, sourceLaunchId, metric) {
  const launch = state.launches[sourceLaunchId];
  if (!launch) {
    fail(`source launch ${sourceLaunchId} does not exist`);
  }
  const publication = state.publications[sourceLaunchId];
  if (!publication) {
    fail(`source launch ${sourceLaunchId} is not promoted`);
  }
  const candidate = state.candidates[publication.candidate_event_id];
  if (!candidate) {
    fail(`source launch ${sourceLaunchId} has no accepted candidate`);
  }
  const metricValue = candidate.metrics[metric];
  if (metricValue === undefined) {
    fail(`source launch ${sourceLaunchId} has no ${metric} metric`);
  }
  return { launch, publication, candidate, metricValue };
}

export function validateBountyAutomationPolicy(policy) {
  object(policy, "policy");
  if (policy.schema !== BOUNTY_AUTOMATION_POLICY_SCHEMA) {
    fail(`policy.schema must be ${BOUNTY_AUTOMATION_POLICY_SCHEMA}`);
  }
  slug(policy.id, "policy.id");
  account(policy.sponsor, "policy.sponsor");
  account(policy.proposer, "policy.proposer");
  account(policy.keeper, "policy.keeper");
  slug(policy.source_launch_id, "policy.source_launch_id");

  const trigger = object(policy.trigger, "policy.trigger");
  if (trigger.event !== "model_promoted") {
    fail("policy.trigger.event must be model_promoted");
  }

  const objective = object(policy.objective, "policy.objective");
  string(objective.metric, "policy.objective.metric");
  if (!['minimize', 'maximize'].includes(objective.direction)) {
    fail("policy.objective.direction must be minimize or maximize");
  }
  uint(objective.relative_improvement_bps, "policy.objective.relative_improvement_bps", {
    positive: true,
    maximum: 9999,
  });

  const budgets = object(policy.budgets, "policy.budgets");
  const bountyUnits = BigInt(
    uintString(budgets.bounty_units, "policy.budgets.bounty_units", { positive: true }),
  );
  const computeUnits = BigInt(
    uintString(budgets.compute_budget_units, "policy.budgets.compute_budget_units", {
      positive: true,
    }),
  );
  const maxSpend = BigInt(
    uintString(budgets.max_total_spend_units, "policy.budgets.max_total_spend_units", {
      positive: true,
    }),
  );
  uintString(
    budgets.manual_approval_above_units,
    "policy.budgets.manual_approval_above_units",
  );
  if (bountyUnits + computeUnits > maxSpend) {
    fail("one automation cycle exceeds policy.budgets.max_total_spend_units");
  }

  const limits = object(policy.limits, "policy.limits");
  uint(limits.max_active_bounties, "policy.limits.max_active_bounties", {
    positive: true,
    maximum: 100,
  });
  uint(limits.max_cycles, "policy.limits.max_cycles", { positive: true, maximum: 10000 });
  uint(limits.cooldown_slots, "policy.limits.cooldown_slots");

  const auction = object(policy.auction, "policy.auction");
  const bidWindow = uint(auction.bid_window_slots, "policy.auction.bid_window_slots", {
    positive: true,
  });
  const revealWindow = uint(
    auction.reveal_window_slots,
    "policy.auction.reveal_window_slots",
    { positive: true },
  );
  const executionWindow = uint(
    auction.execution_window_slots,
    "policy.auction.execution_window_slots",
    { positive: true },
  );
  if (executionWindow <= bidWindow + revealWindow) {
    fail("policy auction execution window must extend beyond bid and reveal windows");
  }
  uintString(
    auction.minimum_collateral_units,
    "policy.auction.minimum_collateral_units",
    { positive: true },
  );
  return policy;
}

export function bountyAutomationPolicySha256(policy) {
  validateBountyAutomationPolicy(policy);
  return sha256Canonical(policy);
}

export function automationCycleSpendUnits(policy) {
  validateBountyAutomationPolicy(policy);
  return (
    BigInt(policy.budgets.bounty_units) + BigInt(policy.budgets.compute_budget_units)
  ).toString();
}

export function automationCycleLaunchId(policyId, cycleIndex) {
  return `${slug(policyId, "policy id")}-cycle-${uint(cycleIndex, "cycle index", {
    positive: true,
  })}`;
}

export function automationTargetValue(baseline, direction, improvementBps) {
  const value = BigInt(uintString(baseline, "automation baseline"));
  const bps = BigInt(
    uint(improvementBps, "automation relative improvement bps", { positive: true }),
  );
  if (direction === "minimize") {
    if (bps >= 10000n) {
      fail("minimized metric improvement must remain below 10000 bps");
    }
    return ((value * (10000n - bps)) / 10000n).toString();
  }
  if (direction === "maximize") {
    return ((value * (10000n + bps) + 9999n) / 10000n).toString();
  }
  fail("automation direction must be minimize or maximize");
}

export function buildAutomatedBountyRecipe(
  state,
  policy,
  cycleIndex,
  sourceLaunchId,
  publishedAt,
) {
  validateBountyAutomationPolicy(policy);
  const cycle = uint(cycleIndex, "cycle index", { positive: true });
  if (Number.isNaN(Date.parse(string(publishedAt, "cycle published_at")))) {
    fail("cycle published_at must be an ISO date-time");
  }
  const { launch, publication, metricValue } = sourceEvidence(
    state,
    sourceLaunchId,
    policy.objective.metric,
  );
  const sourceBounty = launch.recipe.bounties.find(
    (row) => row.metric === policy.objective.metric,
  );
  if (!sourceBounty) {
    fail(`source recipe has no ${policy.objective.metric} bounty contract`);
  }
  if (sourceBounty.direction !== policy.objective.direction) {
    fail("policy objective direction does not match the source bounty contract");
  }

  const recipe = structuredClone(launch.recipe);
  const launchId = automationCycleLaunchId(policy.id, cycle);
  recipe.launch.id = launchId;
  recipe.launch.title = `${launch.recipe.launch.title} · automated frontier ${cycle}`;
  recipe.launch.summary = `Automatically require ${policy.objective.relative_improvement_bps} bps of ${policy.objective.metric} improvement from ${sourceLaunchId} under ${policy.id}.`;
  recipe.launch.mode = "test";
  recipe.launch.status = "open";
  recipe.launch.published_at = publishedAt;
  recipe.proposer.account = policy.proposer;
  recipe.participants.sponsor = policy.sponsor;
  recipe.model.parent_model_hash = publication.model_hash;
  recipe.bounties = [
    {
      ...structuredClone(sourceBounty),
      id: `${policy.id}-frontier-${cycle}`,
      sponsor: policy.sponsor,
      escrow_units: policy.budgets.bounty_units,
      baseline: metricValue,
      target: automationTargetValue(
        metricValue,
        policy.objective.direction,
        policy.objective.relative_improvement_bps,
      ),
    },
  ];
  recipe.publication = null;
  validateModelLaunchRecipe(recipe);
  return recipe;
}

export function validateAutomatedBountyRecipe(
  state,
  policy,
  cycleIndex,
  sourceLaunchId,
  recipe,
) {
  const expected = buildAutomatedBountyRecipe(
    state,
    policy,
    cycleIndex,
    sourceLaunchId,
    recipe?.launch?.published_at,
  );
  if (canonicalJson(recipe) !== canonicalJson(expected)) {
    fail("automated launch recipe does not match its frozen policy and source publication");
  }
  return recipe;
}

export function automationAuctionDeadlines(policy, currentSlot) {
  validateBountyAutomationPolicy(policy);
  const slot = uint(currentSlot, "current slot");
  const bid = slot + policy.auction.bid_window_slots;
  const reveal = bid + policy.auction.reveal_window_slots;
  const execution = slot + policy.auction.execution_window_slots;
  if (![bid, reveal, execution].every(Number.isSafeInteger)) {
    fail("automation auction deadlines exceed the safe integer range");
  }
  return {
    bid_deadline_slot: bid,
    reveal_deadline_slot: reveal,
    execution_deadline_slot: execution,
  };
}

export function buildAutomationCycleOpenPayload(
  state,
  policy,
  cycleIndex,
  sourceLaunchId,
  publishedAt,
) {
  const recipe = buildAutomatedBountyRecipe(
    state,
    policy,
    cycleIndex,
    sourceLaunchId,
    publishedAt,
  );
  return {
    policy_id: policy.id,
    cycle_index: cycleIndex,
    source_launch_id: sourceLaunchId,
    launch_id: recipe.launch.id,
    published_at: recipe.launch.published_at,
    recipe_sha256: sha256Canonical(recipe),
    cycle_spend_units: automationCycleSpendUnits(policy),
  };
}
