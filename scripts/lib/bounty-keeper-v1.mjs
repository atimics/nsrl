import {
  automationAuctionDeadlines,
  automationCycleSpendUnits,
  buildAutomatedBountyRecipe,
  buildAutomationCycleOpenPayload,
} from "./bounty-automation-v1.mjs";
import { signLocalnetIntent } from "./model-localnet-v1.mjs";
import { sha256Canonical } from "./model-launch-v1.mjs";

export const BOUNTY_KEEPER_PLAN_SCHEMA = "nsrl.bounty_keeper_plan.v1";
export const BOUNTY_KEEPER_RESULT_SCHEMA = "nsrl.bounty_keeper_result.v1";

function waiting(policyId, reason, detail, extra = {}) {
  return {
    schema: BOUNTY_KEEPER_PLAN_SCHEMA,
    policy_id: policyId,
    status: "waiting",
    reason,
    detail,
    ...extra,
  };
}

function activeCycleCount(state, record) {
  return Object.values(record.cycles).filter((cycle) => {
    const launch = state.launches[cycle.launch_id];
    return !launch || !["promoted", "expired"].includes(launch.status);
  }).length;
}

function resumableCycle(record) {
  return Object.values(record.cycles)
    .sort((left, right) => left.cycle_index - right.cycle_index)
    .find((cycle) => ["opened", "published"].includes(cycle.status));
}

function planInterruptedCycle(state, policyId, record, cycle) {
  const policy = record.policy;
  const recipe = buildAutomatedBountyRecipe(
    state,
    policy,
    cycle.cycle_index,
    cycle.source_launch_id,
    cycle.published_at,
  );
  const launch = state.launches[cycle.launch_id];
  const actions = [];
  if (!launch) {
    actions.push("publish_launch", "fund_bounty", "fund_compute");
  } else {
    if (!cycle.bounty_funding_event_id) {
      actions.push("fund_bounty");
    }
    if (!cycle.compute_funding_event_id) {
      actions.push("fund_compute");
    }
  }
  if (actions.includes("fund_compute") && state.slot >= cycle.auction.bid_deadline_slot) {
    return waiting(
      policyId,
      "resume_deadline_elapsed",
      "The reserved cycle needs sponsor recovery because its bound bid window elapsed before compute funding.",
      { cycle_index: cycle.cycle_index, launch_id: cycle.launch_id },
    );
  }
  return {
    schema: BOUNTY_KEEPER_PLAN_SCHEMA,
    policy_id: policyId,
    status: "ready",
    reason: "interrupted_cycle",
    cycle_index: cycle.cycle_index,
    source_launch_id: cycle.source_launch_id,
    launch_id: cycle.launch_id,
    current_slot: state.slot,
    cycle_spend_units: cycle.cycle_spend_units,
    target: recipe.bounties[0].target,
    baseline: recipe.bounties[0].baseline,
    metric: recipe.bounties[0].metric,
    cycle_event_id: cycle.event_id,
    recipe,
    auction: cycle.auction,
    actions,
  };
}

export function planBountyKeeperCycle(state, policyId, publishedAt = new Date().toISOString()) {
  const record = state.automation_policies[policyId];
  if (!record) {
    return waiting(policyId, "policy_missing", "Register the signed automation policy first.");
  }
  const policy = record.policy;
  if (record.status !== "active") {
    return waiting(policyId, "policy_paused", "The sponsor paused this automation policy.");
  }
  const interrupted = resumableCycle(record);
  if (interrupted) {
    return planInterruptedCycle(state, policyId, record, interrupted);
  }
  const cycleIndex = Object.keys(record.cycles).length + 1;
  if (cycleIndex > policy.limits.max_cycles) {
    return waiting(policyId, "cycle_cap_reached", "The policy has opened its maximum cycles.");
  }
  if (activeCycleCount(state, record) >= policy.limits.max_active_bounties) {
    return waiting(
      policyId,
      "active_limit_reached",
      "A prior automated bounty must promote or expire before another can open.",
    );
  }
  const sourceLaunchId =
    cycleIndex === 1
      ? policy.source_launch_id
      : record.cycles[String(cycleIndex - 1)]?.launch_id;
  const sourcePublication = state.publications[sourceLaunchId];
  if (!sourcePublication) {
    return waiting(
      policyId,
      "promotion_required",
      `Source launch ${sourceLaunchId} has not been promoted.`,
      { cycle_index: cycleIndex, source_launch_id: sourceLaunchId },
    );
  }
  const cooldownBase = record.last_opened_slot ?? sourcePublication.slot ?? 0;
  const readySlot = cooldownBase + policy.limits.cooldown_slots;
  if (state.slot < readySlot) {
    return waiting(
      policyId,
      "cooldown",
      `The next cycle becomes eligible at logical slot ${readySlot}.`,
      { cycle_index: cycleIndex, source_launch_id: sourceLaunchId, ready_slot: readySlot },
    );
  }
  const cycleSpend = automationCycleSpendUnits(policy);
  if (
    BigInt(record.spent_units) + BigInt(cycleSpend) >
    BigInt(policy.budgets.max_total_spend_units)
  ) {
    return waiting(
      policyId,
      "spend_cap_reached",
      "The next cycle would exceed the sponsor's lifetime automation cap.",
      { cycle_index: cycleIndex, source_launch_id: sourceLaunchId },
    );
  }
  const approvalRequired =
    BigInt(cycleSpend) > BigInt(policy.budgets.manual_approval_above_units);
  if (approvalRequired && !state.automation_approvals[policyId]?.[cycleIndex]) {
    return waiting(
      policyId,
      "manual_approval_required",
      "The next cycle exceeds the automatic per-cycle threshold.",
      {
        cycle_index: cycleIndex,
        source_launch_id: sourceLaunchId,
        approval_units: cycleSpend,
      },
    );
  }
  if (BigInt(state.test_balances[policy.sponsor] ?? "0") < BigInt(cycleSpend)) {
    return waiting(
      policyId,
      "insufficient_balance",
      "The sponsor balance cannot fund the complete bounty and compute budget.",
      { cycle_index: cycleIndex, source_launch_id: sourceLaunchId },
    );
  }

  const cycle = buildAutomationCycleOpenPayload(
    state,
    policy,
    cycleIndex,
    sourceLaunchId,
    publishedAt,
  );
  const recipe = buildAutomatedBountyRecipe(
    state,
    policy,
    cycleIndex,
    sourceLaunchId,
    publishedAt,
  );
  return {
    schema: BOUNTY_KEEPER_PLAN_SCHEMA,
    policy_id: policyId,
    status: "ready",
    reason: "promotion_triggered",
    cycle_index: cycleIndex,
    source_launch_id: sourceLaunchId,
    launch_id: cycle.launch_id,
    current_slot: state.slot,
    cycle_spend_units: cycleSpend,
    target: recipe.bounties[0].target,
    baseline: recipe.bounties[0].baseline,
    metric: recipe.bounties[0].metric,
    cycle,
    recipe,
    auction: automationAuctionDeadlines(policy, state.slot),
    actions: ["open_cycle", "publish_launch", "fund_bounty", "fund_compute"],
  };
}

function append(ledger, identity, eventType, payload) {
  return ledger.append(signLocalnetIntent(identity, eventType, payload)).event;
}

function publicEvent(event) {
  return {
    event_type: event.signed_intent.event_type,
    event_id: event.event_id,
    height: event.height,
  };
}

export function runBountyKeeperCycle(
  ledger,
  policyId,
  { keeper, proposer, sponsor },
  publishedAt = new Date().toISOString(),
) {
  const inspected = ledger.inspect();
  const plan = planBountyKeeperCycle(inspected.state, policyId, publishedAt);
  if (plan.status !== "ready") {
    return {
      schema: BOUNTY_KEEPER_RESULT_SCHEMA,
      policy_id: policyId,
      status: "waiting",
      plan,
      events: [],
    };
  }
  const policy = inspected.state.automation_policies[policyId].policy;
  for (const [role, identity] of [
    ["keeper", keeper],
    ["proposer", proposer],
    ["sponsor", sponsor],
  ]) {
    if (identity.account !== policy[role]) {
      throw new Error(`bounty keeper v1: ${role} key does not match policy account`);
    }
  }

  const events = [];
  let cycleEventId = plan.cycle_event_id;
  if (plan.actions.includes("open_cycle")) {
    const cycleEvent = append(ledger, keeper, "bounty_automation_cycle_opened", plan.cycle);
    events.push(cycleEvent);
    cycleEventId = cycleEvent.event_id;
  }
  if (plan.actions.includes("publish_launch")) {
    events.push(append(ledger, proposer, "launch_published", {
      recipe_sha256: sha256Canonical(plan.recipe),
      recipe: plan.recipe,
      automation_cycle_event_id: cycleEventId,
    }));
  }
  if (plan.actions.includes("fund_bounty")) {
    events.push(append(ledger, sponsor, "bounty_funded", {
      launch_id: plan.launch_id,
      bounty_id: plan.recipe.bounties[0].id,
      escrow_units: policy.budgets.bounty_units,
    }));
  }
  if (plan.actions.includes("fund_compute")) {
    events.push(append(ledger, sponsor, "compute_budget_funded", {
      launch_id: plan.launch_id,
      escrow_units: policy.budgets.compute_budget_units,
      ...plan.auction,
      minimum_collateral_units: policy.auction.minimum_collateral_units,
    }));
  }

  const finalState = ledger.inspect().state;
  const finalCycle = finalState.automation_policies[policyId].cycles[String(plan.cycle_index)];
  if (finalCycle.status !== "funded") {
    throw new Error("bounty keeper v1: keeper tick did not fully fund its planned cycle");
  }

  return {
    schema: BOUNTY_KEEPER_RESULT_SCHEMA,
    policy_id: policyId,
    status: plan.reason === "interrupted_cycle" ? "resumed" : "opened",
    cycle_index: plan.cycle_index,
    source_launch_id: plan.source_launch_id,
    launch_id: plan.launch_id,
    target: plan.target,
    baseline: plan.baseline,
    metric: plan.metric,
    cycle_spend_units: plan.cycle_spend_units,
    events: events.map(publicEvent),
    ledger_height: finalState.height,
  };
}
