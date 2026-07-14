import { createHash } from "node:crypto";

export const RECIPE_SCHEMA = "nsrl.model_launch_recipe.v1";
export const REWARD_BLOCK_SCHEMA = "nsrl.model_reward_block.v1";
export const PUBLISH_RECEIPT_SCHEMA = "nsrl.model_publish_receipt.v1";

export const REWARD_ROLES = ["builder", "compute", "validator", "sponsor", "treasury"];
export const REWARD_EVENTS = [
  "launch_published",
  "stage_accepted",
  "candidate_valid",
  "independent_replay",
  "model_promoted",
];

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const MODEL_HASH_PATTERN = /^0x[a-f0-9]{16}$/;
const ACCOUNT_PATTERN = /^nsrl:[a-z0-9][a-z0-9:_-]{2,127}$/;
const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function fail(message) {
  throw new Error(`model launch v1: ${message}`);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) {
    fail(`${label} must be an array`);
  }
  return value;
}

function string(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function integer(value, label, { minimum = 0n, positive = false } = {}) {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]*)$/.test(value)) {
    fail(`${label} must be an unsigned base-10 integer string`);
  }
  const parsed = BigInt(value);
  if (parsed < minimum || (positive && parsed === 0n)) {
    fail(`${label} is below its allowed minimum`);
  }
  return parsed;
}

function uint(value, label, { minimum = 0, maximum = Number.MAX_SAFE_INTEGER } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail(`${label} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function hash256(value, label) {
  const normalized = string(value, label).toLowerCase();
  if (!SHA256_PATTERN.test(normalized)) {
    fail(`${label} must be a lowercase SHA-256 digest without a prefix`);
  }
  return normalized;
}

function unique(items, label) {
  const seen = new Set();
  for (const item of items) {
    if (seen.has(item)) {
      fail(`${label} contains duplicate ${item}`);
    }
    seen.add(item);
  }
}

export function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function sha256Canonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

export function validateModelLaunchRecipe(recipe) {
  object(recipe, "recipe");
  if (recipe.schema !== RECIPE_SCHEMA) {
    fail(`schema must be ${RECIPE_SCHEMA}`);
  }

  const launch = object(recipe.launch, "launch");
  const launchId = string(launch.id, "launch.id");
  if (!SLUG_PATTERN.test(launchId)) {
    fail("launch.id must be a lowercase kebab-case slug");
  }
  string(launch.title, "launch.title");
  string(launch.summary, "launch.summary");
  string(launch.network, "launch.network");
  if (!["specimen", "test", "production"].includes(launch.mode)) {
    fail("launch.mode must be specimen, test, or production");
  }
  if (!["draft", "open", "running", "candidate", "promoted", "failed", "expired"].includes(launch.status)) {
    fail("launch.status is not recognized");
  }
  if (Number.isNaN(Date.parse(string(launch.published_at, "launch.published_at")))) {
    fail("launch.published_at must be an ISO date-time");
  }

  const proposer = object(recipe.proposer, "proposer");
  const proposerAccount = string(proposer.account, "proposer.account");
  if (!ACCOUNT_PATTERN.test(proposerAccount)) {
    fail("proposer.account must use the nsrl: account namespace");
  }
  integer(proposer.bond_units, "proposer.bond_units");

  const participants = object(recipe.participants, "participants");
  for (const role of REWARD_ROLES) {
    const account = string(participants[role], `participants.${role}`);
    if (!ACCOUNT_PATTERN.test(account)) {
      fail(`participants.${role} must use the nsrl: account namespace`);
    }
  }

  const model = object(recipe.model, "model");
  string(model.family, "model.family");
  string(model.artifact_format, "model.artifact_format");
  string(model.architecture_profile, "model.architecture_profile");
  string(model.tokenizer_contract, "model.tokenizer_contract");
  if (model.parent_model_hash !== null && !MODEL_HASH_PATTERN.test(string(model.parent_model_hash, "model.parent_model_hash"))) {
    fail("model.parent_model_hash must be null or a 64-bit 0x-prefixed model hash");
  }
  integer(model.max_artifact_bytes, "model.max_artifact_bytes", { positive: true });

  const source = object(recipe.source, "source");
  string(source.repository, "source.repository");
  if (!/^[a-f0-9]{40}$/.test(string(source.commit, "source.commit"))) {
    fail("source.commit must be a lowercase 40-character Git commit");
  }
  hash256(source.evaluator_sha256, "source.evaluator_sha256");

  const dataset = object(recipe.dataset, "dataset");
  string(dataset.contract, "dataset.contract");
  string(dataset.dataset_hash, "dataset.dataset_hash");
  string(dataset.manifest, "dataset.manifest");
  uint(dataset.heldout_targets, "dataset.heldout_targets", { minimum: 1 });

  const run = object(recipe.run, "run");
  integer(run.max_compute_units, "run.max_compute_units", { positive: true });
  string(run.compute_unit, "run.compute_unit");
  const stages = array(run.stages, "run.stages");
  if (stages.length === 0) {
    fail("run.stages must not be empty");
  }
  unique(
    stages.map((stage, index) => {
      object(stage, `run.stages[${index}]`);
      const id = string(stage.id, `run.stages[${index}].id`);
      if (!SLUG_PATTERN.test(id)) {
        fail(`run.stages[${index}].id must be a lowercase kebab-case slug`);
      }
      string(stage.command, `run.stages[${index}].command`);
      integer(stage.compute_units, `run.stages[${index}].compute_units`, { positive: true });
      return id;
    }),
    "run.stages",
  );
  const stageCompute = stages.reduce(
    (sum, stage) => sum + BigInt(stage.compute_units),
    0n,
  );
  if (stageCompute > BigInt(run.max_compute_units)) {
    fail("sum of stage compute_units exceeds run.max_compute_units");
  }

  const bounties = array(recipe.bounties, "bounties");
  if (bounties.length === 0) {
    fail("bounties must not be empty");
  }
  unique(
    bounties.map((bounty, index) => {
      object(bounty, `bounties[${index}]`);
      const id = string(bounty.id, `bounties[${index}].id`);
      if (!SLUG_PATTERN.test(id)) {
        fail(`bounties[${index}].id must be a lowercase kebab-case slug`);
      }
      const sponsor = string(bounty.sponsor, `bounties[${index}].sponsor`);
      if (!ACCOUNT_PATTERN.test(sponsor)) {
        fail(`bounties[${index}].sponsor must use the nsrl: account namespace`);
      }
      integer(bounty.escrow_units, `bounties[${index}].escrow_units`, { positive: true });
      string(bounty.metric, `bounties[${index}].metric`);
      const direction = string(bounty.direction, `bounties[${index}].direction`);
      if (!["minimize", "maximize"].includes(direction)) {
        fail(`bounties[${index}].direction must be minimize or maximize`);
      }
      const baseline = integer(bounty.baseline, `bounties[${index}].baseline`);
      const target = integer(bounty.target, `bounties[${index}].target`);
      if (direction === "minimize" && target >= baseline) {
        fail(`bounties[${index}] minimize target must be lower than baseline`);
      }
      if (direction === "maximize" && target <= baseline) {
        fail(`bounties[${index}] maximize target must be higher than baseline`);
      }
      const payout = object(bounty.payout, `bounties[${index}].payout`);
      if (payout.kind !== "linear-with-promotion-bonus") {
        fail(`bounties[${index}].payout.kind is not supported in v1`);
      }
      const promotionBonusBps = uint(
        payout.promotion_bonus_bps,
        `bounties[${index}].payout.promotion_bonus_bps`,
        { maximum: 10_000 },
      );
      if (promotionBonusBps >= 10_000) {
        fail(`bounties[${index}] must reserve a nonzero metric-progress payout`);
      }
      const guardrails = array(bounty.guardrails, `bounties[${index}].guardrails`);
      if (guardrails.length === 0) {
        fail(`bounties[${index}] must include at least one guardrail`);
      }
      for (const [guardrailIndex, guardrail] of guardrails.entries()) {
        object(guardrail, `bounties[${index}].guardrails[${guardrailIndex}]`);
        string(guardrail.metric, `bounties[${index}].guardrails[${guardrailIndex}].metric`);
        if (!["lt", "lte", "eq", "gte", "gt"].includes(guardrail.operator)) {
          fail(`bounties[${index}].guardrails[${guardrailIndex}].operator is invalid`);
        }
        integer(guardrail.value, `bounties[${index}].guardrails[${guardrailIndex}].value`);
      }
      return id;
    }),
    "bounties",
  );

  const promotion = object(recipe.promotion, "promotion");
  string(promotion.contract, "promotion.contract");
  string(promotion.checker_command, "promotion.checker_command");
  hash256(promotion.checker_sha256, "promotion.checker_sha256");
  if (promotion.checker_sha256 !== source.evaluator_sha256) {
    fail("promotion.checker_sha256 must match source.evaluator_sha256");
  }
  if (promotion.valid_exit_code !== 0) {
    fail("promotion.valid_exit_code must be 0");
  }

  const rewards = object(recipe.rewards, "rewards");
  const asset = object(rewards.asset, "rewards.asset");
  if (!/^[A-Z][A-Z0-9]{2,11}$/.test(string(asset.symbol, "rewards.asset.symbol"))) {
    fail("rewards.asset.symbol must be 3-12 uppercase letters or digits");
  }
  string(asset.name, "rewards.asset.name");
  if (asset.decimals !== 0) {
    fail("rewards.asset.decimals must be zero in v1");
  }
  const maxSupply = integer(asset.max_supply_units, "rewards.asset.max_supply_units", { positive: true });
  if (asset.transferability !== "disabled-v1") {
    fail("rewards.asset.transferability must be disabled-v1");
  }
  const utility = array(asset.utility, "rewards.asset.utility");
  if (utility.length === 0) {
    fail("rewards.asset.utility must not be empty");
  }
  utility.forEach((item, index) => string(item, `rewards.asset.utility[${index}]`));

  const emission = object(rewards.emission, "rewards.emission");
  const initialReward = integer(
    emission.initial_block_reward_units,
    "rewards.emission.initial_block_reward_units",
    { positive: true },
  );
  const minimumReward = integer(
    emission.minimum_block_reward_units,
    "rewards.emission.minimum_block_reward_units",
    { positive: true },
  );
  uint(emission.halving_interval_blocks, "rewards.emission.halving_interval_blocks", { minimum: 1 });
  if (minimumReward > initialReward || initialReward > maxSupply) {
    fail("reward emission bounds are inconsistent with max supply");
  }
  const multipliers = object(emission.event_multipliers_bps, "rewards.emission.event_multipliers_bps");
  for (const event of REWARD_EVENTS) {
    uint(multipliers[event], `rewards.emission.event_multipliers_bps.${event}`, {
      minimum: 1,
      maximum: 100_000,
    });
  }

  const allocation = object(rewards.allocation_bps, "rewards.allocation_bps");
  let allocationTotal = 0;
  for (const role of REWARD_ROLES) {
    allocationTotal += uint(allocation[role], `rewards.allocation_bps.${role}`, {
      maximum: 10_000,
    });
  }
  if (allocationTotal !== 10_000) {
    fail("rewards.allocation_bps must sum to 10000");
  }

  if (recipe.publication === null) {
    if (launch.status === "promoted") {
      fail("a promoted launch must include publication evidence");
    }
  } else {
    const publication = object(recipe.publication, "publication");
    if (!MODEL_HASH_PATTERN.test(string(publication.model_hash, "publication.model_hash"))) {
      fail("publication.model_hash must be a 64-bit 0x-prefixed model hash");
    }
    hash256(publication.artifact_sha256, "publication.artifact_sha256");
    string(publication.artifact_path, "publication.artifact_path");
    hash256(publication.proof_sha256, "publication.proof_sha256");
    const metrics = object(publication.metrics, "publication.metrics");
    if (Object.keys(metrics).length === 0) {
      fail("publication.metrics must not be empty");
    }
    for (const [metric, value] of Object.entries(metrics)) {
      string(metric, "publication metric name");
      integer(value, `publication.metrics.${metric}`);
    }
    for (const bounty of bounties) {
      if (!(bounty.metric in metrics)) {
        fail(`publication.metrics is missing bounty metric ${bounty.metric}`);
      }
      for (const guardrail of bounty.guardrails) {
        if (!(guardrail.metric in metrics)) {
          fail(`publication.metrics is missing guardrail metric ${guardrail.metric}`);
        }
      }
    }
  }

  return {
    schema: RECIPE_SCHEMA,
    launch_id: launchId,
    status: launch.status,
    mode: launch.mode,
    bounties: bounties.length,
    stages: stages.length,
    max_compute_units: run.max_compute_units,
    reward_symbol: asset.symbol,
    max_supply_units: asset.max_supply_units,
    recipe_sha256: sha256Canonical(recipe),
    valid: true,
  };
}

function rewardAtHeight(recipe, event, height, cumulativeSupplyUnits) {
  if (!REWARD_EVENTS.includes(event)) {
    fail(`unknown reward event ${event}`);
  }
  const emission = recipe.rewards.emission;
  const interval = BigInt(emission.halving_interval_blocks);
  const era = BigInt(height) / interval;
  let subsidy = BigInt(emission.initial_block_reward_units);
  const minimum = BigInt(emission.minimum_block_reward_units);
  subsidy = era >= 256n ? 0n : subsidy >> era;
  if (subsidy < minimum) {
    subsidy = minimum;
  }
  const multiplier = BigInt(emission.event_multipliers_bps[event]);
  const requested = (subsidy * multiplier) / 10_000n;
  const maxSupply = BigInt(recipe.rewards.asset.max_supply_units);
  const cumulative = BigInt(cumulativeSupplyUnits);
  if (cumulative > maxSupply) {
    fail("cumulative supply exceeds max supply");
  }
  const remaining = maxSupply - cumulative;
  const minted = requested < remaining ? requested : remaining;

  const rows = REWARD_ROLES.map((role, order) => {
    const basisPoints = BigInt(recipe.rewards.allocation_bps[role]);
    const numerator = minted * basisPoints;
    return {
      role,
      account: recipe.participants[role],
      basis_points: Number(basisPoints),
      units: numerator / 10_000n,
      remainder: numerator % 10_000n,
      order,
    };
  });
  let assigned = rows.reduce((sum, row) => sum + row.units, 0n);
  const remainderOrder = [...rows].sort((left, right) => {
    if (left.remainder === right.remainder) {
      return left.order - right.order;
    }
    return left.remainder > right.remainder ? -1 : 1;
  });
  let cursor = 0;
  while (assigned < minted) {
    remainderOrder[cursor % remainderOrder.length].units += 1n;
    assigned += 1n;
    cursor += 1;
  }

  return {
    asset_symbol: recipe.rewards.asset.symbol,
    subsidy_units: subsidy.toString(),
    event_multiplier_bps: Number(multiplier),
    requested_units: requested.toString(),
    minted_units: minted.toString(),
    cumulative_supply_units: (cumulative + minted).toString(),
    allocations: rows.map(({ role, account, basis_points, units }) => ({
      role,
      account,
      basis_points,
      units: units.toString(),
    })),
  };
}

export function createModelPublishReceipt(
  recipe,
  {
    event = "model_promoted",
    height = 0,
    previousBlockSha256 = "0".repeat(64),
    cumulativeSupplyUnits = "0",
  } = {},
) {
  validateModelLaunchRecipe(recipe);
  if (recipe.publication === null) {
    fail("model publication requires publication evidence in the recipe");
  }
  uint(height, "reward block height");
  hash256(previousBlockSha256, "previous block SHA-256");
  integer(cumulativeSupplyUnits, "cumulative supply units");

  const reward = rewardAtHeight(recipe, event, height, cumulativeSupplyUnits);
  const block = {
    schema: REWARD_BLOCK_SCHEMA,
    height,
    previous_block_sha256: previousBlockSha256,
    launch_id: recipe.launch.id,
    recipe_sha256: sha256Canonical(recipe),
    event,
    evidence_sha256: recipe.publication.proof_sha256,
    model_hash: recipe.publication.model_hash,
    metric_values: recipe.publication.metrics,
    reward,
  };
  const blockSha256 = sha256Canonical(block);

  return {
    schema: PUBLISH_RECEIPT_SCHEMA,
    launch_id: recipe.launch.id,
    model_hash: recipe.publication.model_hash,
    artifact_sha256: recipe.publication.artifact_sha256,
    recipe_sha256: sha256Canonical(recipe),
    status: recipe.launch.status,
    reward_block: {
      ...block,
      block_sha256: blockSha256,
    },
  };
}

export function validateModelPublishReceipt(recipe, receipt) {
  validateModelLaunchRecipe(recipe);
  object(receipt, "publish receipt");
  if (receipt.schema !== PUBLISH_RECEIPT_SCHEMA) {
    fail(`publish receipt schema must be ${PUBLISH_RECEIPT_SCHEMA}`);
  }
  const block = object(receipt.reward_block, "publish receipt reward_block");
  const expected = createModelPublishReceipt(recipe, {
    event: block.event,
    height: block.height,
    previousBlockSha256: block.previous_block_sha256,
    cumulativeSupplyUnits: (
      BigInt(block.reward.cumulative_supply_units) - BigInt(block.reward.minted_units)
    ).toString(),
  });
  if (canonicalJson(receipt) !== canonicalJson(expected)) {
    fail("publish receipt does not match deterministic recipe settlement");
  }
  return {
    schema: PUBLISH_RECEIPT_SCHEMA,
    launch_id: receipt.launch_id,
    model_hash: receipt.model_hash,
    block_height: block.height,
    block_sha256: block.block_sha256,
    minted_units: block.reward.minted_units,
    cumulative_supply_units: block.reward.cumulative_supply_units,
    valid: true,
  };
}

export function evaluateBountyGuardrails(bounty, metricValues) {
  const checks = bounty.guardrails.map((guardrail) => {
    const observed = metricValues?.[guardrail.metric];
    if (observed === undefined) {
      return { metric: guardrail.metric, passed: false, reason: "missing" };
    }
    const left = BigInt(observed);
    const right = BigInt(guardrail.value);
    const passed =
      (guardrail.operator === "lt" && left < right) ||
      (guardrail.operator === "lte" && left <= right) ||
      (guardrail.operator === "eq" && left === right) ||
      (guardrail.operator === "gte" && left >= right) ||
      (guardrail.operator === "gt" && left > right);
    return {
      metric: guardrail.metric,
      operator: guardrail.operator,
      expected: guardrail.value,
      observed: observed.toString(),
      passed,
    };
  });
  return { passed: checks.every((check) => check.passed), checks };
}

export function bountyPayoutUnits(bounty, candidateValue, promoted, metricValues) {
  if (!evaluateBountyGuardrails(bounty, metricValues).passed) {
    return "0";
  }
  const baseline = BigInt(bounty.baseline);
  const target = BigInt(bounty.target);
  const candidate = BigInt(candidateValue);
  const escrow = BigInt(bounty.escrow_units);
  const bonusBps = BigInt(bounty.payout.promotion_bonus_bps);
  const progressPool = (escrow * (10_000n - bonusBps)) / 10_000n;
  const bonusPool = escrow - progressPool;

  let numerator;
  let denominator;
  if (bounty.direction === "minimize") {
    numerator = baseline > candidate ? baseline - candidate : 0n;
    denominator = baseline - target;
  } else {
    numerator = candidate > baseline ? candidate - baseline : 0n;
    denominator = target - baseline;
  }
  const bounded = numerator > denominator ? denominator : numerator;
  const progress = denominator === 0n ? 0n : (progressPool * bounded) / denominator;
  return (progress + (promoted ? bonusPool : 0n)).toString();
}
