#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  bountyPayoutUnits,
  canonicalJson,
  createModelPublishReceipt,
  evaluateBountyGuardrails,
  validateModelLaunchRecipe,
  validateModelPublishReceipt,
} from "./lib/model-launch-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const RECIPE_PATH = path.join(
  ROOT,
  "protocol/examples/integer-transformer-proof-v1.launch.json",
);
const RECEIPT_PATH = path.join(
  ROOT,
  "protocol/examples/integer-transformer-proof-v1.publish.json",
);

function sha256File(filePath) {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function expectRejected(mutator) {
  const recipe = JSON.parse(fs.readFileSync(RECIPE_PATH, "utf8"));
  mutator(recipe);
  assert.throws(() => validateModelLaunchRecipe(recipe), /model launch v1:/);
}

function main() {
  const recipe = JSON.parse(fs.readFileSync(RECIPE_PATH, "utf8"));
  const receipt = JSON.parse(fs.readFileSync(RECEIPT_PATH, "utf8"));
  const recipeCheck = validateModelLaunchRecipe(recipe);
  const receiptCheck = validateModelPublishReceipt(recipe, receipt);

  const expectedReceipt = createModelPublishReceipt(recipe);
  assert.equal(canonicalJson(receipt), canonicalJson(expectedReceipt));

  const replayReceipt = createModelPublishReceipt(recipe, {
    event: "independent_replay",
    height: 8,
    previousBlockSha256: receipt.reward_block.block_sha256,
    cumulativeSupplyUnits: receipt.reward_block.reward.cumulative_supply_units,
  });
  assert.equal(replayReceipt.reward_block.reward.subsidy_units, "16000");
  assert.equal(replayReceipt.reward_block.reward.minted_units, "8000");
  assert.equal(
    replayReceipt.reward_block.reward.allocations.reduce(
      (sum, allocation) => sum + BigInt(allocation.units),
      0n,
    ),
    8000n,
  );

  const cappedReceipt = createModelPublishReceipt(recipe, {
    event: "model_promoted",
    height: 1,
    previousBlockSha256: receipt.reward_block.block_sha256,
    cumulativeSupplyUnits: "999500",
  });
  assert.equal(cappedReceipt.reward_block.reward.minted_units, "500");
  assert.equal(cappedReceipt.reward_block.reward.cumulative_supply_units, "1000000");

  const draftRecipe = structuredClone(recipe);
  draftRecipe.launch.status = "draft";
  draftRecipe.publication = null;
  assert.equal(validateModelLaunchRecipe(draftRecipe).valid, true);
  assert.throws(() => createModelPublishReceipt(draftRecipe), /requires publication evidence/);

  const artifactPath = path.join(ROOT, recipe.publication.artifact_path);
  assert.equal(sha256File(artifactPath), recipe.publication.artifact_sha256);
  const proofPath = path.join(
    ROOT,
    "data/experiments/integer-transformer-proof-v1/candidate-default/proof-check.json",
  );
  assert.equal(sha256File(proofPath), recipe.publication.proof_sha256);
  const evaluatorPath = path.join(ROOT, "crates/nsrl-eval/src/contract.rs");
  assert.equal(sha256File(evaluatorPath), recipe.source.evaluator_sha256);

  const freeze = JSON.parse(
    fs.readFileSync(
      path.join(ROOT, "benchmarks/integer-transformer-proof-v1/promoted-candidate.json"),
      "utf8",
    ),
  );
  assert.equal(freeze.status, "promoted");
  assert.equal(freeze.model_hash, recipe.publication.model_hash);
  assert.equal(
    freeze.metrics.probability_error_q15.toString(),
    recipe.publication.metrics.probability_error_q15,
  );
  assert.equal(freeze.metrics.mistakes.toString(), recipe.publication.metrics.mistakes);

  const bounty = recipe.bounties[0];
  const settledBountyUnits = bountyPayoutUnits(
    bounty,
    recipe.publication.metrics[bounty.metric],
    true,
    recipe.publication.metrics,
  );
  assert.equal(settledBountyUnits, bounty.escrow_units);
  assert.equal(evaluateBountyGuardrails(bounty, recipe.publication.metrics).passed, true);
  const regressedMetrics = { ...recipe.publication.metrics, unique_predicted_tokens: "0" };
  assert.equal(
    bountyPayoutUnits(bounty, recipe.publication.metrics[bounty.metric], true, regressedMetrics),
    "0",
  );

  expectRejected((candidate) => {
    candidate.rewards.allocation_bps.builder += 1;
  });
  expectRejected((candidate) => {
    candidate.bounties[0].target = candidate.bounties[0].baseline;
  });
  expectRejected((candidate) => {
    candidate.promotion.checker_sha256 = "0".repeat(64);
  });
  const forgedReceipt = structuredClone(receipt);
  forgedReceipt.reward_block.reward.minted_units = (
    BigInt(forgedReceipt.reward_block.reward.minted_units) + 1n
  ).toString();
  assert.throws(
    () => validateModelPublishReceipt(recipe, forgedReceipt),
    /model launch v1:/,
  );

  process.stdout.write(
    `${JSON.stringify({
      schema: "nsrl.model_launch_check.v1",
      recipe: recipeCheck,
      publication: receiptCheck,
      bounty: {
        id: bounty.id,
        settled_units: settledBountyUnits,
        fully_settled: settledBountyUnits === bounty.escrow_units,
      },
      reward_cases: 3,
      negative_cases: 5,
      valid: true,
    })}\n`,
  );
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
}
