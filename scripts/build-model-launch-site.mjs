#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  bountyPayoutUnits,
  validateModelLaunchRecipe,
  validateModelPublishReceipt,
} from "./lib/model-launch-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const DEFAULT_OUT = path.join(ROOT, "web/launches/network.json");

function parseArgs(argv) {
  let out = DEFAULT_OUT;
  let check = false;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--out" && argv[index + 1]) {
      out = path.resolve(argv[index + 1]);
      index += 1;
    } else if (argv[index] === "--check") {
      check = true;
    } else {
      throw new Error(`unknown or incomplete argument ${argv[index]}`);
    }
  }
  return { out, check };
}

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, relativePath), "utf8"));
}

function buildData() {
  const recipe = readJson("protocol/examples/integer-transformer-proof-v1.launch.json");
  const receipt = readJson("protocol/examples/integer-transformer-proof-v1.publish.json");
  const freeze = readJson("benchmarks/integer-transformer-proof-v1/promoted-candidate.json");
  validateModelLaunchRecipe(recipe);
  validateModelPublishReceipt(recipe, receipt);

  const bounty = recipe.bounties[0];
  const achieved = BigInt(recipe.publication.metrics[bounty.metric]);
  const baseline = BigInt(bounty.baseline);
  const target = BigInt(bounty.target);
  const improvement =
    bounty.direction === "minimize" ? baseline - achieved : achieved - baseline;
  const improvementBps = baseline === 0n ? 0n : (improvement * 10_000n) / baseline;
  const nextTarget = (achieved * 9_000n) / 10_000n;
  const settledBounty = bountyPayoutUnits(
    bounty,
    achieved,
    true,
    recipe.publication.metrics,
  );

  return {
    schema: "nsrl.model_launch_site_data.v1",
    generated_from: recipe.launch.published_at,
    network: {
      name: "NSRL Forge",
      id: recipe.launch.network,
      mode: recipe.launch.mode,
      notice: "Protocol preview · simulated credits · no wallet or financial settlement",
      published_models: 1,
      proof_blocks: 1,
      minted_units: receipt.reward_block.reward.minted_units,
      reward_symbol: recipe.rewards.asset.symbol,
    },
    launch: {
      id: recipe.launch.id,
      title: recipe.launch.title,
      summary: recipe.launch.summary,
      status: recipe.launch.status,
      family: recipe.model.family,
      architecture: recipe.model.architecture_profile,
      dataset_hash: recipe.dataset.dataset_hash,
      model_hash: recipe.publication.model_hash,
      artifact_sha256: recipe.publication.artifact_sha256,
      targets: recipe.publication.metrics.targets,
    },
    bounty: {
      id: bounty.id,
      metric: bounty.metric,
      direction: bounty.direction,
      baseline: bounty.baseline,
      target: bounty.target,
      achieved: achieved.toString(),
      escrow_units: bounty.escrow_units,
      settled_units: settledBounty,
      improvement_bps: improvementBps.toString(),
      promotion_bonus_bps: bounty.payout.promotion_bonus_bps,
      guardrails: bounty.guardrails,
    },
    next_bounty: {
      metric: bounty.metric,
      direction: bounty.direction,
      baseline: achieved.toString(),
      target: nextTarget.toString(),
      improvement_bps: "1000",
      escrow_units: "120000",
      promotion_bonus_bps: bounty.payout.promotion_bonus_bps,
    },
    publication: {
      status: freeze.status,
      proof_contract: freeze.contract,
      block: receipt.reward_block,
      reward_asset: recipe.rewards.asset,
      emission: recipe.rewards.emission,
      allocation_bps: recipe.rewards.allocation_bps,
    },
    gaps: [
      {
        capability: "Frozen evaluation",
        status: "ready",
        label: "Existing",
        detail: "Typed pass/fail checker, fixed baselines, dataset hash, and replay evidence.",
      },
      {
        capability: "Model recipe",
        status: "prototype",
        label: "Prototype",
        detail: "Versioned schema binds source, data, run stages, bounty, promotion, and rewards.",
      },
      {
        capability: "Publication ledger",
        status: "prototype",
        label: "Prototype",
        detail: "Signed append-only events, hash links, replay protection, and exact capped allocation run locally.",
      },
      {
        capability: "Sponsor escrow",
        status: "prototype",
        label: "Test adapter",
        detail: "Conserved test balances enforce funding, deadlines, payout, refund, expiry, and slashing without custody.",
      },
      {
        capability: "Compute market",
        status: "prototype",
        label: "Auction",
        detail: "Sealed bids, deterministic price clearing, collateral, signed meters, and accepted-stage payment replay locally.",
      },
      {
        capability: "Bounty automation",
        status: "prototype",
        label: "Keeper",
        detail: "Sponsor-signed budgets, cooldowns, cycle caps, pause controls, approvals, and resumable reserved funding open exact promoted successors.",
      },
      {
        capability: "Validator network",
        status: "prototype",
        label: "Localnet",
        detail: "Independent keys sign clean stage/candidate quorums with full replay and authority-resolved challenges.",
      },
      {
        capability: "Identity + signatures",
        status: "prototype",
        label: "Ed25519",
        detail: "Registered accounts bind Ed25519 keys to every intent; rotation, delegation, and recovery remain open.",
      },
      {
        capability: "Artifact availability",
        status: "partial",
        label: "Partial",
        detail: "Hashes and S3 provenance exist; replicated content-addressed retention does not.",
      },
    ],
    recipe,
    receipt,
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const output = `${JSON.stringify(buildData(), null, 2)}\n`;
  if (options.check) {
    if (!fs.existsSync(options.out) || fs.readFileSync(options.out, "utf8") !== output) {
      throw new Error(`${path.relative(ROOT, options.out)} is stale; rebuild it with scripts/build-model-launch-site.mjs`);
    }
    process.stdout.write(`${path.relative(ROOT, options.out)} is current\n`);
    return;
  }
  fs.mkdirSync(path.dirname(options.out), { recursive: true });
  fs.writeFileSync(options.out, output);
  process.stdout.write(`${path.relative(ROOT, options.out)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
