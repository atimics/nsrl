#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const checker = new URL("./check-production-training-liveness-v1.mjs", import.meta.url).pathname;
const groups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];

function traceFor(interval, { outputMoved = false, fullGradientPath = false,
  trunkMoved = false, residualSaturation = 0, initialModel = `model-${interval}` } = {}) {
  const active = new Set(fullGradientPath ? groups : ["final_rms", "output", "bias"]);
  return {
    training: { optimizer_steps: 4 },
    cursor: { start_window: interval * 16, next_window: (interval + 1) * 16 },
    movement_l1: { output: outputMoved ? 1 : 0 },
    moved_parameter_groups: [
      ...(outputMoved ? ["output"] : []),
      ...(trunkMoved ? ["q"] : []),
    ],
    diagnostics: {
      gradient_nonzero_count: Object.fromEntries(groups.map((group) => [group, active.has(group) ? 1 : 0])),
      saturation_by_group: Object.fromEntries(groups.map((group) => [group, 0])),
      residual_saturation_by_group: Object.fromEntries(groups.map((group) => [group, 0])),
    },
    health: {
      gradient_saturation_count: 0,
      residual_saturation_count: residualSaturation,
      weight_saturation_count: 0,
    },
    hashes: { initial_model: initialModel, final_model: `model-${interval + 1}` },
  };
}

async function runInterval(directory, interval, trace, extraArgs = []) {
  const tracePath = path.join(directory, `trace-${interval}.json`);
  const statePath = path.join(directory, `state-${interval}.json`);
  const eventPath = path.join(directory, `event-${interval}.json`);
  await writeFile(tracePath, `${JSON.stringify(trace)}\n`);
  const args = [checker, "--trace", tracePath, "--state-out", statePath,
    "--event-out", eventPath, "--interval", String(interval),
    "--output-unlock-deadline-intervals", "4",
    "--trunk-activation-deadline-intervals", "3", ...extraArgs];
  if (interval > 0) args.push("--state-in", path.join(directory, `state-${interval - 1}.json`));
  const result = spawnSync(process.execPath, args, { encoding: "utf8" });
  const event = result.stdout.trim() ? JSON.parse(result.stdout.trim()) : null;
  return { ...result, event, statePath };
}

const root = await mkdtemp(path.join(tmpdir(), "nsrl-liveness-self-test-"));
try {
  const healthyDir = path.join(root, "healthy");
  await mkdir(healthyDir);
  for (let interval = 0; interval < 7; interval += 1) {
    const result = await runInterval(healthyDir, interval, traceFor(interval, {
      outputMoved: interval === 3,
      fullGradientPath: interval >= 6,
    }));
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.event.dead, false);
  }
  assert.equal(JSON.parse(await readFile(path.join(healthyDir, "event-3.json"))).phase,
    "trunk_activation");
  assert.equal(JSON.parse(await readFile(path.join(healthyDir, "event-6.json"))).phase,
    "trunk_live");

  const starvationDir = path.join(root, "starvation");
  await mkdir(starvationDir);
  let starvation;
  for (let interval = 0; interval < 8; interval += 1) {
    starvation = await runInterval(starvationDir, interval, traceFor(interval, {
      outputMoved: interval === 3,
      fullGradientPath: interval >= 6,
    }), ["--require-trunk-update-by-interval", "7"]);
    assert.equal(starvation.status, interval === 7 ? 3 : 0, starvation.stderr);
  }
  assert.equal(starvation.status, 3);
  assert.equal(starvation.event.classification, "trunk_update_timeout");

  const continued = await runInterval(starvationDir, 8,
    traceFor(8, { fullGradientPath: true }), ["--require-trunk-update-by-interval", "7"]);
  assert.notEqual(continued.status, 0);
  assert.match(continued.stderr, /does not bind the next model interval/);

  const lockedDir = path.join(root, "locked");
  await mkdir(lockedDir);
  for (let interval = 0; interval < 4; interval += 1) {
    const result = await runInterval(lockedDir, interval, traceFor(interval));
    assert.equal(result.status, interval === 3 ? 3 : 0, result.stderr);
    assert.equal(result.event.classification, interval === 3 ? "output_unlock_timeout" : "live");
  }

  const saturationDir = path.join(root, "saturation");
  await mkdir(saturationDir);
  const saturation = await runInterval(saturationDir, 0,
    traceFor(0, { residualSaturation: 1 }));
  assert.equal(saturation.status, 3);
  assert.equal(saturation.event.classification, "saturation");

  const staleDir = path.join(root, "stale");
  await mkdir(staleDir);
  const first = await runInterval(staleDir, 0, traceFor(0));
  assert.equal(first.status, 0, first.stderr);
  const stale = await runInterval(staleDir, 1, traceFor(1, { initialModel: "wrong-model" }));
  assert.notEqual(stale.status, 0);
  assert.match(stale.stderr, /does not bind the next model interval/);

  const policyDir = path.join(root, "policy");
  await mkdir(policyDir);
  const policyFirst = await runInterval(policyDir, 0, traceFor(0));
  assert.equal(policyFirst.status, 0, policyFirst.stderr);
  const changedPolicy = await runInterval(policyDir, 1, traceFor(1),
    ["--output-unlock-deadline-intervals", "5"]);
  assert.notEqual(changedPolicy.status, 0);
  assert.match(changedPolicy.stderr, /does not bind the next model interval/);

  console.log(JSON.stringify({ passed: true, checks: 29 }));
} finally {
  await rm(root, { recursive: true, force: true });
}
