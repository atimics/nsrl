#!/usr/bin/env node

import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  auditExperiment,
  decideExperiment,
  freezeExperiment,
  harnessStatus,
  importCompletedExperiment,
  nextActions,
  registerExperiment,
  reviewExperiment,
  runExperiment,
  validateExperimentSpec,
  verifyLedger,
} from "./lib/research-harness-v1.mjs";

const realRepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "nsrl-research-harness-"));
const repoRoot = path.join(temporaryRoot, "repo");
const policyRoot = path.join(repoRoot, "research", "harness");
const stateRoot = path.join(repoRoot, "data", "research-harness");
await mkdir(policyRoot, { recursive: true });

const runnerSource = [
  "import { readFile, writeFile } from 'node:fs/promises';",
  "const input = (await readFile('input.txt', 'utf8')).trim();",
  "await writeFile('result.json', `${JSON.stringify({decision:{supported:input === 'stable-input'}})}\\n`);",
].join("\n");
const checkerSource = [
  "import { readFile } from 'node:fs/promises';",
  "const result = JSON.parse(await readFile('result.json', 'utf8'));",
  "if (typeof result.decision?.supported !== 'boolean') process.exit(1);",
].join("\n");
await writeFile(path.join(repoRoot, "input.txt"), "stable-input\n");
await writeFile(path.join(repoRoot, "runner.mjs"), runnerSource);
await writeFile(path.join(repoRoot, "checker.mjs"), checkerSource);

const policy = {
  schema: "nsrl.research_runner_templates.v1",
  templates: {
    "fixture-runner": {
      command: ["node", "runner.mjs"],
      checker: ["node", "checker.mjs"],
      allowed_partitions: ["proposal"],
      expected_outputs: ["result.json"],
      paid_compute: false,
    },
  },
};
const policyText = `${JSON.stringify(policy, null, 2)}\n`;
await writeFile(path.join(policyRoot, "runner-templates.json"), policyText);

const spec = {
  schema: "nsrl.research_experiment.v1",
  id: "fixture-experiment-v1",
  title: "Harness fixture experiment",
  summary: "Proves the lifecycle, role separation, binding checks, and decision rules.",
  parents: [],
  claim: {
    hypothesis: "The fixture input is stable.",
    estimand: "The exact Boolean result emitted by the allowlisted fixture runner.",
    falsifier: "The runner emits supported=false.",
    evidence_level: "diagnostic",
  },
  bindings: {
    files: [
      { path: "input.txt", role: "dataset" },
      { path: "runner.mjs", role: "source" },
      { path: "checker.mjs", role: "evaluator" },
    ],
  },
  evidence_access: {
    allowed_partitions: ["proposal"],
    excluded_partitions: ["reserved"],
    consumes_reserved_evidence: false,
  },
  design: {
    independent_unit: "fixture",
    planned_units: 1,
    minimum_informative_units: 1,
    family_size: 1,
    controls: ["exact expected input"],
  },
  execution: {
    runner_template: "fixture-runner",
    expected_outputs: ["result.json"],
    max_seconds: 10,
    max_output_bytes: 65536,
  },
  decision: {
    checker_template: "fixture-runner",
    result_path: "result.json",
    rules: [
      { outcome: "supported", all: [{ path: "decision.supported", operator: "eq", value: true }] },
      { outcome: "falsified", all: [{ path: "decision.supported", operator: "eq", value: false }] },
    ],
    default_outcome: "inconclusive",
  },
  authorization: {
    local_execution: true,
    reserved_evidence: false,
    optimizer_change: false,
    paid_compute: false,
  },
};
validateExperimentSpec(spec);
await writeFile(path.join(repoRoot, "experiment.json"), `${JSON.stringify(spec, null, 2)}\n`);

const proposer = { id: "agent:proposer", role: "theorist" };
const reviewer = { id: "agent:reviewer", role: "statistician" };
const protocol = { id: "agent:protocol", role: "protocol" };
const runner = { id: "agent:runner", role: "runner" };
const auditor = { id: "agent:auditor", role: "auditor" };
const curator = { id: "agent:curator", role: "curator" };

await registerExperiment({ repoRoot, stateRoot, specPath: "experiment.json", actor: proposer });
const reviewerInbox = await nextActions(stateRoot, reviewer);
assert.equal(reviewerInbox.action_count, 1);
assert.equal(reviewerInbox.actions[0].action, "review");
await assert.rejects(
  reviewExperiment({
    stateRoot,
    experimentId: spec.id,
    actor: { id: proposer.id, role: "statistician" },
    approved: true,
  }),
  /proposer cannot review/,
);
await reviewExperiment({ stateRoot, experimentId: spec.id, actor: reviewer, approved: true });
await freezeExperiment({ repoRoot, policyRoot, stateRoot, experimentId: spec.id, actor: protocol });

await writeFile(path.join(repoRoot, "input.txt"), "tampered-input\n");
await assert.rejects(
  runExperiment({ repoRoot, policyRoot, stateRoot, experimentId: spec.id, actor: runner }),
  /frozen input bindings changed/,
);
await writeFile(path.join(repoRoot, "input.txt"), "stable-input\n");

await writeFile(path.join(policyRoot, "runner-templates.json"), `${policyText}\n`);
await assert.rejects(
  runExperiment({ repoRoot, policyRoot, stateRoot, experimentId: spec.id, actor: runner }),
  /policy changed after contract freeze/,
);
await writeFile(path.join(policyRoot, "runner-templates.json"), policyText);

await runExperiment({ repoRoot, policyRoot, stateRoot, experimentId: spec.id, actor: runner });
await assert.rejects(
  auditExperiment({
    repoRoot,
    policyRoot,
    stateRoot,
    experimentId: spec.id,
    actor: { id: runner.id, role: "auditor" },
  }),
  /runner cannot audit/,
);
const audit = await auditExperiment({ repoRoot, policyRoot, stateRoot, experimentId: spec.id, actor: auditor });
assert.equal(audit.ok, true);
await assert.rejects(
  decideExperiment({
    repoRoot,
    stateRoot,
    experimentId: spec.id,
    actor: { id: auditor.id, role: "curator" },
  }),
  /prior lifecycle actor cannot curate/,
);
const decision = await decideExperiment({ repoRoot, stateRoot, experimentId: spec.id, actor: curator });
assert.equal(decision.outcome, "supported");

const status = await harnessStatus(stateRoot);
assert.equal(status.experiment_count, 1);
assert.equal(status.event_count, 7);
assert.equal(status.experiments[0].state, "supported");
assert.equal((await verifyLedger(stateRoot)).ok, true);

const reservedSpec = structuredClone(spec);
reservedSpec.id = "reserved-without-authorization-v1";
reservedSpec.evidence_access.allowed_partitions = ["reserved"];
reservedSpec.evidence_access.excluded_partitions = [];
reservedSpec.evidence_access.consumes_reserved_evidence = true;
assert.throws(() => validateExperimentSpec(reservedSpec), /not authorized/);

const eventsPath = path.join(stateRoot, "events.jsonl");
const validLedger = await readFile(eventsPath, "utf8");
await writeFile(eventsPath, validLedger.replace('"event_type":"registered"', '"event_type":"note"'));
await assert.rejects(verifyLedger(stateRoot), /invalid hash|before registration/);

const goldenStateRoot = path.join(temporaryRoot, "golden-state");
const goldenDecision = await importCompletedExperiment({
  repoRoot: realRepoRoot,
  policyRoot: path.join(realRepoRoot, "research", "harness"),
  stateRoot: goldenStateRoot,
  specPath: "research/harness/templates/p10m-boolean-jet-confirmation-v1.experiment.json",
  actors: {
    proposer: { id: "golden:proposer", role: "theorist" },
    reviewer: { id: "golden:reviewer", role: "statistician" },
    protocol: { id: "golden:protocol", role: "protocol" },
    runner: { id: "golden:runner", role: "runner" },
    auditor: { id: "golden:auditor", role: "auditor" },
    curator: { id: "golden:curator", role: "curator" },
  },
});
assert.equal(goldenDecision.outcome, "falsified");
const goldenStatus = await harnessStatus(goldenStateRoot);
assert.equal(goldenStatus.experiments[0].state, "falsified");

console.log(JSON.stringify({
  schema: "nsrl.research_harness_self_test.v1",
  ok: true,
  lifecycle_events: status.event_count,
  role_separation_verified: true,
  frozen_binding_tamper_rejected: true,
  frozen_policy_tamper_rejected: true,
  ledger_tamper_rejected: true,
  golden_experiment: {
    id: goldenDecision.experiment_id,
    outcome: goldenDecision.outcome,
  },
}, null, 2));
