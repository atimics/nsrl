#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const check = args[0] === "--check";
if (check) args.shift();
const contractPath = args[0]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json";
const resultPath = args[1]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-result.json";
const executionDirectory = args[2]
  ?? "data/experiments/production-model-v1/p10m-adaptive-composition-v1/execution";
const replayDirectory = args[3] ?? "/tmp/nsrl-adaptive-composition-v1-replay";
const outputPath = args[4]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-replay-receipt.json";
if (args.length > 5) throw new Error("too many adaptive replay-recorder arguments");

const resolve = (value) => path.isAbsolute(value) ? value : path.join(root, value);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const bind = (value) => {
  const bytes = fs.readFileSync(resolve(value));
  return {path: value, sha256: sha256(bytes), bytes: bytes.length};
};
const assert = (condition, message) => {
  if (!condition) throw new Error(`adaptive replay receipt: ${message}`);
};
const contractBytes = fs.readFileSync(resolve(contractPath));
const resultBytes = fs.readFileSync(resolve(resultPath));
const contract = JSON.parse(contractBytes);
const result = JSON.parse(resultBytes);
assert(contract.schema === "nsrl.adaptive_composition_execution_contract.v1"
  && contract.analysis_role === "frozen_after_calibration_before_adaptive_endpoint",
"wrong frozen execution contract");
assert(result.schema === "nsrl.adaptive_composition_result.v1"
  && result.analysis_role === "preregistered_fresh_source_execution",
"wrong adaptive result");
const checker = bind(contract.bindings.checker.path);
assert(checker.sha256 === contract.bindings.checker.sha256
  && checker.bytes === contract.bindings.checker.bytes,
"frozen execution checker changed");

const checkerOutput = execFileSync(process.execPath, [
  resolve(contract.bindings.checker.path), resolve(contractPath), resolve(resultPath),
  resolve(executionDirectory), resolve(replayDirectory),
], {cwd: root, encoding: "utf8"});
const summary = JSON.parse(checkerOutput);
assert(summary.schema === "nsrl.adaptive_composition_execution_check.v1"
  && summary.ok === true && summary.byte_replay === true
  && summary.verdict === result.verdict,
"frozen execution checker did not confirm exact replay");

const replayArtifacts = [
  "calibration-manifest.json", "calibration-cube.tsv", "calibration-scores.tsv",
  "corrections.tsv", "decisions.tsv", "adaptive-final.nsrlpm",
  "always-abstain-final.nsrlpm", "head-only-final.nsrlpm", "trunk-only-final.nsrlpm",
].map((name) => {
  const executionBytes = fs.readFileSync(path.join(resolve(executionDirectory), name));
  const replayBytes = fs.readFileSync(path.join(resolve(replayDirectory), name));
  assert(executionBytes.equals(replayBytes), `replay artifact differs: ${name}`);
  return {name, bytes: executionBytes.length, sha256: sha256(executionBytes), identical: true};
});
const replayResultBytes = fs.readFileSync(path.join(resolve(replayDirectory), "result.json"));
assert(resultBytes.equals(replayResultBytes), "result JSON differs from exact replay");

const receipt = {
  schema: "nsrl.adaptive_composition_replay_receipt.v1",
  experiment_id: contract.experiment_id,
  sources: {
    contract: {path: contractPath, sha256: sha256(contractBytes), bytes: contractBytes.length},
    result: {path: resultPath, sha256: sha256(resultBytes), bytes: resultBytes.length},
    frozen_checker: checker,
    recorder: bind(path.relative(root, fileURLToPath(import.meta.url))),
  },
  verdict: result.verdict,
  execution_summary: summary,
  replay_artifacts: replayArtifacts,
  result_replay: {
    bytes: resultBytes.length,
    sha256: sha256(resultBytes),
    identical: true,
  },
  guarantees: {
    calibration_byte_replay: true,
    decision_trace_byte_replay: true,
    retained_model_byte_replay: true,
    result_json_byte_replay: true,
    post_outcome_threshold_change: false,
  },
};
const bytes = Buffer.from(`${JSON.stringify(receipt, null, 2)}\n`);
const absoluteOutput = resolve(outputPath);
if (check) {
  assert(fs.existsSync(absoluteOutput) && fs.readFileSync(absoluteOutput).equals(bytes),
    "tracked replay receipt does not byte-replay");
} else {
  assert(!fs.existsSync(absoluteOutput), "refusing to overwrite replay receipt");
  fs.mkdirSync(path.dirname(absoluteOutput), {recursive: true});
  fs.writeFileSync(absoluteOutput, bytes, {flag: "wx"});
}
process.stdout.write(`${JSON.stringify({
  schema: receipt.schema,
  checked: check,
  verdict: receipt.verdict,
  artifacts: receipt.replay_artifacts.length + 1,
  byte_replay: true,
  output: outputPath,
}, null, 2)}\n`);
