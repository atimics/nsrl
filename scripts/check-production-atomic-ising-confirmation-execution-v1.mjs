#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json";
const sourcePath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const freezerPath = new URL(
  "./freeze-production-atomic-ising-confirmation-v1.mjs", import.meta.url);
const proposalSourcePath =
  "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const proposalIsingPath =
  "benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json";
const structureContractPath =
  "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1-contract.json";
const contractBytes = fs.readFileSync(contractPath);
const contract = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const source = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const bindings = [
  ["scripts/check-production-atomic-structure-v1.mjs", "structure_checker_sha256"],
  ["scripts/analyze-production-atomic-ising-confirmation-v1.mjs", "analyzer_sha256"],
  ["scripts/check-production-atomic-ising-confirmation-v1.mjs", "checker_sha256"],
];
assert(contract.schema === "nsrl.production_atomic_ising_confirmation_contract.v1",
  "wrong confirmation contract schema");
for (const [file, key] of bindings) {
  assert(sha256(file) === contract.implementation[key], `${key} mismatch`);
}
assert(source.bindings.source_fnv64 === contract.execution.source_fnv64
  && source.bindings.binary_fnv64 === contract.execution.binary_fnv64
  && source.bindings.manifest_hash === contract.execution.structure_manifest_hash,
"executed source bindings changed");
assert(source.surface.document_start === contract.surface.document_start
  && source.surface.documents === contract.surface.documents
  && source.surface.hard_stop_before_document === contract.surface.hard_stop_before_document,
"executed surface changed");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 64,
  "document accounting changed");
assert(contract.replay_scope.frozen_structure_cube_reexecuted === false
  && contract.replay_scope.frozen_structure_cube_independently_reconstructed === true
  && contract.replay_scope.derived_confirmation_byte_replayed === true,
"clean-checkout replay scope changed");
const replayDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-ising-contract-replay-"));
const replayPath = path.join(replayDirectory, "contract.json");
try {
  const replay = spawnSync(
    process.execPath,
    [
      fileURLToPath(freezerPath),
      proposalSourcePath,
      proposalIsingPath,
      structureContractPath,
      replayPath,
    ],
    {encoding: "utf8"},
  );
  assert(replay.status === 0, `confirmation contract replay failed: ${replay.stderr || replay.stdout}`);
  assert(fs.readFileSync(replayPath).equals(contractBytes),
    "confirmation contract is not byte-replayable");
} finally {
  fs.rmSync(replayDirectory, {recursive: true, force: true});
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_confirmation_execution_check.v1",
  implementation_bindings_checked: bindings.length,
  source_fnv64: source.bindings.source_fnv64,
  binary_fnv64: source.bindings.binary_fnv64,
  structure_manifest_hash: source.bindings.manifest_hash,
  hard_stop_before_document: source.surface.hard_stop_before_document,
  confirmation_contract_byte_replay_verified: true,
  documents_200_212_read: false,
}, null, 2)}\n`);
