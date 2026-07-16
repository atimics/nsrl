#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {execFileSync} from "node:child_process";

const publicationPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-publication.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json";
const resultPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json";
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");

const publicationBytes = fs.readFileSync(publicationPath);
const contractBytes = fs.readFileSync(contractPath);
const resultBytes = fs.readFileSync(resultPath);
const publication = JSON.parse(publicationBytes);
const contract = JSON.parse(contractBytes);
const result = JSON.parse(resultBytes);
const allowed = ["supported", "falsified", "inconclusive"];
assert(publication.schema === "nsrl.solomonic_judgment_publication.v1",
  "wrong publication schema");
assert(publication.source_sha256.contract === sha256(contractBytes)
  && publication.source_sha256.result === sha256(resultBytes),
"publication evidence binding changed");
assert(JSON.stringify(publication.publication_contract.allowed_statuses) === JSON.stringify(allowed)
  && publication.publication_contract.fail_closed_on_unknown_status === true,
"publication contract does not fail closed");
assert(allowed.includes(publication.verdict.status)
  && publication.claims.every((claim) => allowed.includes(claim.status)),
"publication contains an unauthorized status");
assert(Number(publication.verdict.supported) + Number(publication.verdict.falsified)
  + Number(publication.verdict.inconclusive) === 1,
"publication verdict is not trivalent and exclusive");
assert(publication.verdict[publication.verdict.status] === true,
  "publication verdict flag disagrees with status");
assert(publication.authorization.universal_wisdom_claimed === false
  && publication.authorization.optimizer_promotion_authorized === false
  && publication.authorization.paid_scaling_authorized === false,
"publication escaped its authorization boundary");
assert(publication.claims.find((claim) => claim.id === "occult_hash_parity_predictive_feature")
  ?.status === (contract.occult_feature.activation.activated ? "inconclusive" : "falsified"),
"occult feature did not receive its frozen falsifier");

const replayPath = path.join(os.tmpdir(), `solomonic-publication-replay-${process.pid}.json`);
try {
  execFileSync(process.execPath,
    [contract.bindings.publisher.path, contractPath, resultPath, replayPath], {stdio: "pipe"});
  assert(fs.readFileSync(replayPath).equals(publicationBytes), "publication byte replay changed");
} finally {
  fs.rmSync(replayPath, {force: true});
}
assert(result.authorization.optimizer_promotion_authorized === false,
  "result authorization changed");
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomonic_judgment_publication_check.v1",
  allowed_statuses: allowed, verdict: publication.verdict.status,
  claim_statuses: Object.fromEntries(publication.claims.map((claim) => [claim.id, claim.status])),
  fail_closed: true, byte_replay: true,
}, null, 2)}\n`);
