#!/usr/bin/env node

import { createHash } from "node:crypto";
import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let manifestPath = "data/experiments/literary-recursive-swarm-v1/experiment.manifest.json";
let outPath = null;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--manifest") manifestPath = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/check-recursive-literary-swarm-experiment.mjs [--manifest PATH] [--out PATH]");
    process.exit(0);
  } else throw new Error(`unknown argument: ${arg}`);
}

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const checks = [];
const check = (name, pass, detail) => checks.push({ name, pass: Boolean(pass), detail });

async function checkHashedFile(name, descriptor) {
  try {
    const content = await readFile(descriptor.path);
    check(`${name}.exists`, true, descriptor.path);
    check(`${name}.sha256`, sha256(content) === descriptor.sha256, descriptor.sha256);
    return content;
  } catch (error) {
    check(`${name}.exists`, false, `${descriptor?.path}: ${error.message}`);
    return null;
  }
}

async function main() {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  check("schema", manifest.schema === "nsrl.recursive_literary_swarm_experiment.v1", manifest.schema);
  check("topology", manifest.topology === "depth2_ternary_router_triads", manifest.topology);
  check("authors", JSON.stringify(manifest.authors) === JSON.stringify(["crowley", "shakespeare", "blake"]), manifest.authors);
  check("final_test_frozen", manifest.split_contract?.final_test_is_frozen === true, manifest.split_contract);
  check("router_features", manifest.router_feature_schema?.base_dimensions === 32 && manifest.router_feature_schema?.child_probe_dimensions === 9 && manifest.router_feature_schema?.routed_dimensions === 41 && manifest.router_feature_schema?.dtype === "q15_i16", manifest.router_feature_schema);
  check("leaf_job_count", manifest.leaf_jobs?.length === 9, manifest.leaf_jobs?.length);
  check("leaf_job_ids_unique", new Set(manifest.leaf_jobs?.map((job) => job.expert_id)).size === 9, manifest.leaf_jobs?.map((job) => job.expert_id));

  const expectedAuthors = new Map([["crowley", 0], ["shakespeare", 0], ["blake", 0]]);
  for (const job of manifest.leaf_jobs ?? []) {
    if (expectedAuthors.has(job.author)) expectedAuthors.set(job.author, expectedAuthors.get(job.author) + 1);
    try {
      await access(job.text_path);
      check(`leaf.${job.expert_id}.text`, true, job.text_path);
    } catch (error) {
      check(`leaf.${job.expert_id}.text`, false, error.message);
    }
    try {
      await access(job.tokens_path);
      check(`leaf.${job.expert_id}.tokens`, true, job.tokens_path);
    } catch (error) {
      check(`leaf.${job.expert_id}.tokens`, false, error.message);
    }
    check(`leaf.${job.expert_id}.config`, job.seq_len === 32 && job.max_windows === 8192 && job.stride > 0, { seq_len: job.seq_len, max_windows: job.max_windows, stride: job.stride });
  }
  check("three_leaves_per_author", [...expectedAuthors.values()].every((count) => count === 3), Object.fromEntries(expectedAuthors));

  check("router_job_count", manifest.router_jobs?.length === 12, manifest.router_jobs?.length);
  check("router_job_ids_unique", new Set(manifest.router_jobs?.map((job) => job.router_id)).size === 12, manifest.router_jobs?.map((job) => job.router_id));
  check("router_feature_views", manifest.router_jobs?.every((job) => ["semantic", "structural", "full"].includes(job.feature_view)), manifest.router_jobs?.map((job) => job.feature_view));
  check("local_routers_wait_for_oracles", manifest.router_jobs?.filter((job) => job.node_id !== "root-literary-pod").every((job) => job.target_field === "oracle_target" && job.warm_start_ready === false), "local routers cannot train from author labels");
  check("root_warm_start_only", manifest.router_jobs?.filter((job) => job.node_id === "root-literary-pod").every((job) => job.target_field === "bootstrap_target_then_oracle_target" && job.warm_start_ready === true && job.final_training_ready === false), "root author labels are not final utility labels");

  const nodes = manifest.nodes ?? [];
  check("router_node_count", nodes.length === 4, nodes.length);
  for (const node of nodes) {
    check(`node.${node.node_id}.children`, node.kind === "router_triad" && node.children?.length === 3, node.children);
    check(`node.${node.node_id}.replicas`, node.router_replicas?.length === 3 && new Set(node.router_replicas).size === 3, node.router_replicas);
    check(`node.${node.node_id}.consensus`, node.consensus === "q15_rank_sum_then_confidence" && node.beam_width === 2, { consensus: node.consensus, beam_width: node.beam_width });
  }
  const root = nodes.find((node) => node.node_id === manifest.root_node_id);
  check("root_children_are_router_nodes", root?.children?.every((id) => nodes.some((node) => node.node_id === id)), root?.children);

  for (const author of manifest.authors ?? []) {
    const ranges = [];
    for (const split of ["leaf_train", "router_train", "router_calibration", "final_test"]) {
      const descriptor = manifest.splits?.[author]?.[split];
      await checkHashedFile(`split.${author}.${split}`, descriptor);
      try {
        await access(descriptor.tokens_path);
        check(`split.${author}.${split}.tokens`, true, descriptor.tokens_path);
      } catch (error) {
        check(`split.${author}.${split}.tokens`, false, error.message);
      }
      if (descriptor?.raw_byte_range) ranges.push({ split, range: descriptor.raw_byte_range });
    }
    let contiguous = ranges.length === 4 && ranges[0].range[0] === 0;
    for (let index = 1; index < ranges.length; index += 1) {
      contiguous &&= ranges[index - 1].range[1] === ranges[index].range[0];
    }
    check(`split.${author}.nonoverlap_contiguous`, contiguous, ranges);
  }

  const datasetDescriptors = [];
  for (const [split, descriptor] of Object.entries(manifest.router_datasets?.root ?? {})) {
    datasetDescriptors.push([`router.root.${split}`, descriptor, "root"]);
  }
  for (const [author, splits] of Object.entries(manifest.router_datasets?.local ?? {})) {
    for (const [split, descriptor] of Object.entries(splits)) {
      datasetDescriptors.push([`router.local.${author}.${split}`, descriptor, "local"]);
    }
  }
  for (const [name, descriptor, kind] of datasetDescriptors) {
    const content = await checkHashedFile(name, descriptor);
    if (!content) continue;
    const rows = content.toString("utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
    check(`${name}.row_count`, rows.length === descriptor.rows && rows.length > 0, { expected: descriptor.rows, actual: rows.length });
    check(`${name}.feature_shape`, rows.every((row) => row.features_q15?.length === 32 && row.features_q15.every((value) => Number.isInteger(value) && value >= 0 && value <= 32767)), rows.length);
    check(`${name}.candidate_shape`, rows.every((row) => row.candidate_ids?.length === 3), rows.length);
    check(`${name}.oracle_unfilled`, rows.every((row) => row.oracle_target === null && row.oracle_child_losses_q15 === null), "oracle labels must come from measured child utility");
    if (kind === "root") check(`${name}.bootstrap_target`, rows.every((row) => [0, 1, 2].includes(row.bootstrap_target)), rows.length);
    else check(`${name}.bootstrap_target`, rows.every((row) => row.bootstrap_target === null), rows.length);
  }

  const failed = checks.filter((item) => !item.pass);
  let leafCheckpointsPresent = 0;
  for (const job of manifest.leaf_jobs ?? []) {
    try {
      await access(job.model_path);
      leafCheckpointsPresent += 1;
    } catch {}
  }
  const artifactExists = async (artifactPath) => {
    try {
      await access(artifactPath);
      return true;
    } catch {
      return false;
    }
  };
  const experimentDir = path.dirname(path.resolve(manifestPath));
  const localOracleReady = await artifactExists(path.join(experimentDir, "router-oracles", "oracle-report.json"));
  const rootOracleReady = await artifactExists(path.join(experimentDir, "root-oracles", "root-oracle-report.json"));
  const finalEvaluationReady = await artifactExists(path.join(experimentDir, "root-router-report.json"));
  let localRoutersPresent = 0;
  for (const author of manifest.authors ?? []) {
    for (const view of ["semantic", "structural", "full"]) {
      localRoutersPresent += Number(await artifactExists(path.join(experimentDir, "routers", `${author}-router-${view}`, "router.nsrlrt")));
    }
  }
  let rootRoutersPresent = 0;
  for (const view of ["semantic", "structural", "full"]) {
    rootRoutersPresent += Number(await artifactExists(path.join(experimentDir, "routers", `root-router-${view}`, "router.nsrlrt")));
  }
  const blockers = [];
  if (leafCheckpointsPresent < 9) {
    blockers.push(`${9 - leafCheckpointsPresent} leaf checkpoints remain to be trained`);
  }
  if (!localOracleReady) blockers.push("local router oracle labels require per-sample child scoring");
  if (localRoutersPresent < 9) blockers.push(`${9 - localRoutersPresent} local neural router artifacts remain to be trained`);
  if (!rootOracleReady) blockers.push("root oracle labels require child-pod scoring");
  if (rootRoutersPresent < 3) blockers.push(`${3 - rootRoutersPresent} root neural router artifacts remain to be trained`);
  if (!finalEvaluationReady) blockers.push("final recursive routed evaluation has not run");
  const fullExperimentReady = failed.length === 0 && leafCheckpointsPresent === 9 && blockers.length === 0;
  const report = {
    schema: "nsrl.recursive_literary_swarm_preflight.v1",
    manifest: path.resolve(manifestPath),
    preparation_ready: failed.length === 0,
    full_experiment_ready: fullExperimentReady,
    leaf_checkpoints_present: leafCheckpointsPresent,
    leaf_checkpoints_required: 9,
    local_oracle_ready: localOracleReady,
    local_routers_present: localRoutersPresent,
    local_routers_required: 9,
    root_oracle_ready: rootOracleReady,
    root_routers_present: rootRoutersPresent,
    root_routers_required: 3,
    final_evaluation_ready: finalEvaluationReady,
    checks_passed: checks.length - failed.length,
    checks_failed: failed.length,
    failed_checks: failed,
    blockers,
  };
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (outPath) await writeFile(outPath, output);
  else process.stdout.write(output);
  if (failed.length > 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error(`check-recursive-literary-swarm-experiment: ${error.message}`);
  process.exit(1);
});
