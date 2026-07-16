#!/usr/bin/env node

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  auditExperiment,
  decideExperiment,
  freezeExperiment,
  harnessStatus,
  importCompletedExperiment,
  importCompletedRun,
  initHarness,
  nextActions,
  registerExperiment,
  renderHarnessStatus,
  reviewExperiment,
  runExperiment,
  verifyLedger,
} from "./lib/research-harness-v1.mjs";

const defaultRepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function usage() {
  console.log([
    "Usage: node scripts/research-harness.mjs COMMAND [ARGUMENTS] [options]",
    "",
    "Commands:",
    "  init",
    "  register SPEC --actor ID --role scout|theorist|human",
    "  review ID --approve|--reject --actor ID --role statistician|human [--note TEXT]",
    "  freeze ID --actor ID --role protocol|human",
    "  run ID --actor ID --role runner|human [--allow-reserved] [--allow-paid]",
    "  import-run ID --actor ID --role runner|human",
    "  audit ID --actor ID --role auditor|human",
    "  decide ID --actor ID --role curator|human",
    "  import-golden",
    "  next --actor ID --role ROLE",
    "  verify",
    "  status [--json]",
    "",
    "Global options:",
    "  --repo-root PATH       repository root (default: current NSRL repository)",
    "  --state-root PATH      runtime ledger root (default: data/research-harness)",
    "  --policy-root PATH     tracked policy root (default: research/harness)",
  ].join("\n"));
}

function parse(argv) {
  if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
    usage();
    process.exit(argv.length === 0 ? 1 : 0);
  }
  const command = argv[0];
  const positionals = [];
  const options = {
    repoRoot: defaultRepoRoot,
    stateRoot: "data/research-harness",
    policyRoot: "research/harness",
    actor: "",
    role: "",
    note: "",
    approve: null,
    allowReserved: false,
    allowPaid: false,
    json: false,
  };
  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--repo-root") options.repoRoot = requireValue(argv, ++index, arg);
    else if (arg === "--state-root") options.stateRoot = requireValue(argv, ++index, arg);
    else if (arg === "--policy-root") options.policyRoot = requireValue(argv, ++index, arg);
    else if (arg === "--actor") options.actor = requireValue(argv, ++index, arg);
    else if (arg === "--role") options.role = requireValue(argv, ++index, arg);
    else if (arg === "--note") options.note = requireValue(argv, ++index, arg);
    else if (arg === "--approve") options.approve = true;
    else if (arg === "--reject") options.approve = false;
    else if (arg === "--allow-reserved") options.allowReserved = true;
    else if (arg === "--allow-paid") options.allowPaid = true;
    else if (arg === "--json") options.json = true;
    else if (arg.startsWith("--")) throw new Error(`unknown option ${arg}`);
    else positionals.push(arg);
  }
  options.repoRoot = path.resolve(options.repoRoot);
  options.stateRoot = path.resolve(options.repoRoot, options.stateRoot);
  options.policyRoot = path.resolve(options.repoRoot, options.policyRoot);
  return { command, positionals, options };
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) throw new Error(`${flag} requires a value`);
  return argv[index];
}

function requirePositional(positionals, index, label) {
  if (!positionals[index]) throw new Error(`${label} is required`);
  return positionals[index];
}

function actorFrom(options) {
  if (!options.actor || !options.role) throw new Error("--actor and --role are required");
  return { id: options.actor, role: options.role };
}

function print(value) {
  console.log(JSON.stringify(value, null, 2));
}

async function main() {
  const { command, positionals, options } = parse(process.argv.slice(2));
  const common = {
    repoRoot: options.repoRoot,
    policyRoot: options.policyRoot,
    stateRoot: options.stateRoot,
  };
  if (command === "init") {
    const paths = await initHarness(options.stateRoot);
    print({ schema: "nsrl.research_harness_init.v1", ok: true, state_root: paths.root });
  } else if (command === "register") {
    const specPath = requirePositional(positionals, 0, "SPEC");
    print(await registerExperiment({
      repoRoot: options.repoRoot,
      stateRoot: options.stateRoot,
      specPath,
      actor: actorFrom(options),
    }));
  } else if (command === "review") {
    const experimentId = requirePositional(positionals, 0, "ID");
    if (options.approve === null) throw new Error("review requires --approve or --reject");
    const event = await reviewExperiment({
      stateRoot: options.stateRoot,
      experimentId,
      actor: actorFrom(options),
      approved: options.approve,
      note: options.note,
    });
    print(event);
  } else if (command === "freeze") {
    const experimentId = requirePositional(positionals, 0, "ID");
    const contract = await freezeExperiment({ ...common, experimentId, actor: actorFrom(options) });
    print({
      schema: contract.schema,
      experiment_id: experimentId,
      contract_sha256: contract.contract_sha256,
      binding_manifest_sha256: contract.binding_manifest.manifest_sha256,
    });
  } else if (command === "run") {
    const experimentId = requirePositional(positionals, 0, "ID");
    print(await runExperiment({
      ...common,
      experimentId,
      actor: actorFrom(options),
      allowReservedEvidence: options.allowReserved,
      allowPaidCompute: options.allowPaid,
    }));
  } else if (command === "import-run") {
    const experimentId = requirePositional(positionals, 0, "ID");
    print(await importCompletedRun({
      ...common,
      experimentId,
      actor: actorFrom(options),
    }));
  } else if (command === "audit") {
    const experimentId = requirePositional(positionals, 0, "ID");
    print(await auditExperiment({ ...common, experimentId, actor: actorFrom(options) }));
  } else if (command === "decide") {
    const experimentId = requirePositional(positionals, 0, "ID");
    print(await decideExperiment({
      repoRoot: options.repoRoot,
      stateRoot: options.stateRoot,
      experimentId,
      actor: actorFrom(options),
    }));
  } else if (command === "import-golden") {
    print(await importCompletedExperiment({
      ...common,
      specPath: "research/harness/templates/p10m-boolean-jet-confirmation-v1.experiment.json",
      actors: {
        proposer: { id: "legacy:boolean-jet-proposer", role: "theorist" },
        reviewer: { id: "legacy:boolean-jet-statistician", role: "statistician" },
        protocol: { id: "legacy:boolean-jet-protocol", role: "protocol" },
        runner: { id: "legacy:boolean-jet-runner", role: "runner" },
        auditor: { id: "legacy:boolean-jet-auditor", role: "auditor" },
        curator: { id: "legacy:boolean-jet-curator", role: "curator" },
      },
    }));
  } else if (command === "next") {
    print(await nextActions(options.stateRoot, actorFrom(options)));
  } else if (command === "verify") {
    const verification = await verifyLedger(options.stateRoot);
    print({
      schema: verification.schema,
      ok: verification.ok,
      event_count: verification.event_count,
      experiment_count: verification.experiment_count,
      head_event_hash: verification.head_event_hash,
    });
  } else if (command === "status") {
    const status = await harnessStatus(options.stateRoot);
    if (options.json) print(status);
    else process.stdout.write(renderHarnessStatus(status));
  } else {
    throw new Error(`unknown command ${command}`);
  }
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exit(1);
});
