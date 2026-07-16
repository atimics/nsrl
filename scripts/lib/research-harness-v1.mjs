import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  appendFile,
  mkdir,
  open,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

export const EXPERIMENT_SCHEMA = "nsrl.research_experiment.v1";
export const CONTRACT_SCHEMA = "nsrl.research_contract.v1";
export const EVENT_SCHEMA = "nsrl.research_event.v1";
export const RUN_RECEIPT_SCHEMA = "nsrl.research_run_receipt.v1";
export const AUDIT_SCHEMA = "nsrl.research_audit.v1";
export const DECISION_SCHEMA = "nsrl.research_decision.v1";

const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const ACTOR_ID_PATTERN = /^[a-z0-9][a-z0-9:_-]{2,127}$/;
const CONDITION_PATH_PATTERN = /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$/;
const ROLES = new Set([
  "scout",
  "theorist",
  "statistician",
  "protocol",
  "runner",
  "auditor",
  "curator",
  "human",
]);
const OUTCOMES = new Set(["supported", "falsified", "inconclusive"]);
const TERMINAL_STATES = new Set([
  "supported",
  "falsified",
  "inconclusive",
  "invalid",
  "rejected",
]);

function fail(message) {
  throw new Error(`research harness v1: ${message}`);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) fail(`${label} must be an array`);
  return value;
}

function string(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function boolean(value, label) {
  if (typeof value !== "boolean") fail(`${label} must be a boolean`);
  return value;
}

function integer(value, label, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    fail(`${label} must be an integer >= ${minimum}`);
  }
  return value;
}

function uniqueStrings(items, label) {
  const seen = new Set();
  for (const [index, value] of items.entries()) {
    string(value, `${label}[${index}]`);
    if (seen.has(value)) fail(`${label} contains duplicate ${value}`);
    seen.add(value);
  }
  return items;
}

function validateSlug(value, label) {
  string(value, label);
  if (!SLUG_PATTERN.test(value)) fail(`${label} must be a lowercase kebab-case slug`);
  return value;
}

function validateRelativePath(value, label) {
  string(value, label);
  if (path.isAbsolute(value) || value.includes("\0")) {
    fail(`${label} must be a safe repository-relative path`);
  }
  const normalized = path.normalize(value);
  if (normalized === ".." || normalized.startsWith(`..${path.sep}`)) {
    fail(`${label} must not escape the repository`);
  }
  return value;
}

function assertOnlyKeys(value, allowed, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) fail(`${label} contains unknown field ${key}`);
  }
}

export function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
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

export async function sha256File(filePath) {
  const bytes = await readFile(filePath);
  return createHash("sha256").update(bytes).digest("hex");
}

export function validateActor(actor, label = "actor") {
  object(actor, label);
  assertOnlyKeys(actor, new Set(["id", "role"]), label);
  string(actor.id, `${label}.id`);
  if (!ACTOR_ID_PATTERN.test(actor.id)) {
    fail(`${label}.id must use lowercase letters, digits, colon, underscore, or hyphen`);
  }
  string(actor.role, `${label}.role`);
  if (!ROLES.has(actor.role)) fail(`${label}.role is not recognized`);
  return actor;
}

function validateCondition(condition, label) {
  object(condition, label);
  assertOnlyKeys(condition, new Set(["path", "operator", "value"]), label);
  if (!CONDITION_PATH_PATTERN.test(string(condition.path, `${label}.path`))) {
    fail(`${label}.path is not a safe dotted JSON path`);
  }
  if (!["eq", "neq", "lt", "lte", "gt", "gte"].includes(condition.operator)) {
    fail(`${label}.operator is not supported`);
  }
}

export function validateExperimentSpec(spec) {
  object(spec, "experiment");
  assertOnlyKeys(
    spec,
    new Set([
      "schema",
      "id",
      "title",
      "summary",
      "parents",
      "claim",
      "bindings",
      "evidence_access",
      "design",
      "execution",
      "decision",
      "authorization",
    ]),
    "experiment",
  );
  if (spec.schema !== EXPERIMENT_SCHEMA) fail(`schema must be ${EXPERIMENT_SCHEMA}`);
  validateSlug(spec.id, "experiment.id");
  string(spec.title, "experiment.title");
  string(spec.summary, "experiment.summary");
  uniqueStrings(array(spec.parents, "experiment.parents"), "experiment.parents");
  for (const [index, parent] of spec.parents.entries()) {
    validateSlug(parent, `experiment.parents[${index}]`);
  }

  const claim = object(spec.claim, "experiment.claim");
  assertOnlyKeys(claim, new Set(["hypothesis", "estimand", "falsifier", "evidence_level"]), "experiment.claim");
  string(claim.hypothesis, "experiment.claim.hypothesis");
  string(claim.estimand, "experiment.claim.estimand");
  string(claim.falsifier, "experiment.claim.falsifier");
  if (!["diagnostic", "calibration", "confirmation", "replication"].includes(claim.evidence_level)) {
    fail("experiment.claim.evidence_level is not recognized");
  }

  const bindings = object(spec.bindings, "experiment.bindings");
  assertOnlyKeys(bindings, new Set(["files"]), "experiment.bindings");
  const files = array(bindings.files, "experiment.bindings.files");
  if (files.length === 0) fail("experiment.bindings.files must not be empty");
  const bindingPaths = new Set();
  const roles = new Set([
    "source",
    "model",
    "tokenizer",
    "dataset",
    "evaluator",
    "scientific-contract",
    "control",
  ]);
  for (const [index, binding] of files.entries()) {
    object(binding, `experiment.bindings.files[${index}]`);
    assertOnlyKeys(binding, new Set(["path", "role"]), `experiment.bindings.files[${index}]`);
    validateRelativePath(binding.path, `experiment.bindings.files[${index}].path`);
    if (!roles.has(binding.role)) fail(`experiment.bindings.files[${index}].role is invalid`);
    if (bindingPaths.has(binding.path)) fail(`duplicate binding path ${binding.path}`);
    bindingPaths.add(binding.path);
  }

  const access = object(spec.evidence_access, "experiment.evidence_access");
  assertOnlyKeys(
    access,
    new Set(["allowed_partitions", "excluded_partitions", "consumes_reserved_evidence"]),
    "experiment.evidence_access",
  );
  uniqueStrings(array(access.allowed_partitions, "experiment.evidence_access.allowed_partitions"), "experiment.evidence_access.allowed_partitions");
  uniqueStrings(array(access.excluded_partitions, "experiment.evidence_access.excluded_partitions"), "experiment.evidence_access.excluded_partitions");
  for (const [index, partition] of access.allowed_partitions.entries()) {
    validateSlug(partition, `experiment.evidence_access.allowed_partitions[${index}]`);
    if (access.excluded_partitions.includes(partition)) {
      fail(`evidence partition ${partition} is both allowed and excluded`);
    }
  }
  for (const [index, partition] of access.excluded_partitions.entries()) {
    validateSlug(partition, `experiment.evidence_access.excluded_partitions[${index}]`);
  }
  boolean(access.consumes_reserved_evidence, "experiment.evidence_access.consumes_reserved_evidence");

  const design = object(spec.design, "experiment.design");
  assertOnlyKeys(
    design,
    new Set(["independent_unit", "planned_units", "minimum_informative_units", "family_size", "controls"]),
    "experiment.design",
  );
  string(design.independent_unit, "experiment.design.independent_unit");
  integer(design.planned_units, "experiment.design.planned_units", 1);
  integer(design.minimum_informative_units, "experiment.design.minimum_informative_units", 0);
  integer(design.family_size, "experiment.design.family_size", 1);
  uniqueStrings(array(design.controls, "experiment.design.controls"), "experiment.design.controls");
  if (design.minimum_informative_units > design.planned_units) {
    fail("experiment.design.minimum_informative_units exceeds planned_units");
  }

  const execution = object(spec.execution, "experiment.execution");
  assertOnlyKeys(execution, new Set(["runner_template", "expected_outputs", "max_seconds", "max_output_bytes"]), "experiment.execution");
  validateSlug(execution.runner_template, "experiment.execution.runner_template");
  uniqueStrings(array(execution.expected_outputs, "experiment.execution.expected_outputs"), "experiment.execution.expected_outputs");
  if (execution.expected_outputs.length === 0) fail("experiment.execution.expected_outputs must not be empty");
  execution.expected_outputs.forEach((output, index) => validateRelativePath(output, `experiment.execution.expected_outputs[${index}]`));
  integer(execution.max_seconds, "experiment.execution.max_seconds", 1);
  integer(execution.max_output_bytes, "experiment.execution.max_output_bytes", 1024);

  const decision = object(spec.decision, "experiment.decision");
  assertOnlyKeys(decision, new Set(["checker_template", "result_path", "rules", "default_outcome"]), "experiment.decision");
  validateSlug(decision.checker_template, "experiment.decision.checker_template");
  validateRelativePath(decision.result_path, "experiment.decision.result_path");
  const rules = array(decision.rules, "experiment.decision.rules");
  for (const [ruleIndex, rule] of rules.entries()) {
    object(rule, `experiment.decision.rules[${ruleIndex}]`);
    assertOnlyKeys(rule, new Set(["outcome", "all"]), `experiment.decision.rules[${ruleIndex}]`);
    if (!OUTCOMES.has(rule.outcome)) fail(`experiment.decision.rules[${ruleIndex}].outcome is invalid`);
    const conditions = array(rule.all, `experiment.decision.rules[${ruleIndex}].all`);
    if (conditions.length === 0) fail(`experiment.decision.rules[${ruleIndex}].all must not be empty`);
    conditions.forEach((condition, conditionIndex) => validateCondition(condition, `experiment.decision.rules[${ruleIndex}].all[${conditionIndex}]`));
  }
  if (!OUTCOMES.has(decision.default_outcome)) fail("experiment.decision.default_outcome is invalid");
  if (!execution.expected_outputs.includes(decision.result_path)) {
    fail("experiment.decision.result_path must be one of execution.expected_outputs");
  }

  const authorization = object(spec.authorization, "experiment.authorization");
  assertOnlyKeys(
    authorization,
    new Set(["local_execution", "reserved_evidence", "optimizer_change", "paid_compute"]),
    "experiment.authorization",
  );
  for (const key of ["local_execution", "reserved_evidence", "optimizer_change", "paid_compute"]) {
    boolean(authorization[key], `experiment.authorization.${key}`);
  }
  if (access.consumes_reserved_evidence && !authorization.reserved_evidence) {
    fail("reserved evidence consumption is declared but not authorized");
  }
  return spec;
}

function resolveWithin(root, relativePath, label) {
  validateRelativePath(relativePath, label);
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, relativePath);
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) {
    fail(`${label} escapes its root`);
  }
  return resolved;
}

function pathsFor(stateRoot, experimentId = "") {
  const root = path.resolve(stateRoot);
  const experimentRoot = experimentId
    ? resolveWithin(path.join(root, "experiments"), experimentId, "experiment id")
    : "";
  return {
    root,
    events: path.join(root, "events.jsonl"),
    lock: path.join(root, ".events.lock"),
    experiments: path.join(root, "experiments"),
    experimentRoot,
    proposal: experimentRoot ? path.join(experimentRoot, "proposal.json") : "",
    contract: experimentRoot ? path.join(experimentRoot, "contract.json") : "",
    runReceipt: experimentRoot ? path.join(experimentRoot, "run-receipt.json") : "",
    audit: experimentRoot ? path.join(experimentRoot, "audit.json") : "",
    decision: experimentRoot ? path.join(experimentRoot, "decision.json") : "",
  };
}

export async function initHarness(stateRoot) {
  const paths = pathsFor(stateRoot);
  await mkdir(paths.experiments, { recursive: true });
  try {
    await stat(paths.events);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    await writeFile(paths.events, "", { flag: "wx" });
  }
  return paths;
}

async function atomicWrite(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporary, value);
  await rename(temporary, filePath);
}

async function readJson(filePath, label = filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    fail(`cannot read ${label}: ${error.message}`);
  }
}

async function readEventsFile(eventsPath) {
  let text;
  try {
    text = await readFile(eventsPath, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  return text
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        fail(`events.jsonl line ${index + 1} is invalid JSON: ${error.message}`);
      }
    });
}

function eventBody(event) {
  const { event_hash: _eventHash, ...body } = event;
  return body;
}

function reduceEvent(state, event) {
  const current = state || {
    id: event.experiment_id,
    state: "absent",
    actors: {},
    events: [],
    outcome: "",
  };
  const next = { ...current, actors: { ...current.actors }, events: [...current.events, event] };
  const requireRole = (allowed) => {
    if (!allowed.includes(event.actor.role)) {
      fail(`${event.experiment_id} event ${event.event_type} is not allowed for role ${event.actor.role}`);
    }
  };
  switch (event.event_type) {
    case "registered":
      requireRole(["scout", "theorist", "human"]);
      if (current.state !== "absent") fail(`${event.experiment_id} was registered more than once`);
      next.state = "draft";
      next.actors.proposer = event.actor.id;
      next.proposal_sha256 = event.payload.proposal_sha256;
      break;
    case "reviewed":
      requireRole(["statistician", "human"]);
      if (current.state !== "draft") fail(`${event.experiment_id} cannot be reviewed from ${current.state}`);
      if (current.actors.proposer === event.actor.id) fail(`${event.experiment_id} was self-reviewed`);
      next.state = event.payload.approved ? "reviewed" : "rejected";
      next.actors.reviewer = event.actor.id;
      break;
    case "frozen":
      requireRole(["protocol", "human"]);
      if (current.state !== "reviewed") fail(`${event.experiment_id} cannot be frozen from ${current.state}`);
      if ([current.actors.proposer, current.actors.reviewer].includes(event.actor.id)) {
        fail(`${event.experiment_id} freeze violates actor separation`);
      }
      next.state = "frozen";
      next.actors.protocol = event.actor.id;
      next.contract_sha256 = event.payload.contract_sha256;
      break;
    case "run_started":
      requireRole(["runner", "human"]);
      if (!["frozen", "execution-failed"].includes(current.state)) {
        fail(`${event.experiment_id} cannot run from ${current.state}`);
      }
      if ([current.actors.proposer, current.actors.reviewer, current.actors.protocol].includes(event.actor.id)) {
        fail(`${event.experiment_id} run violates actor separation`);
      }
      next.state = "running";
      next.actors.runner = event.actor.id;
      break;
    case "run_completed":
    case "run_imported":
      requireRole(["runner", "human"]);
      if (event.event_type === "run_completed" && current.state !== "running") {
        fail(`${event.experiment_id} cannot complete a run from ${current.state}`);
      }
      if (event.event_type === "run_imported" && current.state !== "frozen") {
        fail(`${event.experiment_id} cannot import a run from ${current.state}`);
      }
      if (event.event_type === "run_imported"
        && [current.actors.proposer, current.actors.reviewer, current.actors.protocol].includes(event.actor.id)) {
        fail(`${event.experiment_id} imported run violates actor separation`);
      }
      if (event.event_type === "run_completed" && current.actors.runner !== event.actor.id) {
        fail(`${event.experiment_id} run completion actor does not match its runner`);
      }
      next.state = "run-complete";
      next.actors.runner = event.actor.id;
      next.run_receipt_sha256 = event.payload.run_receipt_sha256;
      break;
    case "run_failed":
      requireRole(["runner", "human"]);
      if (current.state !== "running") fail(`${event.experiment_id} cannot fail a run from ${current.state}`);
      if (current.actors.runner !== event.actor.id) fail(`${event.experiment_id} run failure actor does not match its runner`);
      next.state = "execution-failed";
      next.actors.runner = event.actor.id;
      break;
    case "audited":
      requireRole(["auditor", "human"]);
      if (current.state !== "run-complete") fail(`${event.experiment_id} cannot be audited from ${current.state}`);
      if ([
        current.actors.proposer,
        current.actors.reviewer,
        current.actors.protocol,
        current.actors.runner,
      ].includes(event.actor.id)) {
        fail(`${event.experiment_id} audit violates actor separation`);
      }
      next.state = event.payload.ok ? "audited" : "invalid";
      next.actors.auditor = event.actor.id;
      next.audit_sha256 = event.payload.audit_sha256;
      break;
    case "decided":
      requireRole(["curator", "human"]);
      if (current.state !== "audited") fail(`${event.experiment_id} cannot be decided from ${current.state}`);
      if (!OUTCOMES.has(event.payload.outcome)) fail(`${event.experiment_id} has an invalid outcome`);
      if ([
        current.actors.proposer,
        current.actors.reviewer,
        current.actors.protocol,
        current.actors.runner,
        current.actors.auditor,
      ].includes(event.actor.id)) {
        fail(`${event.experiment_id} decision violates actor separation`);
      }
      next.state = event.payload.outcome;
      next.outcome = event.payload.outcome;
      next.actors.curator = event.actor.id;
      next.decision_sha256 = event.payload.decision_sha256;
      break;
    case "note":
      requireRole([...ROLES]);
      if (current.state === "absent") fail(`${event.experiment_id} cannot receive a note before registration`);
      break;
    default:
      fail(`unknown event type ${event.event_type}`);
  }
  return next;
}

export function reduceExperiments(events) {
  const states = new Map();
  for (const event of events) {
    states.set(event.experiment_id, reduceEvent(states.get(event.experiment_id), event));
  }
  return states;
}

export async function verifyLedger(stateRoot) {
  const paths = await initHarness(stateRoot);
  const events = await readEventsFile(paths.events);
  let previous = "";
  for (const [index, event] of events.entries()) {
    object(event, `event ${index + 1}`);
    if (event.schema !== EVENT_SCHEMA) fail(`event ${index + 1} has the wrong schema`);
    if (event.sequence !== index + 1) fail(`event ${index + 1} has a noncanonical sequence`);
    validateSlug(event.experiment_id, `event ${index + 1}.experiment_id`);
    string(event.event_type, `event ${index + 1}.event_type`);
    validateActor(event.actor, `event ${index + 1}.actor`);
    if (Number.isNaN(Date.parse(string(event.occurred_at, `event ${index + 1}.occurred_at`)))) {
      fail(`event ${index + 1}.occurred_at is not an ISO date-time`);
    }
    if (event.previous_event_hash !== previous) fail(`event ${index + 1} breaks the hash chain`);
    const expectedHash = sha256Canonical(eventBody(event));
    if (event.event_hash !== expectedHash) fail(`event ${index + 1} has an invalid hash`);
    previous = event.event_hash;
  }
  const states = reduceExperiments(events);
  return {
    schema: "nsrl.research_ledger_check.v1",
    ok: true,
    event_count: events.length,
    experiment_count: states.size,
    head_event_hash: previous,
    events,
    states,
  };
}

async function acquireLock(lockPath, timeoutMs = 5_000) {
  const started = Date.now();
  while (true) {
    try {
      return await open(lockPath, "wx");
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      if (Date.now() - started >= timeoutMs) fail("timed out waiting for the event ledger lock");
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  }
}

async function transition(
  stateRoot,
  experimentId,
  eventType,
  actor,
  payload,
  guard,
  beforeAppend,
) {
  validateSlug(experimentId, "experiment id");
  validateActor(actor);
  const paths = await initHarness(stateRoot);
  const lock = await acquireLock(paths.lock);
  try {
    const verification = await verifyLedger(stateRoot);
    const current = verification.states.get(experimentId) || {
      id: experimentId,
      state: "absent",
      actors: {},
      events: [],
    };
    if (guard) guard(current);
    if (beforeAppend) await beforeAppend(current);
    const body = {
      schema: EVENT_SCHEMA,
      sequence: verification.events.length + 1,
      experiment_id: experimentId,
      event_type: eventType,
      actor,
      occurred_at: new Date().toISOString(),
      previous_event_hash: verification.head_event_hash,
      payload,
    };
    const event = { ...body, event_hash: sha256Canonical(body) };
    reduceEvent(current, event);
    await appendFile(paths.events, `${canonicalJson(event)}\n`);
    return event;
  } finally {
    await lock.close();
    await unlink(paths.lock).catch(() => {});
  }
}

async function loadTemplatePolicy(policyRoot, expectedSha256 = "") {
  const policyPath = path.join(path.resolve(policyRoot), "runner-templates.json");
  let policyText;
  try {
    policyText = await readFile(policyPath, "utf8");
  } catch (error) {
    fail(`cannot read runner template policy: ${error.message}`);
  }
  const policySha256 = createHash("sha256").update(policyText).digest("hex");
  if (expectedSha256 && policySha256 !== expectedSha256) {
    fail("runner template policy changed after contract freeze; create a new contract version");
  }
  let policy;
  try {
    policy = JSON.parse(policyText);
  } catch (error) {
    fail(`runner template policy is invalid JSON: ${error.message}`);
  }
  if (policy.schema !== "nsrl.research_runner_templates.v1") fail("runner template policy has the wrong schema");
  object(policy.templates, "runner template policy.templates");
  for (const [id, template] of Object.entries(policy.templates)) {
    validateSlug(id, `runner template ${id}`);
    object(template, `runner template ${id}`);
    const command = array(template.command, `runner template ${id}.command`);
    const checker = array(template.checker, `runner template ${id}.checker`);
    if (command.length === 0 || checker.length === 0) fail(`runner template ${id} requires command and checker`);
    command.forEach((part, index) => string(part, `runner template ${id}.command[${index}]`));
    checker.forEach((part, index) => string(part, `runner template ${id}.checker[${index}]`));
    uniqueStrings(array(template.allowed_partitions, `runner template ${id}.allowed_partitions`), `runner template ${id}.allowed_partitions`);
    uniqueStrings(array(template.expected_outputs, `runner template ${id}.expected_outputs`), `runner template ${id}.expected_outputs`);
    boolean(template.paid_compute, `runner template ${id}.paid_compute`);
  }
  return { templates: policy.templates, policySha256 };
}

function validatePolicyForSpec(spec, templates) {
  const runner = templates[spec.execution.runner_template];
  if (!runner) fail(`runner template ${spec.execution.runner_template} is not allowlisted`);
  const checker = templates[spec.decision.checker_template];
  if (!checker) fail(`checker template ${spec.decision.checker_template} is not allowlisted`);
  for (const partition of spec.evidence_access.allowed_partitions) {
    if (!runner.allowed_partitions.includes(partition)) {
      fail(`runner template does not authorize evidence partition ${partition}`);
    }
  }
  for (const output of spec.execution.expected_outputs) {
    if (!runner.expected_outputs.includes(output)) {
      fail(`runner template does not authorize output ${output}`);
    }
  }
  if (runner.paid_compute && !spec.authorization.paid_compute) {
    fail("runner requires paid compute but the experiment does not authorize it");
  }
  return { runner, checker };
}

async function buildBindingManifest(repoRoot, spec) {
  const entries = await Promise.all(spec.bindings.files.map(async (binding) => {
    const filePath = resolveWithin(repoRoot, binding.path, `binding ${binding.path}`);
    let info;
    try {
      info = await stat(filePath);
    } catch (error) {
      fail(`binding ${binding.path} cannot be read: ${error.message}`);
    }
    if (!info.isFile()) fail(`binding ${binding.path} is not a file`);
    return {
      path: binding.path,
      role: binding.role,
      bytes: info.size,
      sha256: await sha256File(filePath),
    };
  }));
  entries.sort((left, right) => left.path.localeCompare(right.path));
  return {
    schema: "nsrl.research_binding_manifest.v1",
    files: entries,
    manifest_sha256: sha256Canonical(entries),
  };
}

async function verifyContract(repoRoot, contractPath, expectedHash = "") {
  const contract = await readJson(contractPath, "frozen research contract");
  if (contract.schema !== CONTRACT_SCHEMA) fail("frozen research contract has the wrong schema");
  const { contract_sha256: storedHash, ...body } = contract;
  const actualHash = sha256Canonical(body);
  if (storedHash !== actualHash) fail("frozen research contract hash is invalid");
  if (expectedHash && storedHash !== expectedHash) fail("frozen research contract does not match the ledger");
  validateExperimentSpec(contract.experiment);
  const currentManifest = await buildBindingManifest(repoRoot, contract.experiment);
  if (currentManifest.manifest_sha256 !== contract.binding_manifest.manifest_sha256) {
    fail("one or more frozen input bindings changed after contract freeze");
  }
  return contract;
}

export async function registerExperiment({ repoRoot, stateRoot, specPath, actor }) {
  if (!["scout", "theorist", "human"].includes(validateActor(actor).role)) {
    fail("only a scout, theorist, or human may register an experiment");
  }
  const spec = validateExperimentSpec(await readJson(path.resolve(repoRoot, specPath), "experiment proposal"));
  const paths = pathsFor(stateRoot, spec.id);
  await initHarness(stateRoot);
  await mkdir(paths.experimentRoot, { recursive: true });
  const proposalText = `${JSON.stringify(spec, null, 2)}\n`;
  const proposalSha256 = createHash("sha256").update(proposalText).digest("hex");
  await transition(stateRoot, spec.id, "registered", actor, {
    proposal_sha256: proposalSha256,
    source_path: specPath,
  }, (current) => {
    if (current.state !== "absent") fail(`${spec.id} is already registered`);
  }, async () => {
    await atomicWrite(paths.proposal, proposalText);
  });
  return { id: spec.id, state: "draft", proposal_sha256: proposalSha256 };
}

export async function reviewExperiment({ stateRoot, experimentId, actor, approved, note = "" }) {
  if (!["statistician", "human"].includes(validateActor(actor).role)) {
    fail("only a statistician or human may review an experiment");
  }
  boolean(approved, "approved");
  return transition(stateRoot, experimentId, "reviewed", actor, { approved, note }, (current) => {
    if (current.state !== "draft") fail(`${experimentId} is not awaiting review`);
    if (current.actors.proposer === actor.id) fail("the proposer cannot review their own experiment");
  });
}

export async function freezeExperiment({ repoRoot, policyRoot, stateRoot, experimentId, actor }) {
  if (!["protocol", "human"].includes(validateActor(actor).role)) {
    fail("only a protocol agent or human may freeze an experiment");
  }
  const paths = pathsFor(stateRoot, experimentId);
  const spec = validateExperimentSpec(await readJson(paths.proposal, "registered proposal"));
  const proposalText = await readFile(paths.proposal, "utf8");
  const proposalSha256 = createHash("sha256").update(proposalText).digest("hex");
  const policy = await loadTemplatePolicy(policyRoot);
  validatePolicyForSpec(spec, policy.templates);
  const bindingManifest = await buildBindingManifest(repoRoot, spec);
  const body = {
    schema: CONTRACT_SCHEMA,
    experiment: spec,
    frozen: {
      frozen_at: new Date().toISOString(),
      proposal_sha256: proposalSha256,
      policy_sha256: policy.policySha256,
    },
    binding_manifest: bindingManifest,
  };
  const contract = { ...body, contract_sha256: sha256Canonical(body) };
  await transition(stateRoot, experimentId, "frozen", actor, {
    contract_sha256: contract.contract_sha256,
    binding_manifest_sha256: bindingManifest.manifest_sha256,
  }, (current) => {
    if (current.state !== "reviewed") fail(`${experimentId} is not ready to freeze`);
    if ([current.actors.proposer, current.actors.reviewer].includes(actor.id)) {
      fail("the proposer or reviewer cannot freeze the experiment");
    }
    if (current.proposal_sha256 !== proposalSha256) fail("proposal changed after registration");
  }, async () => {
    await atomicWrite(paths.contract, `${JSON.stringify(contract, null, 2)}\n`);
  });
  return contract;
}

async function hashOutputs(repoRoot, outputs) {
  return Promise.all(outputs.map(async (output) => {
    const filePath = resolveWithin(repoRoot, output, `output ${output}`);
    const info = await stat(filePath).catch((error) => fail(`expected output ${output} is missing: ${error.message}`));
    if (!info.isFile()) fail(`expected output ${output} is not a file`);
    return { path: output, bytes: info.size, sha256: await sha256File(filePath) };
  }));
}

function executeTemplate(parts, repoRoot, spec) {
  const [command, ...args] = parts;
  const started = Date.now();
  const result = spawnSync(command, args, {
    cwd: path.resolve(repoRoot),
    encoding: "utf8",
    timeout: spec.execution.max_seconds * 1000,
    maxBuffer: spec.execution.max_output_bytes,
    env: { ...process.env, NSRL_RESEARCH_EXPERIMENT_ID: spec.id },
  });
  return {
    command: parts,
    exit_code: result.status,
    signal: result.signal || "",
    duration_ms: Date.now() - started,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    error: result.error ? String(result.error.message || result.error) : "",
  };
}

export async function runExperiment({
  repoRoot,
  policyRoot,
  stateRoot,
  experimentId,
  actor,
  allowReservedEvidence = false,
  allowPaidCompute = false,
}) {
  if (!["runner", "human"].includes(validateActor(actor).role)) {
    fail("only a runner or human may execute an experiment");
  }
  const verification = await verifyLedger(stateRoot);
  const current = verification.states.get(experimentId);
  if (!current || !["frozen", "execution-failed"].includes(current.state)) {
    fail(`${experimentId} is not runnable from its current state`);
  }
  const paths = pathsFor(stateRoot, experimentId);
  const contract = await verifyContract(repoRoot, paths.contract, current.contract_sha256);
  const policy = await loadTemplatePolicy(policyRoot, contract.frozen.policy_sha256);
  const spec = contract.experiment;
  if (!spec.authorization.local_execution) fail("local execution is not authorized by the frozen contract");
  if (spec.evidence_access.consumes_reserved_evidence && !allowReservedEvidence) {
    fail("reserved evidence execution requires an explicit runtime authorization");
  }
  if (spec.authorization.paid_compute && !allowPaidCompute) {
    fail("paid compute execution requires an explicit runtime authorization");
  }
  const { runner } = validatePolicyForSpec(spec, policy.templates);
  await transition(stateRoot, experimentId, "run_started", actor, {
    runner_template: spec.execution.runner_template,
    contract_sha256: contract.contract_sha256,
  }, (latest) => {
    if (!["frozen", "execution-failed"].includes(latest.state)) fail(`${experimentId} is not runnable`);
  });
  const executed = executeTemplate(runner.command, repoRoot, spec);
  await atomicWrite(path.join(paths.experimentRoot, "run.stdout.txt"), executed.stdout);
  await atomicWrite(path.join(paths.experimentRoot, "run.stderr.txt"), executed.stderr);
  if (executed.exit_code !== 0 || executed.error) {
    await transition(stateRoot, experimentId, "run_failed", actor, {
      exit_code: executed.exit_code,
      signal: executed.signal,
      duration_ms: executed.duration_ms,
      error: executed.error,
    });
    fail(`experiment runner failed with exit code ${executed.exit_code}: ${executed.error || executed.stderr.trim()}`);
  }
  let outputs;
  try {
    outputs = await hashOutputs(repoRoot, spec.execution.expected_outputs);
  } catch (error) {
    await transition(stateRoot, experimentId, "run_failed", actor, {
      exit_code: executed.exit_code,
      signal: executed.signal,
      duration_ms: executed.duration_ms,
      error: error.message || String(error),
    });
    throw error;
  }
  const receiptBody = {
    schema: RUN_RECEIPT_SCHEMA,
    experiment_id: experimentId,
    contract_sha256: contract.contract_sha256,
    imported: false,
    runner_template: spec.execution.runner_template,
    command: executed.command,
    exit_code: executed.exit_code,
    duration_ms: executed.duration_ms,
    outputs,
  };
  const receipt = { ...receiptBody, receipt_sha256: sha256Canonical(receiptBody) };
  await atomicWrite(paths.runReceipt, `${JSON.stringify(receipt, null, 2)}\n`);
  await transition(stateRoot, experimentId, "run_completed", actor, {
    run_receipt_sha256: receipt.receipt_sha256,
    output_count: outputs.length,
  });
  return receipt;
}

export async function importCompletedRun({ repoRoot, policyRoot, stateRoot, experimentId, actor }) {
  if (!["runner", "human"].includes(validateActor(actor).role)) {
    fail("only a runner or human may import a completed run");
  }
  const verification = await verifyLedger(stateRoot);
  const current = verification.states.get(experimentId);
  if (!current || current.state !== "frozen") fail(`${experimentId} is not ready for a completed-run import`);
  const paths = pathsFor(stateRoot, experimentId);
  const contract = await verifyContract(repoRoot, paths.contract, current.contract_sha256);
  await loadTemplatePolicy(policyRoot, contract.frozen.policy_sha256);
  const outputs = await hashOutputs(repoRoot, contract.experiment.execution.expected_outputs);
  const receiptBody = {
    schema: RUN_RECEIPT_SCHEMA,
    experiment_id: experimentId,
    contract_sha256: contract.contract_sha256,
    imported: true,
    runner_template: contract.experiment.execution.runner_template,
    command: [],
    exit_code: 0,
    duration_ms: 0,
    outputs,
  };
  const receipt = { ...receiptBody, receipt_sha256: sha256Canonical(receiptBody) };
  await atomicWrite(paths.runReceipt, `${JSON.stringify(receipt, null, 2)}\n`);
  await transition(stateRoot, experimentId, "run_imported", actor, {
    run_receipt_sha256: receipt.receipt_sha256,
    output_count: outputs.length,
  });
  return receipt;
}

export async function auditExperiment({ repoRoot, policyRoot, stateRoot, experimentId, actor }) {
  if (!["auditor", "human"].includes(validateActor(actor).role)) {
    fail("only an auditor or human may audit an experiment");
  }
  const verification = await verifyLedger(stateRoot);
  const current = verification.states.get(experimentId);
  if (!current || current.state !== "run-complete") fail(`${experimentId} is not ready for audit`);
  if ([
    current.actors.proposer,
    current.actors.reviewer,
    current.actors.protocol,
    current.actors.runner,
  ].includes(actor.id)) {
    fail("the proposer, reviewer, protocol agent, or runner cannot audit the experiment");
  }
  const paths = pathsFor(stateRoot, experimentId);
  const contract = await verifyContract(repoRoot, paths.contract, current.contract_sha256);
  const policy = await loadTemplatePolicy(policyRoot, contract.frozen.policy_sha256);
  const { checker } = validatePolicyForSpec(contract.experiment, policy.templates);
  const result = executeTemplate(checker.checker, repoRoot, contract.experiment);
  await atomicWrite(path.join(paths.experimentRoot, "audit.stdout.txt"), result.stdout);
  await atomicWrite(path.join(paths.experimentRoot, "audit.stderr.txt"), result.stderr);
  const resultSha256 = await sha256File(resolveWithin(repoRoot, contract.experiment.decision.result_path, "decision result"));
  const auditBody = {
    schema: AUDIT_SCHEMA,
    experiment_id: experimentId,
    contract_sha256: contract.contract_sha256,
    checker_template: contract.experiment.decision.checker_template,
    checker_command: result.command,
    checker_exit_code: result.exit_code,
    checker_signal: result.signal,
    checker_error: result.error,
    duration_ms: result.duration_ms,
    result_sha256: resultSha256,
    ok: result.exit_code === 0 && !result.error,
  };
  const audit = { ...auditBody, audit_sha256: sha256Canonical(auditBody) };
  await atomicWrite(paths.audit, `${JSON.stringify(audit, null, 2)}\n`);
  await transition(stateRoot, experimentId, "audited", actor, {
    audit_sha256: audit.audit_sha256,
    result_sha256: resultSha256,
    ok: audit.ok,
  });
  return audit;
}

function valueAtPath(value, dottedPath) {
  return dottedPath.split(".").reduce((current, key) => {
    if (!current || typeof current !== "object" || !(key in current)) {
      fail(`decision path ${dottedPath} does not exist in the result`);
    }
    return current[key];
  }, value);
}

function conditionMatches(result, condition) {
  const actual = valueAtPath(result, condition.path);
  if (["lt", "lte", "gt", "gte"].includes(condition.operator)) {
    if (typeof actual !== "number" || !Number.isFinite(actual)
      || typeof condition.value !== "number" || !Number.isFinite(condition.value)) {
      fail(`decision condition ${condition.path} requires finite numeric operands`);
    }
  }
  switch (condition.operator) {
    case "eq": return canonicalJson(actual) === canonicalJson(condition.value);
    case "neq": return canonicalJson(actual) !== canonicalJson(condition.value);
    case "lt": return actual < condition.value;
    case "lte": return actual <= condition.value;
    case "gt": return actual > condition.value;
    case "gte": return actual >= condition.value;
    default: fail(`unsupported decision operator ${condition.operator}`);
  }
}

function evaluateDecision(result, decision) {
  for (const rule of decision.rules) {
    if (rule.all.every((condition) => conditionMatches(result, condition))) return rule.outcome;
  }
  return decision.default_outcome;
}

export async function decideExperiment({ repoRoot, stateRoot, experimentId, actor }) {
  if (!["curator", "human"].includes(validateActor(actor).role)) {
    fail("only a curator or human may record a scientific decision");
  }
  const verification = await verifyLedger(stateRoot);
  const current = verification.states.get(experimentId);
  if (!current || current.state !== "audited") fail(`${experimentId} is not ready for a decision`);
  if ([
    current.actors.proposer,
    current.actors.reviewer,
    current.actors.protocol,
    current.actors.runner,
    current.actors.auditor,
  ].includes(actor.id)) {
    fail("a prior lifecycle actor cannot curate the final decision");
  }
  const paths = pathsFor(stateRoot, experimentId);
  const contract = await verifyContract(repoRoot, paths.contract, current.contract_sha256);
  const audit = await readJson(paths.audit, "audit receipt");
  if (audit.schema !== AUDIT_SCHEMA || !audit.ok || audit.audit_sha256 !== current.audit_sha256) {
    fail("a passing, ledger-bound audit is required before decision");
  }
  const resultPath = resolveWithin(repoRoot, contract.experiment.decision.result_path, "decision result");
  const result = await readJson(resultPath, "decision result");
  const resultSha256 = await sha256File(resultPath);
  if (resultSha256 !== audit.result_sha256) fail("decision result changed after audit");
  const outcome = evaluateDecision(result, contract.experiment.decision);
  const decisionBody = {
    schema: DECISION_SCHEMA,
    experiment_id: experimentId,
    contract_sha256: contract.contract_sha256,
    audit_sha256: audit.audit_sha256,
    result_sha256: resultSha256,
    outcome,
    claim: contract.experiment.claim,
    authorization: contract.experiment.authorization,
  };
  const decision = { ...decisionBody, decision_sha256: sha256Canonical(decisionBody) };
  await atomicWrite(paths.decision, `${JSON.stringify(decision, null, 2)}\n`);
  await transition(stateRoot, experimentId, "decided", actor, {
    decision_sha256: decision.decision_sha256,
    outcome,
  });
  return decision;
}

export async function importCompletedExperiment({ repoRoot, policyRoot, stateRoot, specPath, actors }) {
  const registered = await registerExperiment({ repoRoot, stateRoot, specPath, actor: actors.proposer });
  await reviewExperiment({
    stateRoot,
    experimentId: registered.id,
    actor: actors.reviewer,
    approved: true,
    note: "Imported completed experiment reviewed against its original frozen protocol.",
  });
  await freezeExperiment({ repoRoot, policyRoot, stateRoot, experimentId: registered.id, actor: actors.protocol });
  await importCompletedRun({ repoRoot, policyRoot, stateRoot, experimentId: registered.id, actor: actors.runner });
  const audit = await auditExperiment({ repoRoot, policyRoot, stateRoot, experimentId: registered.id, actor: actors.auditor });
  if (!audit.ok) fail(`imported experiment ${registered.id} failed its independent checker`);
  return decideExperiment({ repoRoot, stateRoot, experimentId: registered.id, actor: actors.curator });
}

export async function harnessStatus(stateRoot) {
  const verification = await verifyLedger(stateRoot);
  const experiments = [...verification.states.values()]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((state) => ({
      id: state.id,
      state: state.state,
      outcome: state.outcome || "",
      terminal: TERMINAL_STATES.has(state.state),
      event_count: state.events.length,
      actors: state.actors,
      contract_sha256: state.contract_sha256 || "",
      decision_sha256: state.decision_sha256 || "",
    }));
  return {
    schema: "nsrl.research_harness_status.v1",
    ok: true,
    event_count: verification.event_count,
    experiment_count: experiments.length,
    head_event_hash: verification.head_event_hash,
    experiments,
  };
}

export async function nextActions(stateRoot, actor) {
  validateActor(actor);
  const verification = await verifyLedger(stateRoot);
  const actions = [];
  for (const state of verification.states.values()) {
    let action = "";
    if (["statistician", "human"].includes(actor.role)
      && state.state === "draft"
      && state.actors.proposer !== actor.id) {
      action = "review";
    } else if (["protocol", "human"].includes(actor.role)
      && state.state === "reviewed"
      && ![state.actors.proposer, state.actors.reviewer].includes(actor.id)) {
      action = "freeze";
    } else if (["runner", "human"].includes(actor.role)
      && ["frozen", "execution-failed"].includes(state.state)
      && ![state.actors.proposer, state.actors.reviewer, state.actors.protocol].includes(actor.id)) {
      action = "run";
    } else if (["auditor", "human"].includes(actor.role)
      && state.state === "run-complete"
      && ![
        state.actors.proposer,
        state.actors.reviewer,
        state.actors.protocol,
        state.actors.runner,
      ].includes(actor.id)) {
      action = "audit";
    } else if (["curator", "human"].includes(actor.role)
      && state.state === "audited"
      && ![
        state.actors.proposer,
        state.actors.reviewer,
        state.actors.protocol,
        state.actors.runner,
        state.actors.auditor,
      ].includes(actor.id)) {
      action = "decide";
    }
    if (action) {
      const base = `node scripts/research-harness.mjs ${action} ${state.id}`;
      const identity = `--actor ${actor.id} --role ${actor.role}`;
      const commands = action === "review"
        ? [`${base} --approve ${identity}`, `${base} --reject ${identity}`]
        : [`${base} ${identity}`];
      actions.push({
        experiment_id: state.id,
        state: state.state,
        action,
        commands,
      });
    }
  }
  actions.sort((left, right) => left.experiment_id.localeCompare(right.experiment_id));
  return {
    schema: "nsrl.research_agent_inbox.v1",
    actor,
    action_count: actions.length,
    actions,
  };
}

export function renderHarnessStatus(status) {
  const lines = [
    "# NSRL Research Harness",
    "",
    `Experiments: **${status.experiment_count}**`,
    `Ledger events: **${status.event_count}**`,
    `Ledger head: \`${status.head_event_hash || "empty"}\``,
    "",
    "## Experiments",
    "",
  ];
  if (status.experiments.length === 0) {
    lines.push("- No experiments registered.");
  } else {
    for (const experiment of status.experiments) {
      const outcome = experiment.outcome ? `; outcome **${experiment.outcome}**` : "";
      lines.push(`- **${experiment.id}**: ${experiment.state}${outcome} (${experiment.event_count} events)`);
    }
  }
  return `${lines.join("\n")}\n`;
}
