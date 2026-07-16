import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

export const FACULTY_ORDER = [
  "mathematician",
  "engineer",
  "historian",
  "skeptic",
  "consequence_planner",
  "judge",
];

export const RECOMMENDING_FACULTIES = FACULTY_ORDER.filter((faculty) => faculty !== "judge");
export const DECISION_KINDS = ["select", "request_evidence", "ask_user", "abstain"];

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const shaPattern = /^[0-9a-f]{64}$/;
const dispositions = new Set(["support", "oppose", "abstain", "request_evidence", "ask_user"]);
const calibrationBuckets = new Set([
  "0-99", "100-199", "200-299", "300-399", "400-499",
  "500-599", "600-699", "700-799", "800-899", "900-1000",
]);

export function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

export function sha256Json(value) {
  return sha256Bytes(Buffer.from(stableJson(value)));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, required, optional, label) {
  assert(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const allowed = new Set([...required, ...optional]);
  for (const key of required) assert(Object.hasOwn(value, key), `${label} missing ${key}`);
  for (const key of Object.keys(value)) assert(allowed.has(key), `${label} has unknown field ${key}`);
}

function nonemptyString(value, label) {
  assert(typeof value === "string" && value.trim().length > 0, `${label} must be a nonempty string`);
}

function sha256String(value, label) {
  assert(typeof value === "string" && shaPattern.test(value), `${label} must be lowercase SHA-256`);
}

function boundedInteger(value, minimum, maximum, label) {
  assert(Number.isSafeInteger(value) && value >= minimum && value <= maximum,
    `${label} must be an integer in [${minimum}, ${maximum}]`);
}

function uniqueStrings(values, label, {allowEmpty = true} = {}) {
  assert(Array.isArray(values) && (allowEmpty || values.length > 0), `${label} must be an array`);
  for (const value of values) nonemptyString(value, `${label} item`);
  assert(new Set(values).size === values.length, `${label} must not repeat values`);
}

function subset(values, allowed, label) {
  const allowedSet = new Set(allowed);
  for (const value of values) assert(allowedSet.has(value), `${label} exceeds seal: ${value}`);
}

function manifestPayload(manifest) {
  const payload = structuredClone(manifest);
  delete payload.signature;
  return payload;
}

export function loadCouncilAuthority(base = root) {
  const trustPath = path.join(base, "council/trust-root-v0.json");
  const trustBytes = fs.readFileSync(trustPath);
  const trust = JSON.parse(trustBytes);
  assert(trust.schema === "nsrl.council_trust_root.v0", "wrong council trust-root schema");
  assert(trust.algorithm === "ed25519", "council trust root must use Ed25519");
  nonemptyString(trust.public_key_pem, "trust root public key");
  const manifests = new Map();
  for (const facultyId of FACULTY_ORDER) {
    const manifestPath = path.join(base, `council/seals/${facultyId}.json`);
    const bytes = fs.readFileSync(manifestPath);
    const manifest = JSON.parse(bytes);
    verifySeal(manifest, trust);
    assert(manifest.faculty_id === facultyId, `faculty seal filename mismatch: ${facultyId}`);
    manifests.set(facultyId, {
      manifest,
      path: path.relative(base, manifestPath),
      sha256: sha256Bytes(bytes),
    });
  }
  assert(new Set([...manifests.values()].map(({manifest}) => manifest.seal_id)).size
    === FACULTY_ORDER.length, "faculty seal ids must be unique");
  return {
    trust,
    trust_path: path.relative(base, trustPath),
    trust_sha256: sha256Bytes(trustBytes),
    manifests,
  };
}

export function verifySeal(manifest, trust) {
  exactKeys(manifest, [
    "schema", "seal_id", "faculty_id", "role", "version", "issuer_key_id",
    "capabilities", "allowed_permissions", "allowed_tools", "resource_ceiling",
    "issued_at", "signature",
  ], [], "seal");
  assert(manifest.schema === "nsrl.faculty_capability_seal.v0", "wrong faculty seal schema");
  assert(FACULTY_ORDER.includes(manifest.faculty_id), "unknown sealed faculty");
  assert(manifest.issuer_key_id === trust.key_id, "faculty seal issuer is not trusted");
  assert(manifest.version === 0, "faculty seal version must be zero");
  uniqueStrings(manifest.capabilities, "seal capabilities", {allowEmpty: false});
  uniqueStrings(manifest.allowed_permissions, "seal permissions");
  uniqueStrings(manifest.allowed_tools, "seal tools");
  exactKeys(manifest.resource_ceiling, [
    "input_bytes", "output_bytes", "tool_calls", "tokens", "wall_clock_ms",
  ], [], "seal resource ceiling");
  for (const [key, value] of Object.entries(manifest.resource_ceiling)) {
    boundedInteger(value, 0, Number.MAX_SAFE_INTEGER, `seal resource ceiling ${key}`);
  }
  exactKeys(manifest.signature, ["algorithm", "value_base64"], [], "seal signature");
  assert(manifest.signature.algorithm === "ed25519", "faculty seal signature must use Ed25519");
  const valid = crypto.verify(null, Buffer.from(stableJson(manifestPayload(manifest))),
    trust.public_key_pem, Buffer.from(manifest.signature.value_base64, "base64"));
  assert(valid, `faculty seal signature invalid: ${manifest.faculty_id}`);
  return true;
}

function validateCircle(circle, facultyId, seal) {
  exactKeys(circle, [
    "faculty_id", "permissions", "tools", "budget", "usage", "accessed_evidence_ids",
  ], [], `circle ${facultyId}`);
  assert(circle.faculty_id === facultyId, `circle faculty mismatch: ${facultyId}`);
  uniqueStrings(circle.permissions, `circle ${facultyId} permissions`);
  uniqueStrings(circle.tools, `circle ${facultyId} tools`);
  uniqueStrings(circle.accessed_evidence_ids, `circle ${facultyId} evidence`);
  subset(circle.permissions, seal.allowed_permissions, `circle ${facultyId} permission`);
  subset(circle.tools, seal.allowed_tools, `circle ${facultyId} tool`);
  exactKeys(circle.budget, [
    "input_bytes", "output_bytes", "tool_calls", "tokens", "wall_clock_ms",
  ], [], `circle ${facultyId} budget`);
  exactKeys(circle.usage, [
    "input_bytes", "output_bytes", "tool_calls", "tokens", "wall_clock_ms",
  ], [], `circle ${facultyId} usage`);
  for (const key of Object.keys(circle.budget)) {
    boundedInteger(circle.budget[key], 0, seal.resource_ceiling[key],
      `circle ${facultyId} budget ${key}`);
    boundedInteger(circle.usage[key], 0, circle.budget[key],
      `circle ${facultyId} usage ${key}`);
  }
}

function validateEvidence(evidence) {
  exactKeys(evidence, [
    "evidence_id", "source_uri", "source_sha256", "content_sha256", "retrieved_at",
    "summary", "claim_ids", "accessible_to",
  ], [], `evidence ${evidence?.evidence_id ?? "unknown"}`);
  nonemptyString(evidence.evidence_id, "evidence id");
  nonemptyString(evidence.source_uri, "evidence source URI");
  sha256String(evidence.source_sha256, "evidence source hash");
  sha256String(evidence.content_sha256, "evidence content hash");
  nonemptyString(evidence.retrieved_at, "evidence retrieval time");
  nonemptyString(evidence.summary, "evidence summary");
  uniqueStrings(evidence.claim_ids, "evidence claim ids", {allowEmpty: false});
  uniqueStrings(evidence.accessible_to, "evidence accessible faculties", {allowEmpty: false});
  subset(evidence.accessible_to, FACULTY_ORDER, "evidence faculty access");
}

function validateAction(action, evidenceIds) {
  exactKeys(action, [
    "action_id", "summary", "risk_milli", "reversible", "required_permissions",
    "required_tools", "evidence_ids",
  ], [], `action ${action?.action_id ?? "unknown"}`);
  nonemptyString(action.action_id, "action id");
  nonemptyString(action.summary, "action summary");
  boundedInteger(action.risk_milli, 0, 1000, "action risk");
  assert(typeof action.reversible === "boolean", "action reversibility must be boolean");
  uniqueStrings(action.required_permissions, "action permissions");
  uniqueStrings(action.required_tools, "action tools");
  uniqueStrings(action.evidence_ids, "action evidence");
  for (const evidenceId of action.evidence_ids) {
    assert(evidenceIds.has(evidenceId), `action cites unknown evidence ${evidenceId}`);
  }
}

function validateConsequence(consequence, actionIds, label) {
  exactKeys(consequence, [
    "action_id", "horizon", "description", "impact_milli", "confidence_milli",
  ], [], label);
  assert(actionIds.has(consequence.action_id), `${label} cites unknown action`);
  nonemptyString(consequence.horizon, `${label} horizon`);
  nonemptyString(consequence.description, `${label} description`);
  boundedInteger(consequence.impact_milli, -1000, 1000, `${label} impact`);
  boundedInteger(consequence.confidence_milli, 0, 1000, `${label} confidence`);
}

function validateRecommendation(recommendation, facultyId, actionIds, evidenceById, seal) {
  exactKeys(recommendation, [
    "disposition", "action_id", "rationale", "confidence_milli", "calibration_bucket",
    "evidence_ids", "contradictions", "predicted_consequences", "missing_information",
  ], [], `recommendation ${facultyId}`);
  assert(dispositions.has(recommendation.disposition), `unknown disposition for ${facultyId}`);
  const dispositionCapabilities = {
    support: ["recommend_action"],
    oppose: ["oppose_action", "recommend_action"],
    abstain: ["recommend_abstention", "recommend_action"],
    request_evidence: ["request_evidence"],
    ask_user: ["ask_user"],
  };
  assert(dispositionCapabilities[recommendation.disposition].some(
    (capability) => seal.capabilities.includes(capability)),
  `${facultyId} seal does not permit disposition ${recommendation.disposition}`);
  if (recommendation.action_id !== null) {
    assert(actionIds.has(recommendation.action_id), `${facultyId} cites unknown action`);
  }
  if (["support", "oppose"].includes(recommendation.disposition)) {
    assert(recommendation.action_id !== null, `${facultyId} ${recommendation.disposition} needs action`);
  }
  nonemptyString(recommendation.rationale, `${facultyId} rationale`);
  boundedInteger(recommendation.confidence_milli, 0, 1000, `${facultyId} confidence`);
  assert(calibrationBuckets.has(recommendation.calibration_bucket),
    `${facultyId} has unknown calibration bucket`);
  uniqueStrings(recommendation.evidence_ids, `${facultyId} recommendation evidence`);
  for (const evidenceId of recommendation.evidence_ids) {
    const evidence = evidenceById.get(evidenceId);
    assert(evidence, `${facultyId} cites unknown evidence ${evidenceId}`);
    assert(evidence.accessible_to.includes(facultyId), `${facultyId} cites inaccessible evidence ${evidenceId}`);
  }
  assert(Array.isArray(recommendation.contradictions), `${facultyId} contradictions must be an array`);
  for (const [index, contradiction] of recommendation.contradictions.entries()) {
    exactKeys(contradiction, [
      "severity", "action_id", "claim_id", "conflicts_with_claim_id", "explanation",
    ], [], `${facultyId} contradiction ${index}`);
    assert(["soft", "hard"].includes(contradiction.severity), "unknown contradiction severity");
    if (contradiction.action_id !== null) {
      assert(actionIds.has(contradiction.action_id), "contradiction cites unknown action");
    }
    nonemptyString(contradiction.claim_id, "contradiction claim id");
    nonemptyString(contradiction.conflicts_with_claim_id, "contradiction conflicting claim id");
    nonemptyString(contradiction.explanation, "contradiction explanation");
  }
  assert(Array.isArray(recommendation.predicted_consequences),
    `${facultyId} consequences must be an array`);
  for (const [index, consequence] of recommendation.predicted_consequences.entries()) {
    validateConsequence(consequence, actionIds, `${facultyId} consequence ${index}`);
  }
  assert(Array.isArray(recommendation.missing_information),
    `${facultyId} missing information must be an array`);
  for (const [index, missing] of recommendation.missing_information.entries()) {
    exactKeys(missing, ["kind", "action_id", "question"], [], `${facultyId} missing ${index}`);
    assert(["evidence", "user"].includes(missing.kind), "unknown missing-information kind");
    if (missing.action_id !== null) assert(actionIds.has(missing.action_id), "missing info cites unknown action");
    nonemptyString(missing.question, "missing-information question");
  }
}

function validateController(controller, actionIds) {
  exactKeys(controller, [
    "schema", "allowed_action_ids", "forbidden_action_ids", "max_risk_milli",
    "min_distinct_evidence_sources", "min_supporting_faculties",
    "min_support_confidence_milli", "require_skeptic_review",
    "require_reversible_at_or_above_risk_milli", "tie_break",
  ], [], "mathematical controller");
  assert(controller.schema === "nsrl.mathematical_controller.v0", "wrong controller schema");
  uniqueStrings(controller.allowed_action_ids, "controller allowed actions");
  uniqueStrings(controller.forbidden_action_ids, "controller forbidden actions");
  for (const actionId of [...controller.allowed_action_ids, ...controller.forbidden_action_ids]) {
    assert(actionIds.has(actionId), `controller cites unknown action ${actionId}`);
  }
  assert(controller.allowed_action_ids.every((id) => !controller.forbidden_action_ids.includes(id)),
    "controller action cannot be both allowed and forbidden");
  boundedInteger(controller.max_risk_milli, 0, 1000, "controller maximum risk");
  boundedInteger(controller.min_distinct_evidence_sources, 0, 1000, "controller source minimum");
  boundedInteger(controller.min_supporting_faculties, 1, RECOMMENDING_FACULTIES.length,
    "controller support minimum");
  boundedInteger(controller.min_support_confidence_milli, 0, 1000,
    "controller confidence minimum");
  assert(typeof controller.require_skeptic_review === "boolean", "skeptic review flag must be boolean");
  boundedInteger(controller.require_reversible_at_or_above_risk_milli, 0, 1001,
    "controller reversibility threshold");
  assert(controller.tie_break === "highest_margin_then_lexicographic_action_id",
    "unknown controller tie break");
}

export function validateCouncilRequest(request, authority = loadCouncilAuthority()) {
  exactKeys(request, [
    "schema", "request_id", "recorded_at", "mode", "question", "models", "evidence",
    "actions", "controller", "invocations",
  ], [], "council request");
  assert(request.schema === "nsrl.solomon_council_request.v0", "wrong council request schema");
  nonemptyString(request.request_id, "request id");
  nonemptyString(request.recorded_at, "recorded time");
  assert(request.mode === "shadow", "Solomon Council v0 is shadow-mode only");
  nonemptyString(request.question, "council question");
  assert(Array.isArray(request.models) && request.models.length > 0, "request needs model bindings");
  for (const [index, model] of request.models.entries()) {
    exactKeys(model, ["model_id", "artifact_uri", "artifact_sha256", "role"], [], `model ${index}`);
    nonemptyString(model.model_id, "model id");
    nonemptyString(model.artifact_uri, "model artifact URI");
    sha256String(model.artifact_sha256, "model artifact hash");
    nonemptyString(model.role, "model role");
  }
  assert(Array.isArray(request.evidence), "request evidence must be an array");
  request.evidence.forEach(validateEvidence);
  const evidenceById = new Map(request.evidence.map((evidence) => [evidence.evidence_id, evidence]));
  assert(evidenceById.size === request.evidence.length, "evidence ids must be unique");
  assert(Array.isArray(request.actions) && request.actions.length > 0, "request needs candidate actions");
  const evidenceIds = new Set(evidenceById.keys());
  request.actions.forEach((action) => validateAction(action, evidenceIds));
  const actionIds = new Set(request.actions.map((action) => action.action_id));
  assert(actionIds.size === request.actions.length, "action ids must be unique");
  validateController(request.controller, actionIds);
  assert(Array.isArray(request.invocations) && request.invocations.length === FACULTY_ORDER.length,
    "request must invoke exactly six faculties");
  const invocationByFaculty = new Map();
  for (const invocation of request.invocations) {
    exactKeys(invocation, [
      "invocation_id", "faculty_id", "seal_id", "circle", "recommendation",
    ], [], `invocation ${invocation?.faculty_id ?? "unknown"}`);
    nonemptyString(invocation.invocation_id, "invocation id");
    assert(FACULTY_ORDER.includes(invocation.faculty_id), "unknown invoked faculty");
    assert(!invocationByFaculty.has(invocation.faculty_id), "faculty invoked more than once");
    const seal = authority.manifests.get(invocation.faculty_id).manifest;
    assert(invocation.seal_id === seal.seal_id, `wrong seal for ${invocation.faculty_id}`);
    validateCircle(invocation.circle, invocation.faculty_id, seal);
    for (const evidenceId of invocation.circle.accessed_evidence_ids) {
      const evidence = evidenceById.get(evidenceId);
      assert(evidence, `${invocation.faculty_id} circle accessed unknown evidence ${evidenceId}`);
      assert(evidence.accessible_to.includes(invocation.faculty_id),
        `${invocation.faculty_id} circle accessed forbidden evidence ${evidenceId}`);
    }
    if (invocation.faculty_id === "judge") {
      assert(invocation.recommendation === null, "judge recommendation must be derived, not supplied");
    } else {
      validateRecommendation(
        invocation.recommendation, invocation.faculty_id, actionIds, evidenceById, seal);
      for (const evidenceId of invocation.recommendation.evidence_ids) {
        assert(invocation.circle.accessed_evidence_ids.includes(evidenceId),
          `${invocation.faculty_id} recommendation cites evidence outside its circle`);
      }
    }
    invocationByFaculty.set(invocation.faculty_id, invocation);
  }
  for (const faculty of FACULTY_ORDER) assert(invocationByFaculty.has(faculty), `missing faculty ${faculty}`);
  return {authority, evidenceById, actionIds, invocationByFaculty};
}

function evaluateAction(action, request, context) {
  const recommendations = RECOMMENDING_FACULTIES.map(
    (faculty) => context.invocationByFaculty.get(faculty).recommendation);
  const supports = recommendations.filter(
    (recommendation) => recommendation.action_id === action.action_id
      && recommendation.disposition === "support");
  const opposes = recommendations.filter(
    (recommendation) => recommendation.action_id === action.action_id
      && recommendation.disposition === "oppose");
  const citedEvidence = new Set(supports.flatMap((recommendation) => recommendation.evidence_ids));
  const sourceHashes = new Set(action.evidence_ids.filter((id) => citedEvidence.has(id)).map(
    (id) => context.evidenceById.get(id).source_sha256));
  const contradictions = recommendations.flatMap((recommendation) => recommendation.contradictions)
    .filter((contradiction) => contradiction.action_id === action.action_id
      && contradiction.severity === "hard");
  const skeptic = context.invocationByFaculty.get("skeptic").recommendation;
  const judgeCircle = context.invocationByFaculty.get("judge").circle;
  const missing = recommendations.flatMap((recommendation) => recommendation.missing_information)
    .filter((item) => item.action_id === null || item.action_id === action.action_id);
  const policy = request.controller;
  const checks = {
    explicitly_allowed: policy.allowed_action_ids.includes(action.action_id),
    not_forbidden: !policy.forbidden_action_ids.includes(action.action_id),
    risk_inside_limit: action.risk_milli <= policy.max_risk_milli,
    reversible_when_required: action.risk_milli < policy.require_reversible_at_or_above_risk_milli
      || action.reversible,
    permissions_inside_judge_circle: action.required_permissions.every(
      (permission) => judgeCircle.permissions.includes(permission)),
    tools_inside_judge_circle: action.required_tools.every((tool) => judgeCircle.tools.includes(tool)),
    evidence_source_minimum: sourceHashes.size >= policy.min_distinct_evidence_sources,
    faculty_support_minimum: supports.length >= policy.min_supporting_faculties,
    support_confidence_floor: supports.length > 0 && supports.every(
      (recommendation) => recommendation.confidence_milli >= policy.min_support_confidence_milli),
    skeptic_review_present: !policy.require_skeptic_review
      || (skeptic.action_id === action.action_id && ["support", "oppose"].includes(skeptic.disposition)),
    no_hard_contradiction: contradictions.length === 0,
    no_material_missing_information: missing.length === 0,
  };
  const consequenceMargin = recommendations.flatMap(
    (recommendation) => recommendation.predicted_consequences)
    .filter((consequence) => consequence.action_id === action.action_id)
    .reduce((sum, consequence) => sum + Number(
      (BigInt(consequence.impact_milli) * BigInt(consequence.confidence_milli)) / 1000n), 0);
  const margin = supports.reduce((sum, recommendation) => sum + recommendation.confidence_milli, 0)
    - opposes.reduce((sum, recommendation) => sum + recommendation.confidence_milli, 0)
    - action.risk_milli + consequenceMargin;
  return {
    action_id: action.action_id,
    controller_allowed: Object.values(checks).every(Boolean),
    checks,
    supporting_faculties: RECOMMENDING_FACULTIES.filter((faculty) => {
      const recommendation = context.invocationByFaculty.get(faculty).recommendation;
      return recommendation.action_id === action.action_id && recommendation.disposition === "support";
    }),
    opposing_faculties: RECOMMENDING_FACULTIES.filter((faculty) => {
      const recommendation = context.invocationByFaculty.get(faculty).recommendation;
      return recommendation.action_id === action.action_id && recommendation.disposition === "oppose";
    }),
    distinct_cited_source_hashes: [...sourceHashes].sort(),
    hard_contradictions: contradictions,
    missing_information: missing,
    margin_milli: margin,
  };
}

function deterministicDecision(request, context, comparisons) {
  const eligible = comparisons.filter((comparison) => comparison.controller_allowed).sort(
    (left, right) => right.margin_milli - left.margin_milli
      || left.action_id.localeCompare(right.action_id));
  if (eligible.length > 0) {
    const selected = eligible[0];
    const confidenceValues = selected.supporting_faculties.map(
      (faculty) => context.invocationByFaculty.get(faculty).recommendation.confidence_milli);
    return {
      kind: "select",
      selected_action_id: selected.action_id,
      rationale: "highest controller-allowed integer margin; lexicographic action id breaks exact ties",
      confidence_milli: Math.min(...confidenceValues),
      calibration: {
        method: "minimum supporting-faculty confidence",
        bucket: context.invocationByFaculty.get(selected.supporting_faculties[0])
          .recommendation.calibration_bucket,
        outcome_observed: false,
      },
      mathematical_controller_allowed: true,
      questions: [],
    };
  }
  const recommendations = RECOMMENDING_FACULTIES.map(
    (faculty) => context.invocationByFaculty.get(faculty).recommendation);
  const missing = recommendations.flatMap((recommendation) => recommendation.missing_information);
  const userQuestions = [...new Set(missing.filter((item) => item.kind === "user")
    .map((item) => item.question))].sort();
  if (userQuestions.length > 0) {
    return {
      kind: "ask_user", selected_action_id: null,
      rationale: "a faculty identified material information that only the user can provide",
      confidence_milli: 1000, calibration: {
        method: "deterministic missing-information rule", bucket: "900-1000", outcome_observed: false,
      },
      mathematical_controller_allowed: false, questions: userQuestions,
    };
  }
  const evidenceQuestions = [...new Set([
    ...missing.filter((item) => item.kind === "evidence").map((item) => item.question),
    ...comparisons.flatMap((comparison) => comparison.hard_contradictions.map(
      (contradiction) => contradiction.explanation)),
  ])].sort();
  if (evidenceQuestions.length > 0) {
    return {
      kind: "request_evidence", selected_action_id: null,
      rationale: "material evidence is missing or a hard contradiction remains unresolved",
      confidence_milli: 1000, calibration: {
        method: "deterministic evidence-deficiency rule", bucket: "900-1000", outcome_observed: false,
      },
      mathematical_controller_allowed: false, questions: evidenceQuestions,
    };
  }
  return {
    kind: "abstain", selected_action_id: null,
    rationale: "no candidate action satisfies every mathematical-controller check",
    confidence_milli: 1000,
    calibration: {
      method: "deterministic controller rejection", bucket: "900-1000", outcome_observed: false,
    },
    mathematical_controller_allowed: false, questions: [],
  };
}

function dissentFor(decision, request, context) {
  return RECOMMENDING_FACULTIES.flatMap((faculty) => {
    const recommendation = context.invocationByFaculty.get(faculty).recommendation;
    const agrees = decision.kind === "select"
      ? recommendation.action_id === decision.selected_action_id
        && recommendation.disposition === "support"
      : recommendation.disposition === decision.kind
        || (decision.kind === "abstain" && recommendation.disposition === "abstain");
    if (agrees) return [];
    return [{
      faculty_id: faculty,
      disposition: recommendation.disposition,
      action_id: recommendation.action_id,
      rationale: recommendation.rationale,
      confidence_milli: recommendation.confidence_milli,
    }];
  });
}

export function receiptHash(receipt) {
  const unsigned = structuredClone(receipt);
  delete unsigned.identity.receipt_sha256;
  return sha256Json(unsigned);
}

export function verifyReceiptIdentity(receipt) {
  assert(receipt.schema === "nsrl.wisdom_receipt.v0", "wrong wisdom receipt schema");
  assert(receipt.mode === "shadow", "wisdom receipt escaped shadow mode");
  assert(receipt.identity.receipt_sha256 === receiptHash(receipt), "wisdom receipt identity changed");
  assert(receipt.shadow_execution.action_execution_allowed === false
    && receipt.shadow_execution.action_executed === false, "wisdom receipt claims execution");
  return true;
}

export function deliberate(request, authority = loadCouncilAuthority()) {
  const context = validateCouncilRequest(request, authority);
  const comparisons = request.actions.map((action) => evaluateAction(action, request, context));
  const decision = deterministicDecision(request, context, comparisons);
  assert(DECISION_KINDS.includes(decision.kind), "judge emitted unknown decision kind");
  if (decision.kind === "select") {
    const selected = comparisons.find((comparison) => comparison.action_id === decision.selected_action_id);
    assert(selected?.controller_allowed, "judge selected a controller-forbidden action");
  }
  const invocationRecords = FACULTY_ORDER.map((facultyId) => {
    const invocation = context.invocationByFaculty.get(facultyId);
    const seal = authority.manifests.get(facultyId);
    return {
      invocation_id: invocation.invocation_id,
      faculty_id: facultyId,
      seal: {
        seal_id: seal.manifest.seal_id,
        manifest_path: seal.path,
        manifest_sha256: seal.sha256,
        signature_verified: true,
      },
      circle: invocation.circle,
      recommendation: invocation.recommendation,
    };
  });
  const selectedConsequences = decision.kind === "select"
    ? invocationRecords.filter((record) => record.recommendation).flatMap(
      (record) => record.recommendation.predicted_consequences)
      .filter((consequence) => consequence.action_id === decision.selected_action_id)
    : [];
  const receipt = {
    schema: "nsrl.wisdom_receipt.v0",
    receipt_id: `wisdom-${sha256Json(request).slice(0, 24)}`,
    recorded_at: request.recorded_at,
    mode: "shadow",
    request: {
      request_id: request.request_id,
      question: request.question,
      request_sha256: sha256Json(request),
    },
    bindings: {
      models: request.models,
      trust_root: {
        path: authority.trust_path,
        sha256: authority.trust_sha256,
        key_id: authority.trust.key_id,
      },
      sources: request.evidence.map((evidence) => ({
        evidence_id: evidence.evidence_id,
        source_uri: evidence.source_uri,
        source_sha256: evidence.source_sha256,
        content_sha256: evidence.content_sha256,
      })),
    },
    retrieved_evidence: request.evidence,
    faculty_invocations: invocationRecords,
    deliberation: {
      controller_schema: request.controller.schema,
      comparisons,
      contradictions: invocationRecords.filter((record) => record.recommendation).flatMap(
        (record) => record.recommendation.contradictions.map((contradiction) => ({
          faculty_id: record.faculty_id, ...contradiction,
        }))),
      dissent: dissentFor(decision, request, context),
    },
    decision,
    predicted_consequences: selectedConsequences,
    permissions_and_budget: invocationRecords.map((record) => ({
      faculty_id: record.faculty_id,
      permissions: record.circle.permissions,
      tools: record.circle.tools,
      budget: record.circle.budget,
      usage: record.circle.usage,
    })),
    shadow_execution: {
      selected_action_id: decision.selected_action_id,
      controller_authorized_recommendation: decision.mathematical_controller_allowed,
      action_execution_allowed: false,
      action_executed: false,
      reason: "Council v0 records recommendations only; its circle grants no execution authority.",
    },
    outcome: {
      status: "pending",
      observed_at: null,
      metrics: [],
      notes: "No outcome is inferred in shadow mode.",
    },
    revisions: [],
    identity: {
      canonicalization: "recursive lexicographic JSON keys, UTF-8, no insignificant whitespace",
      receipt_sha256: "",
    },
  };
  receipt.identity.receipt_sha256 = receiptHash(receipt);
  return receipt;
}

export function verifyReceipt(receipt, request, authority = loadCouncilAuthority()) {
  verifyReceiptIdentity(receipt);
  assert(receipt.request.request_sha256 === sha256Json(request), "wisdom receipt request binding changed");
  const replay = deliberate(request, authority);
  assert(stableJson(replay) === stableJson(receipt), "wisdom receipt deterministic replay changed");
  return true;
}

export function reviseReceipt(priorReceipt, observation) {
  verifyReceiptIdentity(priorReceipt);
  exactKeys(observation, [
    "schema", "receipt_id", "observed_at", "observer", "metrics", "notes", "revision",
  ], [], "wisdom outcome observation");
  assert(observation.schema === "nsrl.wisdom_outcome_observation.v0",
    "wrong wisdom outcome observation schema");
  assert(observation.receipt_id === priorReceipt.receipt_id, "observation receipt id changed");
  nonemptyString(observation.observed_at, "observation time");
  nonemptyString(observation.observer, "observation observer");
  assert(Array.isArray(observation.metrics), "observation metrics must be an array");
  for (const [index, metric] of observation.metrics.entries()) {
    exactKeys(metric, ["metric_id", "value", "unit"], [], `observation metric ${index}`);
    nonemptyString(metric.metric_id, "observation metric id");
    assert(["string", "number", "boolean"].includes(typeof metric.value),
      "observation metric value must be scalar");
    nonemptyString(metric.unit, "observation metric unit");
  }
  assert(typeof observation.notes === "string", "observation notes must be a string");
  exactKeys(observation.revision, [
    "reason", "confidence_milli", "calibration_bucket",
  ], [], "wisdom revision");
  nonemptyString(observation.revision.reason, "revision reason");
  boundedInteger(observation.revision.confidence_milli, 0, 1000, "revised confidence");
  assert(calibrationBuckets.has(observation.revision.calibration_bucket),
    "unknown revised calibration bucket");
  const revised = structuredClone(priorReceipt);
  revised.outcome = {
    status: "observed",
    observed_at: observation.observed_at,
    metrics: observation.metrics,
    notes: observation.notes,
  };
  revised.revisions.push({
    revision_index: revised.revisions.length + 1,
    recorded_at: observation.observed_at,
    observer: observation.observer,
    prior_receipt_sha256: priorReceipt.identity.receipt_sha256,
    observation_sha256: sha256Json(observation),
    reason: observation.revision.reason,
    prior_confidence_milli: priorReceipt.revisions.length === 0
      ? priorReceipt.decision.confidence_milli
      : priorReceipt.revisions.at(-1).revised_confidence_milli,
    revised_confidence_milli: observation.revision.confidence_milli,
    revised_calibration_bucket: observation.revision.calibration_bucket,
    decision_changed: false,
  });
  revised.identity.receipt_sha256 = receiptHash(revised);
  return revised;
}

export function verifyReceiptRevision(revisedReceipt, priorReceipt, observation) {
  verifyReceiptIdentity(revisedReceipt);
  const expected = reviseReceipt(priorReceipt, observation);
  assert(stableJson(expected) === stableJson(revisedReceipt), "wisdom receipt revision replay changed");
  return true;
}
