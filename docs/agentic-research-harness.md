# Agentic Research Harness v1

The NSRL research harness is a provider-independent control plane for agents
that propose, review, execute, audit, and interpret computational experiments.
It wraps the repository's existing Rust runners and experiment-specific
checkers. It does not replace their scientific contracts.

The harness makes research actions typed and stateful:

```text
draft
  -> reviewed
  -> frozen
  -> running
  -> run-complete
  -> audited
  -> supported | falsified | inconclusive
```

Execution failures and invalid audits are represented separately from
scientific falsification. A failed runner is not evidence against a hypothesis.

## What v1 enforces

- Every experiment declares a hypothesis, estimand, falsifier, evidence level,
  independent unit, power floor, controls, and multiplicity family size.
- Input source, model, tokenizer, data, evaluator, and scientific-contract files
  are SHA-256 bound when the harness contract is frozen.
- The tracked runner allowlist and checker policy are also hash-bound. Editing
  the policy or any bound input after freeze blocks execution and audit.
- Runners are fixed argument arrays, not agent-supplied shell commands.
- Expected output paths and permitted evidence-partition labels are allowlisted.
- Reserved evidence and paid compute require both frozen contract authorization
  and an explicit runtime flag.
- Proposer, reviewer, runner, auditor, and curator identities are separated.
- Lifecycle events form an append-only, globally hash-chained JSONL ledger.
- Final outcomes are computed from structured JSON-path predicates; agents do
  not provide executable decision expressions.
- Concurrent agents may ask for their next eligible action. Ledger locking and
  state transitions ensure that only the first valid transition succeeds.

The experiment schema is
[`protocol/research-experiment-v1.schema.json`](../protocol/research-experiment-v1.schema.json).
Tracked runner policy and agent-role descriptions live in
[`research/harness/`](../research/harness/).

## Runtime state

The default runtime directory is:

```text
data/research-harness/
  events.jsonl
  experiments/EXPERIMENT_ID/
    proposal.json
    contract.json
    run-receipt.json
    run.stdout.txt
    run.stderr.txt
    audit.json
    audit.stdout.txt
    audit.stderr.txt
    decision.json
```

`data/` is ignored by Git. Promote only reviewed, frozen evidence into
`benchmarks/` or another tracked evidence surface.

## Golden workflow

Initialize the runtime ledger and import the completed Boolean-jet confirmation:

```bash
node scripts/research-harness.mjs init
node scripts/research-harness.mjs import-golden
node scripts/research-harness.mjs verify
node scripts/research-harness.mjs status
```

The import hashes the original source, model, tokenizer, token stream,
scientific contract, runner, and checker. It imports the existing result, runs
the independent checker, applies the frozen structured decision rule, and
records the experiment as `falsified`.

When the default ledger exists, `node scripts/nsrl-status.mjs` includes its
verified experiment/event counts and terminal outcomes in the canonical project
truth surface. A broken ledger is reported as a project warning.

## New experiment workflow

Create a proposal conforming to the v1 schema, then use distinct actor
identities for each authority:

```bash
node scripts/research-harness.mjs register path/to/experiment.json \
  --actor agent:theory-01 --role theorist

node scripts/research-harness.mjs review EXPERIMENT_ID --approve \
  --actor agent:stats-01 --role statistician

node scripts/research-harness.mjs freeze EXPERIMENT_ID \
  --actor agent:protocol-01 --role protocol

node scripts/research-harness.mjs run EXPERIMENT_ID \
  --actor agent:runner-01 --role runner

node scripts/research-harness.mjs audit EXPERIMENT_ID \
  --actor agent:audit-01 --role auditor

node scripts/research-harness.mjs decide EXPERIMENT_ID \
  --actor agent:curator-01 --role curator
```

An agent can discover currently eligible work without receiving authority to
perform a different lifecycle action:

```bash
node scripts/research-harness.mjs next \
  --actor agent:audit-01 --role auditor
```

To bind an already completed result rather than execute its runner, use
`import-run` in the frozen state.

## Adding a runner

Runner templates are trusted policy. Add a template only after reviewing its
fixed command, checker, evidence partitions, output paths, and compute class:

```json
{
  "command": ["node", "scripts/run-example.mjs"],
  "checker": ["node", "scripts/check-example.mjs"],
  "allowed_partitions": ["proposal"],
  "expected_outputs": ["data/example/result.json"],
  "paid_compute": false
}
```

Changing the policy intentionally invalidates execution under contracts frozen
against its previous hash. Freeze a new experiment version after policy edits.

## Security boundary

V1 enforces evidence policy at the contract and allowlisted-runner boundary. It
does **not** create an operating-system data firewall. An agent process with
unrestricted read access to the repository can still inspect any local file.

Before agents are allowed to consume genuinely blind replication data, move
that data outside their workspace and expose a narrow evaluator service. The
service should accept a frozen contract/evaluator/model digest and return a
signed result receipt without exposing raw examples. The current
`reserved_evidence` gate is the control-plane interface for that future service,
not a claim that local filesystem blindness already exists.

## Verification

Run the adversarial lifecycle test directly:

```bash
node scripts/check-research-harness-v1.mjs
```

It verifies the full lifecycle, role separation, input-binding tamper rejection,
runner-policy tamper rejection, ledger-chain tamper rejection, agent inbox, and
the falsified outcome of the imported Boolean-jet golden experiment.
