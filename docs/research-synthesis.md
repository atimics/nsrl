# Cross-Project Research Synthesis

This note records what NSRL should learn from four nearby projects:

- `/Users/ratimics/develop/crlplrimes`
- `/Users/ratimics/develop/asix`
- `/Users/ratimics/develop/holonet`
- `/Users/ratimics/develop/signal`

The user-facing request named these as `../crlplrimes`, `../asix`,
`../holonet`, and `../signal`. In the current local layout, the repositories
resolve under `/Users/ratimics/develop`.

## Executive Summary

NSRL should not present itself as "an integer way to run ordinary transformer
checkpoints." The stronger, cleaner position is:

```text
NSRL is a deterministic integer-native neural runtime whose mathematical
contracts, traces, and demos are auditable from the arithmetic up.
```

The sibling projects converge on four rules:

1. Every demo needs a declared authority boundary.
2. Every learned or replayed claim needs trace rows and stable identifiers.
3. Fuzzy or learned memory may advise, but exact state must decide.
4. Bounded, deterministic systems win trust through replay, diagnostics, and
   explicit non-claims.

For NSRL, this means the next demo should be a small transformer-shaped integer
forward pass with a replayable JSON trace, not a language-model claim. Training
comes after the forward trace contract is stable.

## crlplrimes

`crlplrimes` is the clearest source for evidence discipline. Its central loop is
candidate enumeration, explicit grounding, learned residual scoring, and
replayable trace output. The design invariant is:

```text
The model proposes. The declared grounding operator decides within its authority.
The trace remembers.
```

Relevant patterns:

- Grounding authorities are explicit: `exact_check`, `deterministic_replay`,
  `corpus_proxy`, and `human_review`.
- Promoted evidence is intentionally narrower than sandbox work.
- Task schemas declare candidate grammar, feature provenance, leakage policy,
  outcome row schema, trace protocol, replay protocol, and known non-claims.
- Trace-producing rankers use stable action IDs, deterministic ranking policy,
  and score-margin guards.
- Numeric certificates can prove optimizer hygiene, but they do not promote a
  semantic claim without the domain grounding operator.

NSRL implications:

- Define NSRL authority levels before demos:
  - `integer_runtime_determinism`: same integer inputs and weights produce the
    same integer outputs and hash.
  - `deterministic_training_replay`: a training run can be replayed from seed,
    data, and arithmetic contract.
  - `corpus_proxy`: a text benchmark measures corpus fit only.
  - `human_review`: qualitative behavior is review-gated, not self-certified.
- Declare the key non-claim now: NSRL does not run standard GPT/Llama weights
  unless they were trained or calibrated for NSRL's base-2 attention and static
  residual scale contract.
- Add trace schemas before expanding benchmarks. A demo without trace rows is
  just a showpiece; a demo with trace rows can become evidence.

## asix

`asix` is a compact holographic scheduling kernel. It stores successful
state-action pairs by accumulating a vector-symbolic memory:

```text
M += bind(state, action)
score_i = dot(unbind(M, query_state), action_i)
```

Relevant patterns:

- Dimensions are power-of-two and precomputed FFT tables are built at startup.
- The hot path uses fixed work buffers and cached spectra.
- Memory is bounded: O(dim) regardless of stored examples.
- Retrieval quality is tracked as a fidelity estimate around `1 / sqrt(N)`.
- Store-on-reward is simple and robust: successful actions reinforce future
  selection; failed actions naturally lose selection share.
- The lab treats ablations as first-class outputs: full model, no negative
  trace, no encoder learning, prior-only, contextual fallback, residual trace,
  and others.

NSRL implications:

- Preserve the existing no-allocation, precomputed-LUT instinct in `nsrl-core`.
- Add runtime diagnostics that make precision loss visible: saturation counts,
  residual add saturation, softmax zeroed probability counts, accumulator headroom,
  and per-layer output hashes.
- Treat ablations as product features for research. For example:
  `base2_softmax` vs lookup-disabled reference, causal vs unmasked attention,
  static residual projection vs dynamic rescale, one block vs two blocks.
- If NSRL later adds an associative memory component, use it as an advisory
  router or residual scorer, not as a truth source.

## holonet

`holonet` reframes the holographic work with useful honesty. It explicitly says
the mechanism is classical VSA/HRR via FFT circular convolution, not quantum
computation. It also names the limits of holographic ranking:

- It is a zero-parameter action scorer, not a transformer.
- It has no sequence modeling by itself.
- Raw continuous encoding fails when every state maps to a unique key.
- Bucketing, clustering, and gate-aware encoding are the difference between
  useful recall and noise.
- Holographic memory degrades predictably with `1 / sqrt(N)` interference.

NSRL implications:

- Be equally precise about the base-2 attention claim. NSRL's softmax is not an
  approximation to Euler softmax unless a caller explicitly inserts a
  `log2(e)` conversion. The cleaner contract is native log2-temperature
  attention.
- Make "what this is not" part of the public design:
  - not PyTorch compatibility,
  - not standard checkpoint quantization,
  - not proof of language ability from a forward demo,
  - not a floating-point runtime with integer-looking wrappers.
- If a memory or routing module is introduced, require explicit integer
  bucketing or learned routing. Raw continuous keys should be treated as a
  known failure mode.

## signal

`signal` is the strongest source for deterministic systems practice. It uses a
fixed-tick simulation, authoritative state, replay tools, exact hashes, and a
sharp separation between truth and gossip.

Relevant patterns:

- `signal_replay` rebuilds a deterministic world from seed and input prefix,
  branches candidate actions, and emits JSONL rows.
- Replay rows include schema IDs, prefix/state/event hashes, metrics, and an
  authority string such as `deterministic_seed_prefix_replay`.
- Determinism checks compare repeated runs and native-vs-WASM outputs.
- Float fields are hashed as exact IEEE-754 bits so drift is visible.
- The holographic gossip design separates exact authoritative state from lossy
  advisory memory. Holograms may suggest or bias work, but they must not pay,
  verify, or mutate ledgers.

NSRL implications:

- The first real demo should emit a replay row with:
  - schema ID,
  - arithmetic contract,
  - model hash,
  - input token IDs,
  - output logits or chosen token,
  - per-layer saturation/headroom stats,
  - final output hash,
  - authority `integer_runtime_determinism`.
- Add a determinism gate that runs the same demo twice and diffs JSONL output.
- Later, add a cross-build gate such as debug vs release, native vs WASM, or
  x86_64 vs aarch64 when available.
- Keep any advisory memory/training trace separate from the exact forward
  runtime state.

## Cross-Cutting Lessons

### 1. Authority comes before claims

The project should always say what decided the result. For NSRL:

- The runtime can certify integer determinism.
- A replay harness can certify deterministic training replay.
- A corpus benchmark can certify corpus-proxy performance only.
- A human can review usefulness, but the model cannot self-authorize that claim.

### 2. Trace rows are the evidence surface

Every demo should leave a machine-readable trail. This is more important than
having a polished sample text early. A trace row should be replayable, diffable,
and tied to a schema version.

### 3. Integer-native means native contracts, not mimicry

The base-2 attention design is strongest when treated as the mathematical
definition of NSRL attention. Avoid paying a fixed-point multiply and rounding
tax just to imitate `e^x`.

### 4. Bounded precision needs visible health metrics

Fixed-point systems fail by slow signal erosion. NSRL should expose headroom and
precision health at every layer:

- accumulator max magnitude,
- right-shift amount,
- saturation count,
- masked attention count,
- zero-probability count,
- residual add saturation count,
- per-layer hash.

### 5. Fuzzy memory is advisory

The holographic projects are valuable, but they also warn against overreach.
Associative memory is useful when state encoding clusters similar decisions. It
is not sequence modeling, not truth, and not a substitute for the transformer
core if the goal is language.

## Recommended NSRL Demo Path

### Demo 1: deterministic integer transformer forward pass

Build `crates/nsrl-demo` around the existing `nsrl-core` primitives:

```text
char or byte tokens
  -> integer embedding
  -> one pre-norm base-2 attention block
  -> integer output head
  -> JSONL trace and output hash
```

This demo should make one claim only:

```text
Given fixed integer weights and inputs, NSRL executes a transformer-shaped
base-2 attention block deterministically with no runtime floats.
```

Required trace fields:

- `schema`: `nsrl.forward_trace.v1`
- `authority`: `integer_runtime_determinism`
- `arithmetic`: residual scale, rounding mode, head dimension, LUT versions
- `model_hash`
- `input_ids`
- `layer_stats`
- `output_logits`
- `output_hash`
- `known_non_claims`

### Demo 2: replay gate

Add a command that runs the same forward trace twice and diffs the JSONL bytes:

```text
cargo run -p nsrl-demo -- --trace /tmp/a.jsonl
cargo run -p nsrl-demo -- --trace /tmp/b.jsonl
diff -u /tmp/a.jsonl /tmp/b.jsonl
```

The expected result is an empty diff.

### Demo 3: tiny native training smoke

After the forward contract is stable, add `nsrl-train` for a very small task:

- byte copy,
- next character over a tiny alphabet,
- or bracket/parenthesis prediction.

The training demo should not claim general language modeling. It should claim
only that NSRL can update integer-native weights under the same arithmetic
contract used by inference, then replay the run from seed and data.

## Engineering Backlog From This Research

Completed baseline:

- `docs/schemas.md` defines `nsrl.forward_trace.v1` and sketches the future
  `nsrl.training_trace.v1`.
- `crates/nsrl-demo` runs a fixed one-block transformer model and emits JSONL
  output.
- The demo trace includes model/input/output hashes, per-layer integer health
  diagnostics, attention and MLP residual saturation counts, attention preview
  values, and `known_non_claims`.
- `nsrl-demo` has a replay test that runs the binary twice and compares trace
  bytes.
- `nsrl-demo --preset bench-1m` runs a 4-block, 128-token, 128-wide,
  1,048,576-i8-weight forward benchmark with compact memory, timing, hash, and
  saturation diagnostics.

Remaining work:

1. Promote per-layer statistics from demo-side diagnostics into reusable
   `nsrl-core` instrumentation where appropriate.
2. Add native/WASM or cross-machine replay gates.
3. Add `nsrl-train` and `nsrl.training_trace.v1` for a tiny native training
   smoke.
4. Keep holographic memory as a future advisory/router experiment, not part of
   the first transformer demo.

## Completion Criteria For The First Credible Demo

The demo is credible when all of these are true:

- `cargo test --workspace -- --test-threads=1` passes.
- A demo command runs without runtime floats in the core path.
- The demo emits `nsrl.forward_trace.v1` JSONL.
- Two identical demo runs produce byte-identical trace files.
- The trace includes saturation/headroom diagnostics.
- The README or demo output states the non-claims clearly.
