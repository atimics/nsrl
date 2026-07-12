# Integer Transformer Proof v1

This is NSRL's single promotion milestone. It asks one question:

> Does an NSRL-born deterministic integer transformer learn a frozen next-token
> task better than retrieval, byte n-gram, and independently produced
> floating-point reference baselines?

Solomon and literary routing remain valuable experiment suites, but neither is
the project headline until the substrate passes this proof.

## Frozen contract

- Contract: `integer-transformer-proof-v1`
- Headline suite: `substrate`
- Partition: `eval`
- Candidate: one `NSRLMT5` integer transformer checkpoint
- Required baselines: `retrieval`, `byte-ngram`, `float-reference`
- Primary metric: aggregate `probability_error_q15`, lower is better
- Secondary metric: `mistakes`, lower is better
- Replay: every result row carries a deterministic 64-bit replay hash
- Manifest: `benchmarks/integer-transformer-proof-v1/manifest.tsv`
- Dataset hash: `0x8fe7b86378f81951`
- Context: 64 bytes
- Stride: 1
- Held-out targets: 5,896

The candidate passes only when its probability error is strictly lower than
every baseline and its mistake count is no higher than every baseline on the
same frozen dataset hash and target count. A routing oracle is diagnostic and
cannot occupy a baseline or candidate row.

The train and evaluation bytes are checked in beside the manifest. The hash
binds `train.txt`, a partition separator, and `eval.txt`; changing content,
partition boundaries, geometry, or the minimum target count requires a new
contract version.

## Typed result surface

Results use a TSV with this exact header:

```text
schema	contract	suite	partition	dataset_hash	system	targets	mistakes	probability_error_q15	replay_hash
```

Inspect the machine-readable contract:

```bash
cargo run -p nsrl-eval -- contract
```

Validate the frozen manifest and checked-in baseline matrix:

```bash
cargo run -p nsrl-eval -- manifest \
  --manifest benchmarks/integer-transformer-proof-v1/manifest.tsv

cargo run -p nsrl-eval -- check-baselines \
  --manifest benchmarks/integer-transformer-proof-v1/manifest.tsv \
  --results benchmarks/integer-transformer-proof-v1/baselines.tsv
```

Regenerate the deterministic baseline matrix:

```bash
node scripts/run-integer-transformer-proof-baselines.mjs \
  --manifest benchmarks/integer-transformer-proof-v1/manifest.tsv \
  --out benchmarks/integer-transformer-proof-v1/baselines.tsv
```

Evaluate a context-64 candidate and assemble the full matrix:

```bash
cargo run -p nsrl-train --bin nsrl-mini-transformer-eval -- \
  --tokens benchmarks/integer-transformer-proof-v1/eval.txt \
  --model path/to/candidate.nsrlmt \
  --stride 1 \
  --attention linear \
  --position nope \
  --out path/to/candidate-eval.json

node scripts/build-integer-transformer-proof-results.mjs \
  --manifest benchmarks/integer-transformer-proof-v1/manifest.tsv \
  --baselines benchmarks/integer-transformer-proof-v1/baselines.tsv \
  --candidate-trace path/to/candidate-eval.json \
  --out path/to/proof-results.tsv

cargo run -p nsrl-eval -- check \
  --manifest benchmarks/integer-transformer-proof-v1/manifest.tsv \
  --results path/to/proof-results.tsv
```

The final checker exits zero only for a valid passing proof, one for a valid
failing proof, and two for a malformed or contract-incompatible artifact.

For a reproducible train-evaluate-check run using the repository defaults:

```bash
scripts/run-integer-transformer-proof-candidate.sh \
  data/experiments/integer-transformer-proof-v1/candidate-default
```

The runner exits one when the candidate is valid but fails the benchmark; a
failed proof is measured evidence, not an infrastructure failure.

## Experiment boundaries

- `substrate` owns the frozen benchmark, baselines, model comparison, artifact
  compatibility, and promotion decision.
- `literary` explores expert capacity, routing granularity, and learned
  specialization. Its results may nominate a candidate architecture.
- `solomon` explores symbolic multimodal generation, retrieval grounding, and
  browser deployment. Its gates remain product evidence, not substrate proof.

Experiment-specific schemas must not redefine the substrate pass rule. A
promoted finding moves into the substrate suite through an explicit contract
version or candidate change.

## Architectural boundary

`nsrl-eval` owns proof contracts and comparison policy. `nsrl-train` owns model
training and serialization. Frozen training artifact identifiers live in
`nsrl_train::artifact_contract`; changing a persisted magic or schema requires
an explicit compatibility review rather than an incidental edit in the trainer.

The next extraction boundary is model serialization itself, followed by
optimizer state and experiment runners. Each extraction must preserve the
existing public API until callers have migrated.
