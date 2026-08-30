# Q22 shared-task adapter

The Solomon family now has a concrete adapter for the shared Q22 operation
routing task. It consumes the exact training JSONL and promotion TSV published
by Zero. The manifest freezes both common SHA-256 values, the encoded Solomon
training SHA-256, record counts, the Solomon encoding ID, and the Solomon
verifier ID.

The training encoder writes byte-identity records of this form:

```text
[q22]
input: add -2 7
operation: quantity.add
[/q22]
```

The verifier accepts a two-column prediction file with the exact header
`id<TAB>model_request`. It checks the common evaluation hash, independently
recomputes every task's request and arithmetic artifact, requires one
prediction for every frozen ID, and reports integer
`operation_exact_rate_ppm`.

```bash
cargo run -p nsrl-eval -- q22-encode \
  --manifest benchmarks/q22-shared-task-v1/manifest.tsv \
  --dataset /path/to/quantity-request.train.jsonl \
  --out /tmp/q22-solomon-train.txt

cargo run -p nsrl-eval -- q22-check \
  --manifest benchmarks/q22-shared-task-v1/manifest.tsv \
  --eval /path/to/quantity-request.promotion.tsv \
  --predictions /path/to/predictions.tsv
```

## Prospective Solomon run

The registered Solomon experiment uses the same sparse integer class-head
lineage as the Solomon v2 retrieval spine. It fixes an 8,192-feature signed
integer perceptron, four epochs, and seeds 1, 2, and 3. The command below is the
only supported full run:

```bash
node scripts/run-q22-solomon-prospective.mjs \
  --train-dataset /path/to/quantity-request.train.jsonl \
  --eval /path/to/quantity-request.promotion.tsv \
  --out-dir /path/to/new-empty-output-directory
```

The runner trains and hashes all three models before it reads the evaluation
file. It then creates a two-column blinded ID/input file. The proposer never
receives the gold operation, request, artifact, or summary columns. Training or
model selection after that point is forbidden.

The family result is `go` only when every seed reaches 950,000
`operation_exact_rate_ppm` and all 500 predictions are identical across all
three seeds. A completed miss is a valid `no_go`, not an execution failure.
The local CPU budget is 60 seconds per seed, 180 seconds total, with no paid
compute and no network.

The frozen contract is
`benchmarks/q22-solomon-prospective-v1/contract.json`. Its checked-in state is
`preregistered_not_run`; result artifacts belong in a later evidence change so
the outcome cannot rewrite the design.
