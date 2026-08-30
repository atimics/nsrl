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

This closes the encoding and verifier infrastructure gap. It does not train a
Solomon model and does not claim replication success.
