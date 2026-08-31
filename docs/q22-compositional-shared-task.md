# Q22 compositional routing adapter

This adapter is the shortcut-resistant successor to the first Zero–Solomon
Q22 bridge. Every model input has the same four-token prefix. A balanced
background clause describes the wrong operation, while the later actual-case
clause describes the requested operation. Training and promotion use separate
sentence templates.

Solomon keeps the prior sparse integer perceptron geometry: 8,192 hashed
features and no floating-point path. The common prefix is therefore present in
the inherited feature map but cannot separate the five balanced classes. The
balanced distractors also prevent the model from succeeding by following the
first operation cue it sees.

The adapter checks the exact Zero data hashes, validates the canonical request
and arithmetic artifact independently, and encodes training records as:

```text
[q22-compositional]
input: Route this quantity case. ...
operation: quantity.add
[/q22-compositional]
```

The evaluation firewall trains and hashes every model before creating a
two-column `id`/`input` promotion surface. The proposer never receives the gold
operation, request, artifact, or summary columns.

The prospective contract is
`benchmarks/q22-compositional-solomon-prospective-v1/contract.json`. It freezes
seeds 1, 2, and 3, four epochs, all source hashes, and local CPU-only limits.
The family result is `go` only if every seed scores at least 90% overall,
every operation scores at least 80% in every seed, and at least 95% of cases
receive the same prediction from all three seeds. The prefix-only baseline is
exactly 20%.

After ilXyr pins the merged Solomon commit, the one allowed run is:

```bash
node scripts/run-q22-compositional-solomon-prospective.mjs \
  --train-dataset /path/to/quantity-composition.train.jsonl \
  --eval /path/to/quantity-composition.promotion.tsv \
  --out-dir /path/to/new-empty-output-directory
```

The allowed claim remains narrow. A pass shows held-out paraphrase operation
routing under a balanced distractor. It does not show arithmetic execution,
open-ended language quality, or general reasoning.
