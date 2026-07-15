# Integer Transformer Successor v2

`integer-transformer-successor-v2` is the manifest-bound follow-up to the
system-level v1 proof. It asks whether the parametric integer Transformer, with
all fitted or routed assistance disabled, has lower held-out next-byte negative
log-likelihood than four declared baselines.

## Frozen contract

- Dataset hash: `0x8fe7b86378f81951`
- Evaluation geometry: context 64, stride 1, exactly 5,896 targets
- Candidate: the v1 promoted artifact evaluated in `transformer-only` mode
- Assistance: `suffix-memory=off,retrieval=off,routing-oracle=off`
- Baselines: uniform, retrieval, byte n-gram, and a trained float32 Transformer
- Objective: canonical integer base-2 softmax NLL, summed in millibits
- Zero-weight target floor: 32,000 millibits
- Promotion: candidate NLL must be strictly lower than every baseline NLL
- Mistakes: reported as a secondary diagnostic, not a promotion gate

The checked manifest is
`benchmarks/integer-transformer-successor-v2/manifest.tsv`. Loading it verifies
the train/eval bytes, exact target count, candidate artifact hash, source
candidate hash, ablated model hash, matrix runner hash, assistance profile,
float model hash, and float runner hash. Every matrix row repeats the frozen
dataset and candidate/model/runner/assistance bindings. A syntactically valid
five-row TSV with a substituted candidate, model, runner, dataset, target
count, or assistance profile is rejected.

The frozen identities are:

| Binding | Hash |
| --- | --- |
| Candidate artifact bytes | `0xfe5db873a43b9b52` |
| Source candidate model | `0x6ffd37de48a3121b` |
| Transformer-only model | `0x391adc5e1d1a8713` |
| Matrix runner | `0xec2cd3ffa663d5ef` |
| Assistance profile | `0x83e30b9ff0fe6c77` |
| Float Transformer model | `0x06ed20b6ac52ea82` |
| Float Transformer runner | `0xd0b37c9eb3275c5b` |

## Real float Transformer baseline

The float row is not the v1 interpolated n-gram reference. It is a
deterministic NumPy float32 causal Transformer with learned token and position
embeddings, trained Q/K/V/O projections, scaled dot-product softmax attention,
two residual paths, a trained 32→64→32 feed-forward block, and a trained output
head. The last causal query attends to the full 64-byte context. Adam trains all
ten tensors for 384 updates over 1,024 deterministically spread windows and 12
epochs. Every tensor records nonzero bitwise movement, and the observed batch
NLL falls from 5.6493 to 2.2353 nats.

Evaluation converts its natural-log float logits to base-2 Q8 logits, then uses
the same canonical integer NLL scorer as every other row. The frozen model is
retrained and compared byte-for-byte by the end-to-end runner.

## Result: falsified

The five-system trial is valid and the candidate does not pass.

| System | Mistakes | Total NLL (millibits) | Mean NLL | Zero-weight targets |
| --- | ---: | ---: | ---: | ---: |
| transformer-only | 5,094 | 115,010,055 | 19,506.454 | 2,916 |
| uniform | 5,896 | 47,168,000 | 8,000.000 | 0 |
| retrieval | 2,505 | 38,293,936 | 6,494.901 | 0 |
| byte-ngram | 2,574 | 36,952,920 | 6,267.456 | 0 |
| float-transformer | 4,497 | 23,216,345 | 3,937.643 | 9 |

The candidate loses to the weakest NLL baseline, uniform, by 67,842,055
millibits and to the strongest baseline, the float Transformer, by 91,793,710.
Its exponent approximation assigns zero weight to the held-out target on 2,916
windows (49.46%), which dominates the declared NLL floor. The result therefore
falsifies successor-v2; it does not authorize promotion or a headline claim
that the unassisted integer Transformer beats the baselines.

This conclusion is narrower than the v1 system result. It does not invalidate
the deterministic integer runtime, exact replay, or the measured combined
Transformer-plus-suffix-memory artifact. It shows that this frozen parametric
model, after assistance removal, is not a successful next-token model under the
declared likelihood objective.

## Reproduction and checks

Run the complete candidate ablation, float training, five-system matrix, and
gate:

```bash
scripts/run-integer-transformer-successor-v2.sh \
  data/experiments/integer-transformer-successor-v2/replay
```

Exit code 0 means a valid passing trial, 1 means a valid falsification, and 2
means an invalid or mismatched artifact. The frozen result intentionally exits
1.

Inspect the contract, manifest, and result directly:

```bash
cargo run -p nsrl-eval -- successor-contract
cargo run -p nsrl-eval -- successor-manifest \
  --manifest benchmarks/integer-transformer-successor-v2/manifest.tsv
cargo run -p nsrl-eval -- successor-check \
  --manifest benchmarks/integer-transformer-successor-v2/manifest.tsv \
  --results benchmarks/integer-transformer-successor-v2/results.tsv
```

The final command also intentionally exits 1 while printing `"passed":false`.
`node scripts/check-integer-transformer-successor-v2.mjs` treats that frozen
falsification as valid evidence and detects identity or result drift.
