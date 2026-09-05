# Fixed-answer probability controls

The current integer proof reports distance from a one-hot target as
`probability_error_q15`. A confidence-only change can improve that score while
keeping every chosen byte fixed. The new evaluator option exposes this effect
beside an exact Brier score and a count of zero-probability targets.

## Existing result audit

The source is the promoted `calibrated-v2-suffix-memory` candidate and its
committed component ablation at `df244eb414e68ad97c7bcce228b6129522e9a969`.
The artifact SHA-256 is
`37acae6a4f763182730c76f762c351eda5bb37d6d197358c252733b1f08dca10`.

The old score sums absolute differences from the target vector, whose selected
entry is 32,767. A control assigns all 32,767 units to the model's existing
chosen byte. Its error is exactly `2 * 32767 * mistakes`. This gives a complete
counterfactual from the recorded mistake counts:

| Component | Targets | Mistakes, held fixed | Recorded error | Forced-confidence error |
| --- | ---: | ---: | ---: | ---: |
| Combined | 5,896 | 2,482 | 260,536,589 | 162,655,388 |
| Transformer only | 5,896 | 5,094 | 337,139,495 | 333,830,196 |
| Suffix memory only | 5,896 | 2,482 | 384,884,984 | 162,655,388 |

The combined model's error falls by 37.5691% through confidence alone. This is
an analytic result from existing records. The original promotion record keeps
its status and metrics. The counterfactual highlights a measurement failure:
this score alone is insufficient evidence for a claim about better probability
estimates. Forced confidence also assigns zero probability to every wrong
target; such events carry infinite log loss.

## Executable control

`nsrl-mini-transformer-eval --probability-controls-out PATH` writes each chosen
byte, target, probability vector, original score, exact normalized Brier score,
and zero-probability event. The target enters only scoring. The three arms are:

- Native probabilities from the model's final logits. Their original score
  must equal the score returned by the native forward pass.
- Point mass on the existing chosen byte.
- Fixed smoothing: 29,491 units on that byte, with the remaining 3,276 units
  spread across the other 255 bytes in byte order.

The Brier score is the sum of squared distances from the one-hot target after
dividing each probability by the vector's actual mass. Both numerator and
denominator use integers. This accounts for rounding in native Q15 softmax.
The point-mass and smoothed arms each have exactly 32,767 units.

## Known-window smoke

The smoke uses the evaluator's 16 evenly spread windows from the existing
5,960-byte evaluation file. All three arms make the same four mistakes.

| Arm | Original error | Mean Brier score | Zero-probability targets |
| --- | ---: | ---: | ---: |
| Native | 666,174 | 0.702257 | 2 |
| Point mass | 262,136 | 0.500000 | 4 |
| Fixed smoothing | 340,656 | 0.459847 | 0 |

These known windows serve as an engineering check. Fresh-data evidence belongs
to a separately frozen comparison. Full-set native Brier scores require a new
evaluation; the aggregate records provide only the point-mass calculation.

The checker rebuilds each score with Python fractions and verifies all 48 rows,
window identities, fixed answers, model identity, and historical file bindings.
Rust tests cover all chosen bytes, mass conservation, invalid vectors, and a
four-target example where forced confidence improves the old score while
worsening Brier. Mutated row and incomplete-roster checks exercise rejection.

Reproduce the bounded smoke:

```bash
cargo test -p nsrl-train --bin nsrl-mini-transformer-eval --features mini-heads-8,mini-calibrated
cargo build --release -p nsrl-train --bin nsrl-mini-transformer-eval --features mini-heads-8,mini-calibrated
python3 scripts/check-probability-controls.py
```

The [frozen smoke](../benchmarks/integer-transformer-proof-v1/probability-controls-smoke.json)
binds source files, model bytes, input bytes, raw-output digests, and exact scores.
The checker prints a temporary directory containing native output and errors.

## Development failures retained

The first checker assumed consecutive windows. Native output showed that
`--max-windows 16` spreads windows across the file. The checker now verifies the
documented capped-window rule and rejects a changed window identity.

A strict local Clippy run with `-D warnings` stopped on existing
`manual_is_multiple_of` warnings in `nsrl-eval/src/q22_compositional.rs`.
The repository's configured Clippy command and the focused checks are reported
separately in the PR.

## Next comparison

Freeze the same three probability controls for the combined, transformer-only,
and suffix-memory-only models. Report mistakes, normalized Brier score, zero
events, model bytes, and complete inference cost on the same fresh corpus.
This separates the contribution of learned probability estimates from suffix
memory's contribution to chosen answers.
