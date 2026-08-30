# Solomon Q22 prospective result

The preregistered three-seed run completed with outcome `go` on 2026-08-30.
Each seed selected all 500 frozen promotion operations exactly, for
`operation_exact_rate_ppm = 1000000`. All 500 predictions also agreed across
seeds, so the secondary Solomon family gate passed at `1000000` ppm.

The models were trained and hashed before the evaluation was opened. The
immutable preregistration contract remains unchanged and has SHA-256
`b3b2e2d2648802ac99395c0d1110207409f240cc5e4b83d8c89186c36238ccc0`.
The run used source commit `5de132b52b361b8b79638969bb0f0ab6c04111a6`.

Run `node scripts/check-q22-solomon-evidence.mjs` to verify every published
artifact, the model freeze boundary, the three exact checks, and cross-seed
agreement without retraining or reopening the gold evaluation file.
