# Solomon Q22 compositional routing result

The preregistered three-seed run completed with outcome `no_go` on 2026-08-31.
The worst seed reached `425000` exact-rate ppm against the frozen `900000`
gate. The worst seed-class pair reached `0` ppm against `800000`, and
all-seed agreement reached `531000` ppm against `950000`.

The three seed rates were `425000`, `430000`, and `533000` ppm. The minimum
result remained `225000` ppm above the exactly balanced prefix-only baseline,
so the models learned some operation evidence. They did not transfer reliably
across the 20 held-out sentence-template families. Seed 1 scored zero on
`quantity.add`; seed 3 was strongest overall but still missed four of the five
per-class gates.

The models were trained and hashed before evaluation opened. The immutable
preregistration contract remains unchanged at SHA-256
`f5fdd260ae7e7ef2fdab85bb4aaf0aaebb2716015b3e8026a677aaf8fa8d0ae4`.
The run used NSRL merge commit
`fa60312f93ec859e42a57f383f5972b8df4cf807` and Zero merge commit
`8deeb138e4c01ba5b64bb22e5125d5c823c37f78`.

Run `node scripts/check-q22-compositional-solomon-evidence.mjs` to verify all
published artifacts, the model-freeze boundary, each exact check, and the
cross-seed agreement count without retraining or reopening gold labels.
