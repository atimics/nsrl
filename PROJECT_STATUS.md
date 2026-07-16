# Project Status

This repo's canonical status surface is:

```bash
node scripts/nsrl-status.mjs
```

Use JSON when another tool needs to consume the same truth surface:

```bash
node scripts/nsrl-status.mjs --json
```

Run fresh local checks when you want evidence instead of artifact inspection:

```bash
node scripts/nsrl-status.mjs --run-hygiene
node scripts/nsrl-status.mjs --refresh-fast-diagnostic
```

## Current Read

As of 2026-07-15, NSRL is not release-ready. The integer runtime and research
artifacts exist, but the Solomon product proof is incomplete.

The active substrate promotion contract is
`integer-transformer-successor-v2`. The original complete replay is preserved
as a frozen falsification; the active deterministic repair passes promotion.
On the same 5,896 targets and dataset hash `0x8fe7b86378f81951`, the physically
unassisted transformer scores 25,347,655 canonical NLL millibits with zero
zero-probability windows, versus 47,168,000 uniform, 38,271,425 retrieval,
38,025,720 byte n-gram, and 40,847,697 for the genuine trained float
transformer. `node scripts/run-integer-transformer-successor-v2.mjs --check`
rebuilds and byte-compares the training trace, candidate, complete matrix, and
evidence; the frozen v1
transformer-plus-suffix-memory result remains separately replayable. Solomon
and literary results are experiment evidence rather than alternate headline
criteria.

Solomon Council v0 is **shadow-ready**. All six faculty seals verify; the judge
exercises selection, evidence request, user question, and abstention; circle
overruns and unauthorized evidence fail closed; dissent is retained; and both
the initial wisdom receipt and its outcome/revision chain replay byte-for-byte.
The bounded solomonic experiment records eight favorable fired passages and
`-52381` Q32 signed regret. Its original publication is supported under an
explicit conditional unsafe-intensity null, but MJ-20 supplies an exact
exchangeable counterexample to deriving that null from marginal coverage. The
non-crossing e-process therefore does not establish sequential safety. The
checked finite-horizon replacement requires 119 calibration source panels per
family and is not execution-ready. The eight-dimension same-model wisdom
evaluation also remains unmeasured, so council promotion is not authorized.

The project headline is `NSRL-MME v0`, a model-native multimodal LLM eval
defined in `docs/multimodal-llm-eval.md`. The current local score is **371 per
mille**, below the 700 target. The quality report and generated-output integrity
gates remain red, so this is measured diagnostic evidence rather than a passing
headline result.

Known facts from the status command:

- the working tree is dirty,
- the production Q23/Q47-Newton path now carries exact-replayable integer signal
  through target probabilities, but its first materialized output boundary
  regresses dev; the document-blocked rescue-stratified alignment audit fails
  on the proposal surface while its output-head sample aligns, and the v3
  source-matched control shows that removing all nonzero rescue changes sampled
  trunk magnitudes but no signs or descent decisions; the exact rank-two cube
  then found a post-hoc transfer-only trunk/head synergy, but a frozen
  64-document confirmation reversed the aggregate conditional sign and favored
  head-only on 11 non-tied documents versus 7 for the joint move, with 46 ties.
  That move-family repair is rejected; the next gate is a new prospectively
  defined, stability-aware proposal rule rather than rescue removal, more bits,
  fixed-mass tuning, or shift search,
- the complete proposal-only six-atom cube is now measured. Its Q32 cubic tail
  is tiny and selects the exact aggregate minimizer, but exact support has
  maximal induced width, Q20/Q32 coefficient support and signs disagree
  materially, and the proposal block contains only one source cluster. A
  derived cubic Walsh surrogate has zero canonical gap on the aggregate and all
  document cubes in both grids, but still selects the falsified all-atom move.
  This is descriptive structure discovery, not optimizer or scaling
  authorization,
- the frozen Ising confirmation on documents 136--199 passes all three
  within-source document endpoints after exact Holm correction: masks `59` and
  `61` beat baseline, and the probe router improves over mask `47` on all 17
  rerouted documents. Only atom 5's one-body field replicates as a stable
  low-order parameter; no pair coupling does, and the pairwise/Gibbs masks
  re-estimated on confirmation change. The leading mechanism is therefore a
  probe-routed conditional exchange of atom 4 for atom 2, not a global Ising
  MAP. That confirmation alone does not identify cross-source transfer;
  documents 200--212 remain sealed,
- the prospective M3 source-panel experiment now supports that conditional
  exchange on its frozen 71-source distinct-author English Project Gutenberg
  frame. The disjoint split uses 16 fitting, 39 calibration, and 16 untouched
  evaluation sources. Its 95% correction is 4,326 Q32; all 16 evaluation panels
  are covered, 5 fire, and none is unsafe: coverage is 16/16, firing is 5/16,
  and marginal unsafe-action rate is 0/16. Signed regret relative to always
  abstaining is -40,769 Q32 in aggregate (-40,769/16 Q32 per evaluation panel),
  with zero positive regret. A checked publication layer reports only
  `supported`, `falsified`, or `inconclusive` and maps a vacuous envelope to
  `inconclusive`; this result is `supported`. It is bounded to one sampled
  passage and two adjacent targets per source and does not authorize an
  optimizer change or paid scaling,
- the prospective M4 extension is complete but does not promote an overall
  multi-family certificate. It freezes 104 whole-publication sources across
  Federal Register, new Gutenberg, RFC, and open-access science, with 3
  fitting, 19 family-calibration, and 4 untouched evaluation sources per
  family, plus four nonoverlapping passages per source. Family rank-19
  corrections are `2,326`, `2,141`, `4,307`, and `4,272` Q32. Evaluation
  coverage is 14/16 because two Gutenberg panels miss their envelope; exactly
  two failures are the preregistered gray zone, so the checked verdict is
  `coverage_inconclusive_no_promotion`. Twelve passages fire across Federal
  Register, Gutenberg, and RFC, all twelve are favorable, none is unsafe, and
  net held-out improvement is `63,541` Q32. Federal Register and RFC pass their
  frozen family gates, Gutenberg is withheld by coverage, and science
  abstains. Optimizer changes and paid scaling remain unauthorized, and
  documents 200--212 remain sealed,
- the headline multimodal LLM eval is measured but failing at 371 per mille,
- the checked-in attention artifacts are smoke-scale, not promoted-profile,
- local quality-report and objective-coverage artifacts exist,
- no `release-proof.json` or completed Solomon `pipeline-complete.json` is
  present under `data/`,
- raw/free-running attention text is still diagnostic-only,
- coherent Solomon text currently comes from prompted or memory-assisted paths.

## Headline Eval

The number we are chasing is:

```text
NSRL-MME v0 headline_score_per_mille
```

It is the floor across model-native multimodal task families:

- text prompt -> symbolic image plan,
- seal image -> identity, attributes, and source text,
- text plus seal -> grounded explanation and match behavior,
- prompt/name -> identity and source binding,
- match/no-match hard negatives.

The first target is `>= 700` per mille with the required source-grounding,
held-out generated-output integrity, green quality report, and objective
coverage gates. Replay and sampler metrics remain useful debugging evidence,
but they are not the headline.

## LLM Path

The immediate LLM path is not a converted HuggingFace/Llama model. NSRL models
must be born into the integer/base-2 attention contract.

The practical sequence is:

1. Keep `node scripts/nsrl-status.mjs` green enough that project state is obvious.
2. Produce `data/processed/nsrl-mme-v0.json` with:

   ```bash
   node scripts/check-nsrl-mme-v0.mjs --out data/processed/nsrl-mme-v0.json
   ```

3. Feed the scorer a measured `quality-report.json` with confidence-trace
   evidence plus objective coverage.
4. Repair the failing product-proof/self-test surface.
5. Run the Graviton product path:

   ```bash
   NSRL_S3_URI=s3://BUCKET/PREFIX scripts/aws/run-solomon-end-to-end.sh
   ```

6. Prove the completed run:

   ```bash
   scripts/aws/prove-solomon-product-run.sh \
     --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME \
     --launch-dir data/aws-launches/RUN_NAME \
     --require-launch-dir
   ```

7. Promote the first narrow NSRL-born `NSRLLMM1` expert before scaling outward
   into routed expert swarms.
