# NSRL-MME v0: Headline Multimodal LLM Eval

NSRL's headline number is `NSRL-MME v0`, schema
`nsrl.multimodal_llm_eval.v0`.

The score is the minimum per-mille score across model-native multimodal task
families. This is intentionally a floor metric: one strong replay table or one
coherent prompted sample cannot hide a missing direction.

## Headline Score

`headline_score_per_mille = min(component_score_per_mille)`

Required scored components:

- `text_prompt_to_image_plan`: text prompt -> symbolic image plan.
- `seal_image_to_text`: seal image -> identity, attributes, and source text.
- `text_and_seal_to_explanation`: text plus seal -> grounded explanation and
  match behavior.
- `identity_source_binding`: prompt/name -> identity and source binding.
- `hard_negative_match`: match/no-match and wrong-image/wrong-prompt hard
  negatives.

Each scored component must have at least 72 held-out or product-scope rows.
The initial target is `>= 700` per mille on the headline floor.

## Required Gates

These do not increase the score, but the headline eval cannot pass without
them:

- source-grounded text/image evidence is present and green,
- held-out generated output integrity is green,
- the quality report is green,
- objective coverage exists for the run.

## What Does Not Count As The Headline

These are diagnostics, not the project's headline number:

- browser sampler probes,
- raw sample galleries,
- memory-assisted sample coherence,
- corpus replay top-1/top-5,
- latent prior retrieval,
- bitmap denoiser retrieval.

They are still useful for debugging and product demos, but they do not answer
the question "do we have a multimodal LLM?"

## Current Status

At the time this contract was introduced, the repo does not contain a measured
`NSRL-MME v0` result under `data/`. The public Pages results are therefore
diagnostic, not a headline multimodal LLM eval.

Use the status surface for the current answer:

```bash
node scripts/nsrl-status.mjs
node scripts/nsrl-status.mjs --json
```
