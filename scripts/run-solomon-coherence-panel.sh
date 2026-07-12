#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run a small deterministic Solomon text-to-bitmap coherence panel.

Common knobs:
  NSRL_RUN_NAME=coherence-panel-signature-20260623
  NSRL_RUN_ROOT=data/processed/key-solomon-goetia-latent-v1
  NSRL_SOLOMON_LATENT_MODEL=data/aws-runs/<run>/latent/model.nsrllat
  NSRL_SAMPLES=4
  NSRL_CANDIDATE_MULTIPLIER=8
  NSRL_DIVERSITY_WEIGHT=1
  NSRL_PASSES=2
  NSRL_BASELINE_MANIFEST=data/processed/key-solomon-goetia-latent-v1/coherence-panel-001/manifest.tsv

Override prompts with newline-separated specs:
  NSRL_PROMPT_SPECS=$'slug|panel label|prompt\n...'

Artifacts:
  data/processed/key-solomon-goetia-latent-v1/<run-name>/
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${NSRL_RUN_ROOT:-data/processed/key-solomon-goetia-latent-v1}"
run_name="${NSRL_RUN_NAME:-coherence-panel-signature-${timestamp}}"
run_dir="${run_root}/${run_name}"
model="${NSRL_SOLOMON_MODEL:-data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch}"
latent_model="${NSRL_SOLOMON_LATENT_MODEL:-}"
samples="${NSRL_SAMPLES:-4}"
candidate_multiplier="${NSRL_CANDIDATE_MULTIPLIER:-8}"
diversity_weight="${NSRL_DIVERSITY_WEIGHT:-1}"
passes="${NSRL_PASSES:-2}"
preview_columns="${NSRL_PREVIEW_COLUMNS:-4}"
seed_prefix="${NSRL_SEED_PREFIX:-solomon-coherence-panel-v1}"
baseline_manifest="${NSRL_BASELINE_MANIFEST:-}"

if [[ ! -f "$model" ]]; then
  echo "missing text-conditioned Solomon sampler model: $model" >&2
  exit 2
fi
if [[ -z "$latent_model" ]]; then
  echo "NSRL_SOLOMON_LATENT_MODEL is required; use a latent model from the prior smoke" >&2
  exit 2
fi
if [[ ! -f "$latent_model" ]]; then
  echo "missing Solomon latent model: $latent_model" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to collect sampler traces" >&2
  exit 2
fi

rm -rf "$run_dir"
mkdir -p "$run_dir"

default_prompt_specs=(
  "crocell|Crocell|Crocell"
  "stolas|Stolas|Stolas"
  "bael|Bael|Bael"
  "hidden-geometry-waters|hidden geometry and rushing waters|hidden geometry and rushing waters"
  "astronomy-herbs-teacher|astronomy and herbs teacher|astronomy and herbs teacher"
)

prompt_specs=()
if [[ -n "${NSRL_PROMPT_SPECS:-}" ]]; then
  while IFS= read -r prompt_spec; do
    [[ -z "$prompt_spec" ]] && continue
    prompt_specs+=("$prompt_spec")
  done <<< "$NSRL_PROMPT_SPECS"
else
  prompt_specs=("${default_prompt_specs[@]}")
fi

cargo build --release -q -p nsrl-train --bin nsrl-bitmap-sample

manifest="${run_dir}/manifest.tsv"
printf 'slug\tlabel\tprompt\tinit\tlatent_target\tselected_mean_wash_penalty_q8\tout_dir\tpng\tsampler_target_source\tsampler_target_number\tsampler_target_name\tsampler_target_score\tsampler_target_latent_score\tsampler_target_lexical_score\tselected_min_text_distance\tselected_mean_text_distance_q8\tselected_mean_score_q8\tlatent_dim\tlatent_text_features\n' > "$manifest"

pngs=()
labels=()
for prompt_spec in "${prompt_specs[@]}"; do
  IFS='|' read -r slug label prompt <<< "$prompt_spec"
  if [[ -z "$slug" || -z "$label" || -z "$prompt" ]]; then
    echo "invalid prompt spec: $prompt_spec" >&2
    exit 2
  fi
  out_dir="${run_dir}/${slug}"
  mkdir -p "$out_dir"
  sample_cmd=(
    target/release/nsrl-bitmap-sample
    --model "$model" \
    --latent-model "$latent_model" \
    --prompt "$prompt" \
    --out-dir "$out_dir" \
    --samples "$samples" \
    --candidate-multiplier "$candidate_multiplier" \
    --diversity-weight "$diversity_weight" \
    --passes "$passes" \
    --preview-columns "$preview_columns" \
    --seed "${seed_prefix}-${slug}" \
    --init noise
  )
  "${sample_cmd[@]}"
  target_source="$(jq -r '.latent_target_source // ""' "${out_dir}/trace.json")"
  if [[ "$target_source" != "decoded-latent" ]]; then
    echo "${out_dir}: sampler target source was '${target_source}', expected decoded-latent" >&2
    exit 3
  fi

  png="${out_dir}/samples.png"
  if command -v magick >/dev/null 2>&1; then
    magick "${out_dir}/samples.pgm" "$png"
  else
    png="${out_dir}/samples.pgm"
  fi

  jq -r \
    --arg slug "$slug" \
    --arg label "$label" \
    --arg prompt "$prompt" \
    --arg init "noise" \
    --arg latent_target "decoded-latent" \
    --arg out_dir "$out_dir" \
    --arg png "$png" \
    '[
      $slug,
      $label,
      $prompt,
      $init,
      $latent_target,
      .selected_mean_wash_penalty_q8,
      $out_dir,
      $png,
      .latent_target_source,
      .latent_target_number,
      .latent_target_name,
      .latent_target_score,
      .latent_target_latent_score,
      .latent_target_lexical_score,
      .selected_min_text_distance,
      .selected_mean_text_distance_q8,
      .selected_mean_score_q8,
      .latent_dim,
      .latent_text_features
    ] | @tsv' \
    "${out_dir}/trace.json" >> "$manifest"
  pngs+=("$png")
  labels+=("$label")
done

panel="${run_dir}/panel.png"
if command -v magick >/dev/null 2>&1; then
  montage_args=(-background white -fill black -pointsize 16)
  for index in "${!pngs[@]}"; do
    montage_args+=(-label "${labels[$index]}" "${pngs[$index]}")
  done
  magick montage "${montage_args[@]}" -tile 1x -geometry +0+14 "$panel" >/dev/null 2>&1 || true
fi

if [[ -n "$baseline_manifest" && -f "$baseline_manifest" ]]; then
  comparison="${run_dir}/comparison.tsv"
  awk '
    BEGIN { FS = OFS = "\t" }
    FNR == NR {
      if (FNR == 1) {
        for (i = 1; i <= NF; i++) base_col[$i] = i
        next
      }
      slug = $(base_col["slug"])
      base_min[slug] = $(base_col["selected_min_text_distance"])
      base_mean[slug] = $(base_col["selected_mean_text_distance_q8"])
      next
    }
    FNR == 1 {
      for (i = 1; i <= NF; i++) col[$i] = i
      print "slug", "baseline_min", "candidate_min", "delta_min", "baseline_mean_q8", "candidate_mean_q8", "delta_mean_q8"
      next
    }
    {
      slug = $(col["slug"])
      candidate_min = $(col["selected_min_text_distance"])
      candidate_mean = $(col["selected_mean_text_distance_q8"])
      print slug, base_min[slug], candidate_min, candidate_min - base_min[slug], base_mean[slug], candidate_mean, candidate_mean - base_mean[slug]
    }
  ' "$baseline_manifest" "$manifest" > "$comparison"
fi

echo "manifest: $manifest"
if [[ -f "$panel" ]]; then
  echo "panel: $panel"
fi
if [[ -f "${run_dir}/comparison.tsv" ]]; then
  echo "comparison: ${run_dir}/comparison.tsv"
fi
