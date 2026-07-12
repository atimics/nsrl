#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run a Solomon text-to-layout prior smoke on a native Linux/Graviton runner.

Common knobs:
  NSRL_RUN_NAME=solomon-prior-smoke-001
  NSRL_RUN_ROOT=/mnt/nsrl/aws-runs
  NSRL_S3_URI=s3://bucket/prefix
  NSRL_SOLOMON_LATENT_EPOCHS=16
  NSRL_SOLOMON_LATENT_DIM=64
  NSRL_SOLOMON_TEXT_FEATURES=8192
  NSRL_SOLOMON_SAMPLES=4
  NSRL_SOLOMON_SEED_VARIANTS=a,b,c
  NSRL_SOLOMON_SAMPLER_WORKERS=4
  NSRL_SOLOMON_SYNC_AFTER_SAMPLE=1
  NSRL_SOLOMON_TEXT_WEIGHT=96
  NSRL_SOLOMON_MAX_INTRA_PROMPT_DISTANCE=8192
  NSRL_SOLOMON_MAX_TARGET_DISTANCE=24576
  NSRL_SOLOMON_MIN_INTER_CLASS_DISTANCE=1024
  NSRL_SOLOMON_MIN_TARGET_INK_CELLS=8
  NSRL_SOLOMON_MAX_TARGET_INK_CELLS=224

Artifacts:
  <run-dir>/latent/model.nsrllat
  <run-dir>/latent/trace.json
  <run-dir>/eval-ledger.jsonl
  <run-dir>/prior-gate/decoded-layout-panel.png
  <run-dir>/samples/<prompt>/trace.json
  <run-dir>/manifest.tsv
  <run-dir>/smoke-check.json
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${NSRL_RUN_NAME:-solomon-prior-smoke-${timestamp}}"
run_root="${NSRL_RUN_ROOT:-data/aws-runs}"
run_dir="${run_root}/${run_name}"
latent_dir="${run_dir}/latent"
gate_dir="${run_dir}/prior-gate"
sample_dir="${run_dir}/samples"
mkdir -p "$latent_dir" "$gate_dir" "$sample_dir"
target_dir="${CARGO_TARGET_DIR:-target}"
release_bin_dir="${target_dir%/}/release"

text_index="${NSRL_SOLOMON_TEXT_INDEX:-data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv}"
prompts="${NSRL_SOLOMON_PROMPTS:-data/processed/key-solomon-goetia-latent-v1/scaling-curve/prompts-1425.jsonl}"
gold="${NSRL_SOLOMON_GOLD:-data/processed/key-solomon-goetia-latent-v1/gold.tsv}"
denoise_model="${NSRL_SOLOMON_DENOISE_MODEL:-data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch}"
latent_model="${latent_dir}/model.nsrllat"
latent_epochs="${NSRL_SOLOMON_LATENT_EPOCHS:-16}"
latent_dim="${NSRL_SOLOMON_LATENT_DIM:-64}"
text_features="${NSRL_SOLOMON_TEXT_FEATURES:-8192}"
samples="${NSRL_SOLOMON_SAMPLES:-4}"
candidate_multiplier="${NSRL_SOLOMON_CANDIDATE_MULTIPLIER:-8}"
diversity_weight="${NSRL_SOLOMON_DIVERSITY_WEIGHT:-1}"
passes="${NSRL_SOLOMON_PASSES:-2}"
sampler_workers="${NSRL_SOLOMON_SAMPLER_WORKERS:-}"
sync_after_sample="${NSRL_SOLOMON_SYNC_AFTER_SAMPLE:-1}"
text_weight="${NSRL_SOLOMON_TEXT_WEIGHT:-96}"
seed_prefix="${NSRL_SOLOMON_SEED_PREFIX:-solomon-prior-smoke-v1}"
seed_variants="${NSRL_SOLOMON_SEED_VARIANTS:-a,b,c}"
max_intra_prompt_distance="${NSRL_SOLOMON_MAX_INTRA_PROMPT_DISTANCE:-8192}"
max_target_distance="${NSRL_SOLOMON_MAX_TARGET_DISTANCE:-24576}"
min_inter_class_distance="${NSRL_SOLOMON_MIN_INTER_CLASS_DISTANCE:-1024}"
min_target_ink_cells="${NSRL_SOLOMON_MIN_TARGET_INK_CELLS:-8}"
max_target_ink_cells="${NSRL_SOLOMON_MAX_TARGET_INK_CELLS:-224}"
min_eval_class_top1="${NSRL_SOLOMON_MIN_EVAL_CLASS_TOP1:-1}"
# The sampler's only latent target source is the decoded latent path (the
# NSRLCLS class-head checkpoint extension had no writer and has been removed).
expected_target_source="${NSRL_SOLOMON_EXPECTED_TARGET_SOURCE:-decoded-latent}"
IFS=',' read -r -a seed_variant_list <<< "$seed_variants"
seed_variant_values=()
for seed_variant in "${seed_variant_list[@]}"; do
  seed_variant="${seed_variant//[[:space:]]/}"
  [[ -z "$seed_variant" ]] && continue
  seed_variant_values+=("$seed_variant")
done
seed_variant_list=("${seed_variant_values[@]}")
if [[ "${#seed_variant_list[@]}" -eq 0 ]]; then
  echo "NSRL_SOLOMON_SEED_VARIANTS must list at least one variant" >&2
  exit 2
fi

for required in "$text_index" "$prompts" "$gold" "$denoise_model"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

sync_artifacts() {
  if [[ -n "${NSRL_S3_URI:-}" ]]; then
    aws s3 sync "$run_dir" "${NSRL_S3_URI%/}/runs/${run_name}" --only-show-errors
  fi
}

echo "[0/6] checking Solomon denoiser model"
node scripts/check-solomon-denoiser-model.mjs --model "$denoise_model"

echo "[1/6] building Solomon prior binaries"
cargo build --release -p nsrl-train --no-default-features \
  --bin nsrl-solomon-latent-train \
  --bin nsrl-solomon-eval \
  --bin nsrl-bitmap-sample

echo "[2/6] training hard 16x16 class-separated latent prior"
"${release_bin_dir}/nsrl-solomon-latent-train" \
  --text-index "$text_index" \
  --prompts "$prompts" \
  --gold "$gold" \
  --out-dir "$latent_dir" \
  --model-out "$latent_model" \
  --epochs "$latent_epochs" \
  --latent-dim "$latent_dim" \
  --text-features "$text_features"
python3 - "$latent_dir/trace.json" <<'PY'
import json
import pathlib
import sys

trace_path = pathlib.Path(sys.argv[1])
trace = json.loads(trace_path.read_text())
class_count = int(trace.get("class_count", 0))
class_top1 = int(trace.get("class_top1_per_mille", 0))
text_decision_errors = int(trace.get("text_signature_decision_errors", 0))
image_decision_errors = int(trace.get("image_signature_decision_errors", 0))
if class_count <= 0:
    raise SystemExit(f"{trace_path}: missing trained class head")
print(
    "class_head: "
    f"classes={class_count} "
    f"train_class_top1_per_mille={class_top1} "
    f"text_decision_errors={text_decision_errors} "
    f"image_decision_errors={image_decision_errors}"
)
PY
sync_artifacts

echo "[3/6] evaluating held-out prior separation"
"${release_bin_dir}/nsrl-solomon-eval" \
  --text-index "$text_index" \
  --prompts "$prompts" \
  --gold "$gold" \
  --model "$latent_model" \
  --ledger "${run_dir}/eval-ledger.jsonl" \
  --partition-out "${run_dir}/partition.tsv"
sync_artifacts

echo "[4/6] running JS prior gate"
node scripts/run-solomon-prior-gate.mjs \
  --text-index "$text_index" \
  --prompts "$prompts" \
  --out-dir "$gate_dir" \
  --epochs 1 \
  --classifier centroid
sync_artifacts

manifest="${run_dir}/manifest.tsv"
printf 'prompt_slug\tseed_variant\tprompt\tout_dir\tpgm\tpng\tlatent_target_source\ttext_weight\tlatent_target_number\tlatent_target_name\tlatent_target_score\n' > "$manifest"

prompt_specs=(
  "crocell|Crocell"
  "stolas|Stolas"
  "bael|Bael"
  "hidden-geometry-waters|hidden geometry and rushing waters"
  "astronomy-herbs-teacher|astronomy and herbs teacher"
)

echo "[5/6] sampling fixed prompt panel through latent ${expected_target_source} path"
for prompt_spec in "${prompt_specs[@]}"; do
  IFS='|' read -r slug prompt <<< "$prompt_spec"
  for seed_variant in "${seed_variant_list[@]}"; do
    out_dir="${sample_dir}/${slug}-${seed_variant}"
    mkdir -p "$out_dir"
    sample_cmd=(
      "${release_bin_dir}/nsrl-bitmap-sample" \
      --model "$denoise_model" \
      --latent-model "$latent_model" \
      --prompt "$prompt" \
      --out-dir "$out_dir" \
      --samples "$samples" \
      --candidate-multiplier "$candidate_multiplier" \
      --diversity-weight "$diversity_weight" \
      --passes "$passes" \
      --preview-columns "$samples" \
      --seed "${seed_prefix}-${slug}-${seed_variant}" \
      --init noise \
      --text-weight "$text_weight"
    )
    if [[ -n "$sampler_workers" ]]; then
      sample_cmd+=(--workers "$sampler_workers")
    fi
    "${sample_cmd[@]}"
    png="${out_dir}/samples.png"
    if command -v magick >/dev/null 2>&1; then
      magick "${out_dir}/samples.pgm" "$png"
    else
      png=""
    fi
    python3 - "$slug" "$seed_variant" "$prompt" "$out_dir" "$png" "$expected_target_source" <<'PY' >> "$manifest"
import json
import pathlib
import sys

slug, seed_variant, prompt, out_dir, png, expected_target_source = sys.argv[1:]
trace = json.loads(pathlib.Path(out_dir, "trace.json").read_text())
if trace.get("latent_target_source") != expected_target_source:
    raise SystemExit(
        f"{out_dir}: sampler used {trace.get('latent_target_source')} "
        f"expected {expected_target_source}"
    )
print("\t".join([
    slug,
    seed_variant,
    prompt,
    out_dir,
    str(pathlib.Path(out_dir, "samples.pgm")),
    png,
    str(trace.get("latent_target_source", "")),
    str(trace.get("text_weight", "")),
    str(trace.get("latent_target_number", "")),
    str(trace.get("latent_target_name", "")),
    str(trace.get("latent_target_score", "")),
]))
PY
    if [[ "$sync_after_sample" != "0" ]]; then
      sync_artifacts
    fi
  done
done

echo "[6/6] verifying smoke artifacts"
node scripts/check-solomon-prior-smoke.mjs \
  --run-dir "$run_dir" \
  --min-seed-variants "${#seed_variant_list[@]}" \
  --max-intra-prompt-distance "$max_intra_prompt_distance" \
  --max-target-distance "$max_target_distance" \
  --min-inter-class-distance "$min_inter_class_distance" \
  --min-target-ink-cells "$min_target_ink_cells" \
  --max-target-ink-cells "$max_target_ink_cells" \
  --min-eval-class-top1 "$min_eval_class_top1" \
  --expected-target-source "$expected_target_source" \
  > "${run_dir}/smoke-check.json"

sync_artifacts
echo "run_dir: $run_dir"
echo "manifest: $manifest"
echo "latent_trace: ${latent_dir}/trace.json"
echo "prior_gate: ${gate_dir}/trace.json"
echo "smoke_check: ${run_dir}/smoke-check.json"
