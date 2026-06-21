#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

corpus="${CORPUS:-data/processed/wiki-bard-corpus.txt}"
model="${MODEL:-data/processed/visionary-lexeme-sweep-20260621T074142Z/small-v1024-w4096.nsrllm}"
vocab="${VOCAB:-data/processed/visionary-lexeme-sweep-20260621T074142Z/small-v1024-w4096.vocab.tsv}"
tokens="${TOKENS:-data/processed/visionary-lexeme-sweep-20260621T074142Z/small-v1024-w4096.tokens.u16}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_dir="${OUT_DIR:-data/processed/simplewiki-self-synthesis-${run_id}}"

prompt_count="${PROMPT_COUNT:-24}"
prompt_stride="${PROMPT_STRIDE:-160}"
prompt_min_chars="${PROMPT_MIN_CHARS:-96}"
prompt_max_chars="${PROMPT_MAX_CHARS:-240}"
max_new_tokens="${MAX_NEW_TOKENS:-160}"
top_k="${TOP_K:-8}"
sample_seed="${SAMPLE_SEED:-101}"
repeat_window="${REPEAT_WINDOW:-32}"
repeat_penalty_shift="${REPEAT_PENALTY_SHIFT:-1}"
max_repeat_run="${MAX_REPEAT_RUN:-3}"
corpus_prior_logit_shift="${CORPUS_PRIOR_LOGIT_SHIFT:-0}"

mkdir -p "$out_dir"

for path in "$corpus" "$model" "$vocab" "$tokens"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

prompts_path="$out_dir/prompts.txt"
synthetic_text="$out_dir/synthetic-corpus.txt"
trace_jsonl="$out_dir/generation.trace.jsonl"
manifest="$out_dir/manifest.json"

echo "run_id=$run_id"
echo "corpus=$corpus"
echo "model=$model"
echo "vocab=$vocab"
echo "tokens=$tokens"
echo "out_dir=$out_dir"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

awk \
  -v want="$prompt_count" \
  -v stride="$prompt_stride" \
  -v min_len="$prompt_min_chars" \
  -v max_len="$prompt_max_chars" \
  '
  function trim(value) {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    return value
  }
  function emit(value) {
    value = trim(value)
    gsub(/\047\047\047/, "", value)
    gsub(/\047\047/, "", value)
    gsub(/[[:space:]]+/, " ", value)
    if (length(value) < min_len) {
      return
    }
    if (value ~ /[\[\]\{\}\|]/) {
      return
    }
    if (value ~ /^\*/) {
      return
    }
    seen += 1
    if ((seen - 1) % stride != 0) {
      return
    }
    if (length(value) > max_len) {
      value = substr(value, 1, max_len)
      sub(/[[:space:]][^[:space:]]*$/, "", value)
    }
    print value
    emitted += 1
    if (emitted >= want) {
      exit
    }
  }
  /^<\|source:simplewiki\|>$/ {
    inside = 1
    next
  }
  inside && /^<\|source:/ {
    emit(paragraph)
    exit
  }
  !inside {
    next
  }
  /^<\|page:/ {
    emit(paragraph)
    paragraph = ""
    next
  }
  /^[[:space:]]*$/ {
    emit(paragraph)
    paragraph = ""
    next
  }
  {
    if (paragraph != "") {
      paragraph = paragraph " " $0
    } else {
      paragraph = $0
    }
  }
  END {
    emit(paragraph)
  }
  ' "$corpus" > "$prompts_path"

actual_prompts="$(wc -l < "$prompts_path" | tr -d ' ')"
if [[ "$actual_prompts" -eq 0 ]]; then
  echo "no SimpleWiki prompts extracted from $corpus" >&2
  exit 1
fi

{
  echo "<|source:synthetic-simplewiki-self|>"
  echo "<|run:${run_id}|>"
  echo
} > "$synthetic_text"
: > "$trace_jsonl"

index=0
while IFS= read -r prompt; do
  sample_number="$(printf "%04d" "$index")"
  sample_text="$out_dir/sample-${sample_number}.txt"
  sample_trace="$out_dir/sample-${sample_number}.trace.jsonl"
  seed="$((sample_seed + index))"

  echo "== prompt ${sample_number}/${actual_prompts}: seed=${seed}"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$model" \
    --vocab "$vocab" \
    --tokens "$tokens" \
    --prompt "$prompt" \
    --max-new-tokens "$max_new_tokens" \
    --decode sample \
    --sample-seed "$seed" \
    --top-k "$top_k" \
    --repeat-window "$repeat_window" \
    --repeat-penalty-shift "$repeat_penalty_shift" \
    --max-repeat-run "$max_repeat_run" \
    --corpus-prior \
    --corpus-prior-logit-shift "$corpus_prior_logit_shift" \
    --strict-adjacency \
    --generated-only \
    --text-out "$sample_text" \
    --trace "$sample_trace"

  cat "$sample_trace" >> "$trace_jsonl"
  {
    echo "<|sample:${sample_number}|>"
    echo "$prompt"
    cat "$sample_text"
    echo
    echo
  } >> "$synthetic_text"

  index="$((index + 1))"
done < "$prompts_path"

cat > "$manifest" <<EOF
{
  "schema": "nsrl.simplewiki_self_synthesis_manifest.v1",
  "run_id": "$run_id",
  "corpus": "$corpus",
  "model": "$model",
  "vocab": "$vocab",
  "tokens": "$tokens",
  "prompt_count_requested": $prompt_count,
  "prompt_count_actual": $actual_prompts,
  "prompt_stride": $prompt_stride,
  "prompt_min_chars": $prompt_min_chars,
  "prompt_max_chars": $prompt_max_chars,
  "max_new_tokens": $max_new_tokens,
  "top_k": $top_k,
  "sample_seed": $sample_seed,
  "repeat_window": $repeat_window,
  "repeat_penalty_shift": $repeat_penalty_shift,
  "max_repeat_run": $max_repeat_run,
  "corpus_prior_logit_shift": $corpus_prior_logit_shift,
  "text_out_mode": "prompt_plus_generated_only",
  "synthetic_text": "$synthetic_text",
  "trace_jsonl": "$trace_jsonl",
  "prompts": "$prompts_path"
}
EOF

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "prompts=$prompts_path"
echo "synthetic_text=$synthetic_text"
echo "trace_jsonl=$trace_jsonl"
echo "manifest=$manifest"
