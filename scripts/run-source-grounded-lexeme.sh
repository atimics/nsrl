#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${RUN_DIR:-data/processed/visionary-balanced-prose-balanced-prose-literary-v1}"
out_dir="${OUT_DIR:-data/processed/source-grounded-lexeme-${run_id}}"

model="${MODEL:-$run_dir/v4096.nsrllm}"
vocab="${VOCAB:-$run_dir/v4096.vocab.tsv}"
profile="${DECODE_PROFILE:-grounded-prose}"
max_new_tokens="${MAX_NEW_TOKENS:-220}"
top_k_default="${TOP_K:-12}"
corpus_prior_order="${CORPUS_PRIOR_ORDER:-3}"
corpus_prior_logit_shift="${CORPUS_PRIOR_LOGIT_SHIFT:-7}"
sample_seed_default="${SAMPLE_SEED:-17}"
best_of_n="${BEST_OF_N:-1}"
strict_memory_on_steps="${STRICT_MEMORY_ON_STEPS:-}"
strict_memory_off_steps="${STRICT_MEMORY_OFF_STEPS:-}"
if [[ "${STRICT_MEMORY_ALWAYS:-0}" == "1" ]]; then
  strict_memory_on_steps=0
  strict_memory_off_steps=0
fi

mkdir -p "$out_dir/source-priors"
mkdir -p "$out_dir/candidates"

for path in "$model" "$vocab"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
done

source_shakespeare="${SOURCE_SHAKESPEARE:-$run_dir/shakespeare.source.txt}"
source_blake="${SOURCE_BLAKE:-$run_dir/blake.source.txt}"
source_crowley="${SOURCE_CROWLEY:-$run_dir/crowley.source.txt}"
source_simplewiki="${SOURCE_SIMPLEWIKI:-$run_dir/simplewiki-synthetic.source.txt}"

declare -a sources=(
  "shakespeare:$source_shakespeare"
  "blake:$source_blake"
  "crowley:$source_crowley"
  "simplewiki:$source_simplewiki"
)

for spec in "${sources[@]}"; do
  source_path="${spec#*:}"
  if [[ ! -f "$source_path" ]]; then
    echo "missing source file: $source_path" >&2
    exit 1
  fi
done

echo "run_id=$run_id"
echo "run_dir=$run_dir"
echo "out_dir=$out_dir"
echo "model=$model"
echo "vocab=$vocab"
echo "profile=$profile"
echo "best_of_n=$best_of_n"
echo "corpus_prior_order=$corpus_prior_order"
if [[ -n "$strict_memory_on_steps" || -n "$strict_memory_off_steps" ]]; then
  echo "strict_memory_on_steps=${strict_memory_on_steps:-profile}"
  echo "strict_memory_off_steps=${strict_memory_off_steps:-profile}"
fi
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

decode_extra_args=()
if [[ -n "$strict_memory_on_steps" ]]; then
  decode_extra_args+=(--strict-memory-on-steps "$strict_memory_on_steps")
fi
if [[ -n "$strict_memory_off_steps" ]]; then
  decode_extra_args+=(--strict-memory-off-steps "$strict_memory_off_steps")
fi

tokenize_source() {
  local label="$1"
  local source_path="$2"
  local tokens="$out_dir/source-priors/${label}.tokens.u16"
  local trace="$out_dir/source-priors/${label}.tokens.trace.jsonl"

  date -u +"${label}_tokenize_started_at=%Y-%m-%dT%H:%M:%SZ" >&2
  cargo run --release -q -p nsrl-corpus -- lexeme-tokenize-fixed-vocab \
    --corpus "$source_path" \
    --vocab "$vocab" \
    --tokens-out "$tokens" \
    --trace "$trace" \
    --seq-len 32 \
    --stride 1 \
    --preview-tokens 16
  echo "$tokens"
}

generate_sample() {
  local label="$1"
  local source_label="$2"
  local tokens="$3"
  local prompt="$4"
  local seed="$5"
  local top_k="$6"

  local text_out="$out_dir/${label}.txt"
  local trace_out="$out_dir/${label}.trace.jsonl"
  local best_score=""
  local best_seed="$seed"
  local best_text=""
  local best_trace=""
  local best_candidate_index=0

  for ((candidate_index = 0; candidate_index < best_of_n; candidate_index += 1)); do
    local candidate_seed=$((seed + candidate_index * 101))
    local candidate_label="${label}-cand${candidate_index}-seed${candidate_seed}"
    local candidate_text="$out_dir/candidates/${candidate_label}.txt"
    local candidate_trace="$out_dir/candidates/${candidate_label}.trace.jsonl"

    date -u +"${candidate_label}_generation_started_at=%Y-%m-%dT%H:%M:%SZ"
    cargo run --release -q -p nsrl-train -- \
      --mode lexeme-generate \
      --model "$model" \
      --vocab "$vocab" \
      --tokens "$tokens" \
      --prompt "$prompt" \
      --max-new-tokens "$max_new_tokens" \
      --decode sample \
      --sample-seed "$candidate_seed" \
      --decode-profile "$profile" \
      --top-k "$top_k" \
      --corpus-prior \
      --corpus-prior-order "$corpus_prior_order" \
      --corpus-prior-logit-shift "$corpus_prior_logit_shift" \
      --strict-adjacency \
      ${decode_extra_args[@]+"${decode_extra_args[@]}"} \
      --text-out "$candidate_text" \
      --trace "$candidate_trace"

    local score
    score="$(perl -Mstrict -Mwarnings -MJSON::PP -e '
      my ($trace) = @ARGV;
      open my $fh, "<", $trace or die "open $trace: $!";
      my $json = decode_json(<$fh>);
      close $fh;
      my $m = $json->{metrics} // {};
      my $distinct = $m->{distinct_token_per_mille} // 0;
      my $memory_context = $m->{steps_with_memory_context} // 0;
      my $memory_adjust = $m->{steps_with_memory_adjust} // 0;
      my $corpus_adjust = $m->{steps_with_corpus_prior_adjust} // 0;
      my $candidate_count = $m->{mean_candidate_count} // 0;
      my $max_run = $m->{max_token_run} // 0;
      my $max_token_count = $m->{max_token_count} // 0;
      my $repeated_bigram = $m->{repeated_bigram_count} // 0;
      my $repeated_trigram = $m->{repeated_trigram_count} // 0;
      my $target_distinct = 620;
      my $distinct_distance = abs($distinct - $target_distinct);
      my $score =
          ($distinct * 4)
        + ($memory_context * 3)
        + ($memory_adjust * 2)
        + ($corpus_adjust * 2)
        + ($candidate_count * 12)
        - ($distinct_distance * 2)
        - ($max_run * 90)
        - ($max_token_count * 8)
        - ($repeated_bigram * 35)
        - ($repeated_trigram * 70);
      print int($score);
    ' "$candidate_trace")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$source_label" "$prompt" "$candidate_seed" "$top_k" "$score" \
      "$candidate_index" "$candidate_text" "$candidate_trace" \
      >> "$out_dir/candidate-samples.tsv"

    if [[ -z "$best_score" || "$score" -gt "$best_score" ]]; then
      best_score="$score"
      best_seed="$candidate_seed"
      best_text="$candidate_text"
      best_trace="$candidate_trace"
      best_candidate_index="$candidate_index"
    fi
  done

  cp "$best_text" "$text_out"
  cp "$best_trace" "$trace_out"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$source_label" "$prompt" "$best_seed" "$text_out" "$trace_out" \
    >> "$out_dir/samples.tsv"
  printf '%s\t%s\t%s\t%s\n' "$label" "$best_candidate_index" "$best_seed" "$best_score" \
    >> "$out_dir/selected-candidates.tsv"
}

printf '%s\n' "sample	source	prompt	seed	text	trace" > "$out_dir/samples.tsv"
printf '%s\n' "sample	source	prompt	seed	top_k	score	candidate_index	text	trace" \
  > "$out_dir/candidate-samples.tsv"
printf '%s\n' "sample	candidate_index	seed	score" > "$out_dir/selected-candidates.tsv"

shakespeare_tokens=""
blake_tokens=""
crowley_tokens=""
simplewiki_tokens=""
for spec in "${sources[@]}"; do
  label="${spec%%:*}"
  source_path="${spec#*:}"
  tokens="$(tokenize_source "$label" "$source_path")"
  case "$label" in
    shakespeare) shakespeare_tokens="$tokens" ;;
    blake) blake_tokens="$tokens" ;;
    crowley) crowley_tokens="$tokens" ;;
    simplewiki) simplewiki_tokens="$tokens" ;;
    *)
      echo "unknown source label: $label" >&2
      exit 1
      ;;
  esac
done

generate_sample "to-be-shakespeare-grounded" "shakespeare" "$shakespeare_tokens" \
  "to be or not to be" 7 8
generate_sample "world-blake-grounded" "blake" "$blake_tokens" \
  "the world is" "$sample_seed_default" "$top_k_default"
generate_sample "soul-blake-grounded" "blake" "$blake_tokens" \
  "the soul is" 11 "$top_k_default"
generate_sample "world-crowley-grounded" "crowley" "$crowley_tokens" \
  "the world is" 23 "$top_k_default"
generate_sample "world-simplewiki-grounded" "simplewiki" "$simplewiki_tokens" \
  "the world is" 29 "$top_k_default"

metrics_tsv="$out_dir/generation-metrics.tsv"
printf '%s\n' \
  "sample	source	prompt	seed	generated_token_count	distinct_token_per_mille	max_token_count	max_token_run	repeated_bigram_count	repeated_trigram_count	mean_selected_probability_q15	mean_candidate_count	steps_with_any_decode_adjust	steps_with_local_frequency_adjust	steps_with_memory_adjust	steps_with_corpus_prior_adjust	text" \
  > "$metrics_tsv"

perl -Mstrict -Mwarnings -MJSON::PP -F'\t' -lane '
  next if $. == 1;
  my ($sample, $source, $prompt, $seed, $text_path, $trace_path) = @F;
  open my $fh, "<", $trace_path or die "open $trace_path: $!";
  my $line = <$fh>;
  close $fh;
  my $json = decode_json($line);
  my $m = $json->{metrics} // {};
  my @keys = qw(
    generated_token_count
    distinct_token_per_mille
    max_token_count
    max_token_run
    repeated_bigram_count
    repeated_trigram_count
    mean_selected_probability_q15
    mean_candidate_count
    steps_with_any_decode_adjust
    steps_with_local_frequency_adjust
    steps_with_memory_adjust
    steps_with_corpus_prior_adjust
  );
  print join "\t", $sample, $source, $prompt, $seed, (map { $m->{$_} // 0 } @keys), $text_path;
' "$out_dir/samples.tsv" >> "$metrics_tsv"

source_metrics_tsv="$out_dir/source-prior-metrics.tsv"
printf '%s\n' \
  "source	input_bytes	normalized_bytes	known_lexeme_tokens	fallback_byte_tokens	token_count	window_count	token_hash" \
  > "$source_metrics_tsv"
for spec in "${sources[@]}"; do
  label="${spec%%:*}"
  trace="$out_dir/source-priors/${label}.tokens.trace.jsonl"
  perl -Mstrict -Mwarnings -MJSON::PP -e '
    my ($label, $trace) = @ARGV;
    open my $fh, "<", $trace or die "open $trace: $!";
    my $json = decode_json(<$fh>);
    close $fh;
    print join("\t",
      $label,
      $json->{input}->{bytes} // 0,
      $json->{input}->{normalized_bytes} // 0,
      $json->{lexemes}->{known_lexeme_tokens} // 0,
      $json->{lexemes}->{fallback_byte_tokens} // 0,
      $json->{tokens}->{count} // 0,
      $json->{windows}->{count} // 0,
      $json->{tokens}->{hash} // "",
    ), "\n";
  ' "$label" "$trace" >> "$source_metrics_tsv"
done

manifest="$out_dir/manifest.json"
perl -Mstrict -Mwarnings -MJSON::PP -e '
  my (
    $manifest, $run_id, $run_dir, $out_dir, $model, $vocab, $profile,
    $metrics, $source_metrics, $best_of_n, $corpus_prior_order,
    $corpus_prior_logit_shift, $strict_memory_on_steps, $strict_memory_off_steps,
    $source_shakespeare, $source_blake, $source_crowley, $source_simplewiki
  ) = @ARGV;
  my $doc = {
    schema => "nsrl.source_grounded_lexeme_run.v1",
    run_id => $run_id,
    run_dir => $run_dir,
    out_dir => $out_dir,
    model => $model,
    vocab => $vocab,
    decode_profile => $profile,
    best_of_n => int($best_of_n),
    corpus_prior_order => int($corpus_prior_order),
    corpus_prior_logit_shift => int($corpus_prior_logit_shift),
    strict_memory_on_steps => length($strict_memory_on_steps) ? int($strict_memory_on_steps) : undef,
    strict_memory_off_steps => length($strict_memory_off_steps) ? int($strict_memory_off_steps) : undef,
    sources => {
      shakespeare => $source_shakespeare,
      blake => $source_blake,
      crowley => $source_crowley,
      simplewiki => $source_simplewiki,
    },
    metrics => $metrics,
    source_metrics => $source_metrics,
    candidate_samples => "$out_dir/candidate-samples.tsv",
    selected_candidates => "$out_dir/selected-candidates.tsv",
  };
  open my $fh, ">", $manifest or die "open $manifest: $!";
  print {$fh} JSON::PP->new->canonical->pretty->encode($doc);
' "$manifest" "$run_id" "$run_dir" "$out_dir" "$model" "$vocab" "$profile" \
  "$metrics_tsv" "$source_metrics_tsv" "$best_of_n" "$corpus_prior_order" \
  "$corpus_prior_logit_shift" "$strict_memory_on_steps" "$strict_memory_off_steps" \
  "$source_shakespeare" "$source_blake" "$source_crowley" "$source_simplewiki"

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "metrics=$metrics_tsv"
echo "source_metrics=$source_metrics_tsv"
echo "manifest=$manifest"
