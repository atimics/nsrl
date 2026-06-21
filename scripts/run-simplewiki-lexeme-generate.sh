#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
base_dir="${BASE_DIR:-data/processed/simplewiki-expository-v1}"
out_dir="${OUT_DIR:-data/processed/simplewiki-lexeme-generation-${run_id}}"

model="${MODEL:-$base_dir/batch-accum-sweep-20260621T114426Z/seq8-probe/seq8-b8-s19.nsrllm}"
vocab="${VOCAB:-$base_dir/v4096.vocab.tsv}"
tokens="${TOKENS:-$base_dir/v4096.tokens.u16}"
profile="${DECODE_PROFILE:-coherent-prose}"
max_new_tokens="${MAX_NEW_TOKENS:-180}"
top_k_default="${TOP_K:-8}"
sample_seed_default="${SAMPLE_SEED:-19}"
best_of_n="${BEST_OF_N:-4}"
corpus_prior_order="${CORPUS_PRIOR_ORDER:-3}"
corpus_prior_logit_shift="${CORPUS_PRIOR_LOGIT_SHIFT:-7}"
prompt_topic_radius="${PROMPT_TOPIC_RADIUS:-64}"
prompt_topic_min_q15="${PROMPT_TOPIC_MIN_Q15:-1024}"
prompt_topic_logit_shift="${PROMPT_TOPIC_LOGIT_SHIFT:-3}"
local_frequency_cap="${LOCAL_FREQUENCY_CAP:-}"
local_frequency_min_q15="${LOCAL_FREQUENCY_MIN_Q15:-}"
local_frequency_logit_shift="${LOCAL_FREQUENCY_LOGIT_SHIFT:-}"
local_frequency_hard_cap="${LOCAL_FREQUENCY_HARD_CAP:-}"

strict_memory_on_steps="${STRICT_MEMORY_ON_STEPS:-}"
strict_memory_off_steps="${STRICT_MEMORY_OFF_STEPS:-}"
if [[ "${STRICT_MEMORY_ALWAYS:-0}" == "1" ]]; then
  strict_memory_on_steps=0
  strict_memory_off_steps=0
fi

mkdir -p "$out_dir/candidates"

for path in "$model" "$vocab" "$tokens"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
done

echo "run_id=$run_id"
echo "out_dir=$out_dir"
echo "model=$model"
echo "vocab=$vocab"
echo "tokens=$tokens"
echo "profile=$profile"
echo "best_of_n=$best_of_n"
echo "corpus_prior_order=$corpus_prior_order"
echo "local_frequency_hard_cap=${local_frequency_hard_cap:-0}"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

decode_extra_args=()
if [[ -n "$strict_memory_on_steps" ]]; then
  decode_extra_args+=(--strict-memory-on-steps "$strict_memory_on_steps")
fi
if [[ -n "$strict_memory_off_steps" ]]; then
  decode_extra_args+=(--strict-memory-off-steps "$strict_memory_off_steps")
fi
if [[ -n "$local_frequency_cap" ]]; then
  decode_extra_args+=(--decode-local-frequency-cap "$local_frequency_cap")
fi
if [[ -n "$local_frequency_min_q15" ]]; then
  decode_extra_args+=(--decode-local-frequency-min-q15 "$local_frequency_min_q15")
fi
if [[ -n "$local_frequency_logit_shift" ]]; then
  decode_extra_args+=(--decode-local-frequency-logit-shift "$local_frequency_logit_shift")
fi
if [[ -n "$local_frequency_hard_cap" ]]; then
  decode_extra_args+=(--decode-local-frequency-hard-cap "$local_frequency_hard_cap")
fi

score_candidate() {
  local trace="$1"
  local text="$2"
  perl -Mstrict -Mwarnings -MJSON::PP -e '
    my ($trace, $text_path) = @ARGV;
    open my $tfh, "<", $trace or die "open $trace: $!";
    my $json = decode_json(<$tfh>);
    close $tfh;
    open my $xfh, "<", $text_path or die "open $text_path: $!";
    local $/;
    my $text = lc <$xfh>;
    close $xfh;

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

    my @words = ($text =~ /[a-z][a-z'\''-]*/g);
    my %counts;
    $counts{$_}++ for @words;
    my $unique_words = scalar keys %counts;
    my $word_count = scalar @words;
    my $max_word_count = 0;
    my $repeated_words_gt4 = 0;
    for my $count (values %counts) {
      $max_word_count = $count if $count > $max_word_count;
      $repeated_words_gt4 += 1 if $count > 4;
    }
    my @cruft = ($text =~ /\b(?:http|www|nbsp|ndash|references|website|websites|category|class|align|bgcolor|boot|mathbf|prime|tannhauser|mercy|dread|hast|thou|thee)\b/g);
    my @sentence_marks = ($text =~ /[.!?]/g);
    my @explainers = ($text =~ /\b(?:is|are|was|were|has|have|can|called|because|means|used|made|found|known|includes|example|type|part|people|water|earth|science|computer|energy)\b/g);
    my $function_soup = 0;
    $function_soup += ($text =~ /\b(?:the|of|and|in|to|a|is|that|as|for|on)\b(?:\W+\b(?:the|of|and|in|to|a|is|that|as|for|on)\b){5,}/g);

    my $target_distinct = 620;
    my $distinct_distance = abs($distinct - $target_distinct);
    my $score =
        ($distinct * 4)
      + ($unique_words * 5)
      + ($memory_context * 2)
      + ($memory_adjust * 2)
      + ($corpus_adjust * 2)
      + ($candidate_count * 10)
      + (scalar(@sentence_marks) * 35)
      + (scalar(@explainers) * 8)
      - ($distinct_distance * 2)
      - ($max_run * 90)
      - ($max_token_count * 8)
      - ($repeated_bigram * 35)
      - ($repeated_trigram * 70)
      - ($max_word_count * 45)
      - ($repeated_words_gt4 * 80)
      - (scalar(@cruft) * 120)
      - ($function_soup * 200);

    print join("\t", int($score), $word_count, $unique_words, $max_word_count, $repeated_words_gt4, scalar(@cruft), scalar(@sentence_marks), scalar(@explainers));
  ' "$trace" "$text"
}

generate_sample() {
  local label="$1"
  local prompt="$2"
  local seed="$3"
  local top_k="$4"

  local text_out="$out_dir/${label}.txt"
  local trace_out="$out_dir/${label}.trace.jsonl"
  local best_score=""
  local best_seed="$seed"
  local best_text=""
  local best_trace=""
  local best_candidate_index=0
  local best_extra=""

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
      --prompt-topic-radius "$prompt_topic_radius" \
      --prompt-topic-min-q15 "$prompt_topic_min_q15" \
      --prompt-topic-logit-shift "$prompt_topic_logit_shift" \
      --strict-adjacency \
      --quality-weight-profile prose-aware \
      ${decode_extra_args[@]+"${decode_extra_args[@]}"} \
      --text-out "$candidate_text" \
      --trace "$candidate_trace"

    local scored
    scored="$(score_candidate "$candidate_trace" "$candidate_text")"
    local score="${scored%%$'\t'*}"
    local extra="${scored#*$'\t'}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$prompt" "$candidate_seed" "$top_k" "$score" "$candidate_index" \
      "$extra" "$candidate_text" "$candidate_trace" \
      >> "$out_dir/candidate-samples.tsv"

    if [[ -z "$best_score" || "$score" -gt "$best_score" ]]; then
      best_score="$score"
      best_seed="$candidate_seed"
      best_text="$candidate_text"
      best_trace="$candidate_trace"
      best_candidate_index="$candidate_index"
      best_extra="$extra"
    fi
  done

  cp "$best_text" "$text_out"
  cp "$best_trace" "$trace_out"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$prompt" "$best_seed" "$best_score" "$text_out" "$trace_out" \
    >> "$out_dir/samples.tsv"
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$best_candidate_index" "$best_seed" "$best_score" "$best_extra" \
    >> "$out_dir/selected-candidates.tsv"
}

printf '%s\n' "sample	prompt	seed	score	text	trace" > "$out_dir/samples.tsv"
printf '%s\n' "sample	prompt	seed	top_k	score	candidate_index	word_count	unique_words	max_word_count	repeated_words_gt4	cruft_count	sentence_marks	explainer_count	text	trace" \
  > "$out_dir/candidate-samples.tsv"
printf '%s\n' "sample	candidate_index	seed	score	word_count	unique_words	max_word_count	repeated_words_gt4	cruft_count	sentence_marks	explainer_count" \
  > "$out_dir/selected-candidates.tsv"

generate_sample "earth" "the earth is" "$sample_seed_default" "$top_k_default"
generate_sample "water" "water is" 23 "$top_k_default"
generate_sample "computer" "a computer is" 29 "$top_k_default"
generate_sample "science" "science is" 31 "$top_k_default"
generate_sample "energy" "energy is" 37 "$top_k_default"

metrics_tsv="$out_dir/generation-metrics.tsv"
printf '%s\n' \
  "sample	prompt	seed	score	generated_token_count	distinct_token_per_mille	max_token_count	max_token_run	repeated_bigram_count	repeated_trigram_count	mean_selected_probability_q15	mean_candidate_count	steps_with_any_decode_adjust	steps_with_local_frequency_adjust	steps_with_memory_adjust	steps_with_corpus_prior_adjust	word_count	unique_words	max_word_count	repeated_words_gt4	cruft_count	sentence_marks	explainer_count	text" \
  > "$metrics_tsv"

perl -Mstrict -Mwarnings -MJSON::PP -F'\t' -lane '
  next if $. == 1;
  my ($sample, $prompt, $seed, $score, $text_path, $trace_path) = @F;
  open my $fh, "<", $trace_path or die "open $trace_path: $!";
  my $line = <$fh>;
  close $fh;
  my $json = decode_json($line);
  my $m = $json->{metrics} // {};
  open my $xfh, "<", $text_path or die "open $text_path: $!";
  local $/;
  my $text = lc <$xfh>;
  close $xfh;
  my @words = ($text =~ /[a-z][a-z'\''-]*/g);
  my %counts;
  $counts{$_}++ for @words;
  my $max_word_count = 0;
  my $repeated_words_gt4 = 0;
  for my $count (values %counts) {
    $max_word_count = $count if $count > $max_word_count;
    $repeated_words_gt4 += 1 if $count > 4;
  }
  my @cruft = ($text =~ /\b(?:http|www|nbsp|ndash|references|website|websites|category|class|align|bgcolor|boot|mathbf|prime|tannhauser|mercy|dread|hast|thou|thee)\b/g);
  my @sentence_marks = ($text =~ /[.!?]/g);
  my @explainers = ($text =~ /\b(?:is|are|was|were|has|have|can|called|because|means|used|made|found|known|includes|example|type|part|people|water|earth|science|computer|energy)\b/g);
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
  print join "\t", $sample, $prompt, $seed, $score, (map { $m->{$_} // 0 } @keys), scalar(@words), scalar(keys %counts), $max_word_count, $repeated_words_gt4, scalar(@cruft), scalar(@sentence_marks), scalar(@explainers), $text_path;
' "$out_dir/samples.tsv" >> "$metrics_tsv"

manifest="$out_dir/manifest.json"
perl -Mstrict -Mwarnings -MJSON::PP -e '
  my ($manifest, $run_id, $out_dir, $model, $vocab, $tokens, $profile, $best_of_n, $max_new_tokens, $top_k, $corpus_order, $corpus_shift, $topic_radius, $topic_min, $topic_shift, $local_hard_cap, $metrics) = @ARGV;
  my $doc = {
    schema => "nsrl.simplewiki_lexeme_generation_run.v1",
    run_id => $run_id,
    out_dir => $out_dir,
    model => $model,
    vocab => $vocab,
    tokens => $tokens,
    decode_profile => $profile,
    best_of_n => int($best_of_n),
    max_new_tokens => int($max_new_tokens),
    top_k => int($top_k),
    corpus_prior_order => int($corpus_order),
    corpus_prior_logit_shift => int($corpus_shift),
    prompt_topic_radius => int($topic_radius),
    prompt_topic_min_q15 => int($topic_min),
    prompt_topic_logit_shift => int($topic_shift),
    local_frequency_hard_cap => $local_hard_cap eq "" ? 0 : int($local_hard_cap),
    metrics => $metrics,
    candidate_samples => "$out_dir/candidate-samples.tsv",
    selected_candidates => "$out_dir/selected-candidates.tsv",
  };
  open my $fh, ">", $manifest or die "open $manifest: $!";
  print {$fh} JSON::PP->new->canonical->pretty->encode($doc);
' "$manifest" "$run_id" "$out_dir" "$model" "$vocab" "$tokens" "$profile" \
  "$best_of_n" "$max_new_tokens" "$top_k_default" "$corpus_prior_order" \
  "$corpus_prior_logit_shift" "$prompt_topic_radius" "$prompt_topic_min_q15" \
  "$prompt_topic_logit_shift" "$local_frequency_hard_cap" "$metrics_tsv"

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "metrics=$metrics_tsv"
echo "manifest=$manifest"
