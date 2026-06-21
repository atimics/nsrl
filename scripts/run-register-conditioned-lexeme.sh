#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
source_run_dir="${SOURCE_RUN_DIR:-data/processed/visionary-balanced-prose-balanced-prose-literary-v1}"
out_dir="${OUT_DIR:-data/processed/register-conditioned-lexeme-${run_id}}"

source_bytes="${SOURCE_BYTES:-120000}"
chunk_bytes="${CHUNK_BYTES:-2048}"
register_interval_chunks="${REGISTER_INTERVAL_CHUNKS:-4}"
vocab_size="${VOCAB_SIZE:-4096}"
embed_dim="${EMBED_DIM:-16}"
embed_windows="${EMBED_WINDOWS:-64000}"
softmax_windows="${SOFTMAX_WINDOWS:-64000}"
softmax_seq_len="${SOFTMAX_SEQ_LEN:-8}"
softmax_lr_shift="${SOFTMAX_LR_SHIFT:-20}"
softmax_max_lr_shift="${SOFTMAX_MAX_LR_SHIFT:-22}"
hidden_dim="${HIDDEN_DIM:-32}"
hidden_lr_shift="${HIDDEN_LR_SHIFT:-15}"
context_features="${LEXEME_CONTEXT_FEATURES:-ordered}"
freq_cap="${FREQ_CAP:-4096}"
max_new_tokens="${MAX_NEW_TOKENS:-220}"
generated_only="${GENERATED_ONLY:-1}"

mkdir -p "$out_dir"

declare -a source_specs=(
  "shakespeare:shakespeare:$source_run_dir/shakespeare.source.txt"
  "blake:blake:$source_run_dir/blake.source.txt"
  "crowley:crowley:$source_run_dir/crowley.source.txt"
  "simplewiki:simple wiki:$source_run_dir/simplewiki-synthetic.source.txt"
)

for spec in "${source_specs[@]}"; do
  path="${spec##*:}"
  if [[ ! -f "$path" ]]; then
    echo "missing source file: $path" >&2
    exit 1
  fi
done

corpus="$out_dir/corpus.txt"
prior_corpus="$out_dir/corpus.decode-prior.txt"
tokens="$out_dir/v${vocab_size}.tokens.u16"
prior_tokens="$out_dir/v${vocab_size}.decode-prior.tokens.u16"
vocab="$out_dir/v${vocab_size}.vocab.tsv"
token_trace="$out_dir/v${vocab_size}.tokens.trace.jsonl"
prior_token_trace="$out_dir/v${vocab_size}.decode-prior.tokens.trace.jsonl"
embedding="$out_dir/v${vocab_size}.nsrllex"
embedding_trace="$out_dir/v${vocab_size}.embedding.trace.jsonl"
softmax="$out_dir/v${vocab_size}.nsrllm"
softmax_trace="$out_dir/v${vocab_size}.softmax.trace.jsonl"
metrics_tsv="$out_dir/generation-metrics.tsv"
manifest="$out_dir/manifest.json"

echo "run_id=$run_id"
echo "source_run_dir=$source_run_dir"
echo "out_dir=$out_dir"
echo "source_bytes=$source_bytes"
echo "chunk_bytes=$chunk_bytes"
echo "register_interval_chunks=$register_interval_chunks"
echo "embed_windows=$embed_windows"
echo "softmax_windows=$softmax_windows"
echo "context_features=$context_features"
echo "hidden_dim=$hidden_dim"
echo "generated_only=$generated_only"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

perl -Mstrict -Mwarnings -e '
  my ($out_path, $prior_out_path, $source_bytes, $chunk_bytes, $register_interval_chunks, @specs) = @ARGV;
  open my $out, ">", $out_path or die "open $out_path: $!";
  open my $prior, ">", $prior_out_path or die "open $prior_out_path: $!";
  die "REGISTER_INTERVAL_CHUNKS must be positive\n" if int($register_interval_chunks) < 1;

  for my $spec (@specs) {
    my ($label, $phrase, $path) = split /:/, $spec, 3;
    open my $fh, "<", $path or die "open $path: $!";
    local $/;
    my $text = <$fh>;
    close $fh;
    $text =~ s/\r\n?/\n/g;
    my $limit = int($source_bytes);
    $limit = length($text) if $limit > length($text);
    my $offset = 0;
    my $chunk_index = 0;
    while ($offset < $limit) {
      my $take = int($chunk_bytes);
      my $left = $limit - $offset;
      $take = $left if $take > $left;
      my $chunk = substr($text, $offset, $take);
      $chunk =~ s/<\|[^|\n]+\|>//g;
      if (($chunk_index % int($register_interval_chunks)) == 0) {
        print {$out} "register $phrase voice\n";
      }
      print {$out} $chunk;
      print {$out} "\n\n";
      print {$prior} $chunk;
      print {$prior} "\n\n";
      $offset += $take;
      $chunk_index++;
    }
  }
  close $out;
  close $prior;
' "$corpus" "$prior_corpus" "$source_bytes" "$chunk_bytes" "$register_interval_chunks" "${source_specs[@]}"

date -u +"tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-corpus -- lexeme-tokenize \
  --corpus "$corpus" \
  --tokens-out "$tokens" \
  --vocab-out "$vocab" \
  --trace "$token_trace" \
  --seq-len 32 \
  --stride 1 \
  --max-vocab "$vocab_size" \
  --lexeme-vocab-profile balanced \
  --lexeme-frequency-cap "$freq_cap" \
  --preview-tokens 32

date -u +"decode_prior_tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-corpus -- lexeme-tokenize-fixed-vocab \
  --corpus "$prior_corpus" \
  --vocab "$vocab" \
  --tokens-out "$prior_tokens" \
  --trace "$prior_token_trace" \
  --seq-len 32 \
  --stride 1 \
  --preview-tokens 32

date -u +"embedding_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens "$tokens" \
  --vocab "$vocab" \
  --model-out "$embedding" \
  --trace "$embedding_trace" \
  --vocab-size "$vocab_size" \
  --embedding-dim "$embed_dim" \
  --context-radius 2 \
  --stride 1 \
  --max-windows "$embed_windows" \
  --epochs 1 \
  --lr-shift 8 \
  --concept-frequency-cap "$freq_cap" \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware

date -u +"softmax_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens "$tokens" \
  --vocab "$vocab" \
  --model "$embedding" \
  --model-out "$softmax" \
  --trace "$softmax_trace" \
  --seq-len "$softmax_seq_len" \
  --lexeme-context-features "$context_features" \
  --stride 1 \
  --max-windows "$softmax_windows" \
  --epochs 1 \
  --lr-shift "$softmax_lr_shift" \
  --lr-shift-decay-windows "$((softmax_windows / 2))" \
  --lr-shift-decay-step 1 \
  --max-lr-shift "$softmax_max_lr_shift" \
  --max-weight-delta 1 \
  --target-frequency-cap "$freq_cap" \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware \
  --lexeme-hidden-dim "$hidden_dim" \
  --lexeme-hidden-lr-shift "$hidden_lr_shift" \
  --max-hidden-weight-delta 1

generate_sample() {
  local label="$1"
  local prompt="$2"
  local seed="$3"
  local top_k="$4"
  local text_out="$out_dir/${label}.txt"
  local trace_out="$out_dir/${label}.trace.jsonl"
  local -a text_flags=()
  if [[ "$generated_only" != "0" ]]; then
    text_flags=(--generated-only)
  fi

  date -u +"${label}_generation_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$softmax" \
    --vocab "$vocab" \
    --tokens "$prior_tokens" \
    --prompt "$prompt" \
    --max-new-tokens "$max_new_tokens" \
    --decode sample \
    --sample-seed "$seed" \
    --top-k "$top_k" \
    --decode-profile coherent-prose \
    --corpus-prior \
    --corpus-prior-order 3 \
    --corpus-prior-logit-shift 7 \
    --strict-adjacency \
    --text-out "$text_out" \
    --trace "$trace_out" \
    "${text_flags[@]}"

  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$prompt" "$seed" "$text_out" "$trace_out" \
    >> "$out_dir/samples.tsv"
}

printf '%s\n' "sample	prompt	seed	text	trace" > "$out_dir/samples.tsv"
generate_sample "blake-world" "register blake voice the world is" 17 12
generate_sample "blake-soul" "register blake voice the soul is" 11 12
generate_sample "shakespeare-to-be" "register shakespeare voice to be or not to be" 7 8
generate_sample "crowley-world" "register crowley voice the world is" 23 12
generate_sample "simplewiki-world" "register simple wiki voice the world is" 29 12

printf '%s\n' \
  "sample	prompt	seed	generated_token_count	distinct_token_per_mille	max_token_count	max_token_run	repeated_bigram_count	repeated_trigram_count	mean_selected_probability_q15	mean_candidate_count	steps_with_any_decode_adjust	steps_with_local_frequency_adjust	steps_with_memory_adjust	steps_with_corpus_prior_adjust	text" \
  > "$metrics_tsv"
perl -Mstrict -Mwarnings -MJSON::PP -F'\t' -lane '
  next if $. == 1;
  my ($sample, $prompt, $seed, $text_path, $trace_path) = @F;
  open my $fh, "<", $trace_path or die "open $trace_path: $!";
  my $json = decode_json(<$fh>);
  close $fh;
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
  print join "\t", $sample, $prompt, $seed, map { $m->{$_} // 0 } @keys, $text_path;
' "$out_dir/samples.tsv" >> "$metrics_tsv"

perl -Mstrict -Mwarnings -MJSON::PP -e '
  my ($manifest, $run_id, $out_dir, $corpus, $prior_corpus, $tokens, $prior_tokens, $vocab, $model, $metrics, $generated_only) = @ARGV;
  my $doc = {
    schema => "nsrl.register_conditioned_lexeme_run.v1",
    run_id => $run_id,
    out_dir => $out_dir,
    corpus => $corpus,
    decode_prior_corpus => $prior_corpus,
    tokens => $tokens,
    decode_prior_tokens => $prior_tokens,
    vocab => $vocab,
    model => $model,
    metrics => $metrics,
    generated_only => $generated_only eq "0" ? JSON::PP::false : JSON::PP::true,
  };
  open my $fh, ">", $manifest or die "open $manifest: $!";
  print {$fh} JSON::PP->new->canonical->pretty->encode($doc);
' "$manifest" "$run_id" "$out_dir" "$corpus" "$prior_corpus" "$tokens" "$prior_tokens" "$vocab" "$softmax" "$metrics_tsv" "$generated_only"

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "model=$softmax"
echo "metrics=$metrics_tsv"
echo "manifest=$manifest"
