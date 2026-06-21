#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_dir="${OUT_DIR:-data/processed/visionary-balanced-prose-${run_id}}"

raw_shakespeare="${SHAKESPEARE:-data/raw/shakespeare-gutenberg-100.txt}"
blake_poems="${BLAKE_POEMS:-data/processed/blake-poems.clean.txt}"
blake_marriage="${BLAKE_MARRIAGE:-data/processed/blake-marriage-heaven-hell.clean.txt}"
crowley_household="${CROWLEY_HOUSEHOLD:-data/processed/crowley-household-gods.clean.txt}"
crowley_tannhauser="${CROWLEY_TANNHAUSER:-data/processed/crowley-tannhauser.clean.txt}"
synthetic_simplewiki="${SYNTHETIC_SIMPLEWIKI:-data/processed/simplewiki-self-synthesis-visionary4096-p64-corrected/filtered-synthetic-corpus.txt}"

source_bytes="${SOURCE_BYTES:-320000}"
shakespeare_bytes="${SHAKESPEARE_BYTES:-$source_bytes}"
blake_bytes="${BLAKE_BYTES:-$source_bytes}"
crowley_bytes="${CROWLEY_BYTES:-$source_bytes}"
synthetic_bytes="${SYNTHETIC_BYTES:-$source_bytes}"
interleave_chunk_bytes="${INTERLEAVE_CHUNK_BYTES:-4096}"
vocab_size="${VOCAB_SIZE:-4096}"
embed_dim="${EMBED_DIM:-16}"
freq_cap="${FREQ_CAP:-4096}"
embed_windows="${EMBED_WINDOWS:-256000}"
softmax_windows="${SOFTMAX_WINDOWS:-256000}"
softmax_seq_len="${SOFTMAX_SEQ_LEN:-8}"
lexeme_context_features="${LEXEME_CONTEXT_FEATURES:-mean}"
softmax_lr_shift="${SOFTMAX_LR_SHIFT:-21}"
softmax_max_lr_shift="${SOFTMAX_MAX_LR_SHIFT:-23}"
corpus_prior_order="${CORPUS_PRIOR_ORDER:-2}"
corpus_prior_logit_shift="${CORPUS_PRIOR_LOGIT_SHIFT:-7}"
repeat_window="${REPEAT_WINDOW:-96}"
repeat_penalty_shift="${REPEAT_PENALTY_SHIFT:-3}"
max_repeat_run="${MAX_REPEAT_RUN:-2}"
no_repeat_ngram="${NO_REPEAT_NGRAM:-3}"

mkdir -p "$out_dir"

for path in \
  "$raw_shakespeare" \
  "$blake_poems" \
  "$blake_marriage" \
  "$crowley_household" \
  "$crowley_tannhauser" \
  "$synthetic_simplewiki"
do
  if [[ ! -f "$path" ]]; then
    echo "missing source file: $path" >&2
    exit 1
  fi
done

clean_shakespeare="$out_dir/shakespeare.clean.txt"
shakespeare_body="$out_dir/shakespeare.body.txt"
corpus="$out_dir/corpus.txt"
tokens="$out_dir/v${vocab_size}.tokens.u16"
vocab_tsv="$out_dir/v${vocab_size}.vocab.tsv"
token_trace="$out_dir/v${vocab_size}.tokens.trace.jsonl"
embedding="$out_dir/v${vocab_size}.nsrllex"
embedding_trace="$out_dir/v${vocab_size}.embedding.trace.jsonl"
softmax="$out_dir/v${vocab_size}.nsrllm"
softmax_trace="$out_dir/v${vocab_size}.softmax.trace.jsonl"
metrics_tsv="$out_dir/generation-metrics.tsv"
manifest="$out_dir/manifest.json"

echo "run_id=$run_id"
echo "out_dir=$out_dir"
echo "source_bytes=$source_bytes"
echo "shakespeare_bytes=$shakespeare_bytes"
echo "blake_bytes=$blake_bytes"
echo "crowley_bytes=$crowley_bytes"
echo "synthetic_bytes=$synthetic_bytes"
echo "interleave_chunk_bytes=$interleave_chunk_bytes"
echo "softmax_seq_len=$softmax_seq_len"
echo "lexeme_context_features=$lexeme_context_features"
echo "corpus_prior_order=$corpus_prior_order"
echo "corpus_prior_logit_shift=$corpus_prior_logit_shift"
echo "repeat_window=$repeat_window"
echo "repeat_penalty_shift=$repeat_penalty_shift"
echo "max_repeat_run=$max_repeat_run"
echo "no_repeat_ngram=$no_repeat_ngram"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

cargo run --release -q -p nsrl-corpus -- clean-gutenberg \
  --corpus "$raw_shakespeare" \
  --out "$clean_shakespeare"

awk '
  /^THE SONNETS$/ {
    seen += 1
    if (seen == 2) {
      emit = 1
    }
  }
  emit {
    print
  }
' "$clean_shakespeare" > "$shakespeare_body"

build_source_bundle() {
  local source_bundle="$1"
  shift
  : > "$source_bundle"
  for path in "$@"; do
    cat "$path" >> "$source_bundle"
    printf '\n\n' >> "$source_bundle"
  done

  local bundle_bytes
  bundle_bytes="$(wc -c < "$source_bundle" | tr -d ' ')"
  if [[ "$bundle_bytes" -eq 0 ]]; then
    echo "empty source bundle: $source_bundle" >&2
    exit 1
  fi
}

write_interleaved_corpus() {
  local chunk_bytes="$1"
  shift

  if [[ "$chunk_bytes" -le 0 ]]; then
    echo "INTERLEAVE_CHUNK_BYTES must be positive" >&2
    exit 1
  fi

  perl -Mstrict -Mwarnings -e '
    my ($out_path, $chunk_bytes, @spec) = @ARGV;
    die "source specs must be label/target/path triples\n" unless @spec % 3 == 0;
    open my $out, ">", $out_path or die "open $out_path: $!";

    my @sources;
    while (@spec) {
      my ($label, $target_bytes, $path) = splice @spec, 0, 3;
      $target_bytes = int($target_bytes);
      next if $target_bytes <= 0;
      open my $fh, "<", $path or die "open $path: $!";
      local $/;
      my $text = <$fh>;
      close $fh;
      die "empty source bundle for $label\n" if length($text) == 0;
      push @sources, {
        label => $label,
        target => $target_bytes,
        text => $text,
        offset => 0,
        emitted => 0,
      };
    }
    die "at least one source must have positive target bytes\n" unless @sources;

    print {$out} "<|balanced-interleaved-prose-v1|>\n";
    print {$out} join(" ", map { "<|source:" . $_->{label} . "|>" } @sources), "\n\n";

    my $remaining = 1;
    while ($remaining) {
      $remaining = 0;
      for my $source (@sources) {
        next if $source->{emitted} >= $source->{target};
        $remaining = 1;
        my $want = $chunk_bytes;
        my $left = $source->{target} - $source->{emitted};
        $want = $left if $want > $left;

        my $chunk = "";
        while (length($chunk) < $want) {
          my $text_len = length($source->{text});
          if ($source->{offset} >= $text_len) {
            $source->{offset} = 0;
            $chunk .= "\n";
            next;
          }
          my $take = $want - length($chunk);
          my $available = $text_len - $source->{offset};
          $take = $available if $take > $available;
          $chunk .= substr($source->{text}, $source->{offset}, $take);
          $source->{offset} += $take;
        }

        print {$out} $chunk;
        print {$out} "\n\n";
        $source->{emitted} += $want;
      }
    }

    close $out;
  ' "$corpus" "$chunk_bytes" "$@"
}

shakespeare_bundle="$out_dir/shakespeare.source.txt"
blake_bundle="$out_dir/blake.source.txt"
crowley_bundle="$out_dir/crowley.source.txt"
synthetic_bundle="$out_dir/simplewiki-synthetic.source.txt"

build_source_bundle "$shakespeare_bundle" "$shakespeare_body"
build_source_bundle "$blake_bundle" "$blake_poems" "$blake_marriage"
build_source_bundle "$crowley_bundle" "$crowley_household" "$crowley_tannhauser"
build_source_bundle "$synthetic_bundle" "$synthetic_simplewiki"

write_interleaved_corpus "$interleave_chunk_bytes" \
  shakespeare "$shakespeare_bytes" "$shakespeare_bundle" \
  blake "$blake_bytes" "$blake_bundle" \
  crowley "$crowley_bytes" "$crowley_bundle" \
  simplewiki-synthetic "$synthetic_bytes" "$synthetic_bundle"

date -u +"tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-corpus -- lexeme-tokenize \
  --corpus "$corpus" \
  --tokens-out "$tokens" \
  --vocab-out "$vocab_tsv" \
  --trace "$token_trace" \
  --seq-len 32 \
  --stride 1 \
  --max-vocab "$vocab_size" \
  --lexeme-vocab-profile balanced \
  --lexeme-frequency-cap "$freq_cap" \
  --preview-tokens 32

date -u +"embedding_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens "$tokens" \
  --vocab "$vocab_tsv" \
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
  --vocab "$vocab_tsv" \
  --model "$embedding" \
  --model-out "$softmax" \
  --trace "$softmax_trace" \
  --seq-len "$softmax_seq_len" \
  --lexeme-context-features "$lexeme_context_features" \
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
  --quality-weight-profile cruft-aware

generate_sample() {
  local label="$1"
  local prompt="$2"
  local seed="$3"
  local top_k="$4"

  local sample_text="$out_dir/${label}.txt"
  local sample_trace="$out_dir/${label}.trace.jsonl"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$softmax" \
    --vocab "$vocab_tsv" \
    --tokens "$tokens" \
    --prompt "$prompt" \
    --max-new-tokens 220 \
    --decode sample \
    --sample-seed "$seed" \
    --top-k "$top_k" \
    --repeat-window "$repeat_window" \
    --repeat-penalty-shift "$repeat_penalty_shift" \
    --max-repeat-run "$max_repeat_run" \
    --no-repeat-ngram "$no_repeat_ngram" \
    --corpus-prior \
    --corpus-prior-logit-shift "$corpus_prior_logit_shift" \
    --corpus-prior-order "$corpus_prior_order" \
    --strict-adjacency \
    --quality-weight-profile cruft-aware \
    --text-out "$sample_text" \
    --trace "$sample_trace"
}

date -u +"generation_started_at=%Y-%m-%dT%H:%M:%SZ"
generate_sample to-be-top8-rp2-s7 "to be or not to be" 7 8
generate_sample soul-top12-rp2-s11 "the soul is" 11 12
generate_sample world-top12-rp2-s17 "the world is" 17 12

printf '%s\n' "sample	cruft	words	distinct	repeated_words_gt3	max_word_count" > "$metrics_tsv"
for sample_text in "$out_dir"/*-rp2-s*.txt; do
  [[ -f "$sample_text" ]] || continue
  sample_name="$(basename "$sample_text" .txt)"
  perl -0777 -Mstrict -Mwarnings -e '
    my ($sample) = @ARGV;
    my $text = lc <STDIN>;
    my @cruft = ($text =~ /\b(?:http|www|class|align|bgcolor|fefefe|vcard|adr|nbsp|ndash|references|websites|linear|id|fn|org|com|km|january|february|march|april|june|july|august|september|october|november|december)\b/g);
    my @words = ($text =~ /[a-z][a-z'\'']*/g);
    my %counts;
    $counts{$_}++ for @words;
    my $max_count = 0;
    my $repeated = 0;
    for my $count (values %counts) {
      $max_count = $count if $count > $max_count;
      $repeated += 1 if $count > 3;
    }
    print join("\t", $sample, scalar(@cruft), scalar(@words), scalar(keys %counts), $repeated, $max_count), "\n";
  ' "$sample_name" < "$sample_text" >> "$metrics_tsv"
done

cat > "$manifest" <<EOF
{
  "schema": "nsrl.visionary_balanced_prose_run.v1",
  "run_id": "$run_id",
  "out_dir": "$out_dir",
  "source_bytes_per_section": $source_bytes,
  "source_layout": "round_robin_interleaved_chunks",
  "interleave_chunk_bytes": $interleave_chunk_bytes,
  "source_bytes": {
    "shakespeare": $shakespeare_bytes,
    "blake": $blake_bytes,
    "crowley": $crowley_bytes,
    "synthetic_simplewiki": $synthetic_bytes
  },
  "vocab_size": $vocab_size,
  "embedding_dim": $embed_dim,
  "frequency_cap": $freq_cap,
  "embedding_windows": $embed_windows,
  "softmax_windows": $softmax_windows,
  "softmax_seq_len": $softmax_seq_len,
  "lexeme_context_features": "$lexeme_context_features",
  "corpus_prior_order": $corpus_prior_order,
  "corpus_prior_logit_shift": $corpus_prior_logit_shift,
  "repeat_window": $repeat_window,
  "repeat_penalty_shift": $repeat_penalty_shift,
  "max_repeat_run": $max_repeat_run,
  "no_repeat_ngram": $no_repeat_ngram,
  "softmax_lr_shift": $softmax_lr_shift,
  "softmax_max_lr_shift": $softmax_max_lr_shift,
  "raw_shakespeare": "$raw_shakespeare",
  "blake_sources": ["$blake_poems", "$blake_marriage"],
  "crowley_sources": ["$crowley_household", "$crowley_tannhauser"],
  "synthetic_simplewiki": "$synthetic_simplewiki",
  "corpus": "$corpus",
  "tokens": "$tokens",
  "vocab": "$vocab_tsv",
  "model": "$softmax",
  "metrics": "$metrics_tsv"
}
EOF

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "corpus=$corpus"
echo "tokens=$tokens"
echo "vocab=$vocab_tsv"
echo "model=$softmax"
echo "metrics=$metrics_tsv"
echo "manifest=$manifest"
