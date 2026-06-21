#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
source_run_dir="${SOURCE_RUN_DIR:-data/processed/simplewiki-expository-v1/topic-earth-curriculum-holo-sentence-stop-smoke-20260621}"
topic="${TOPIC:-earth}"
out_dir="${OUT_DIR:-$source_run_dir/paragraph-bestof-${run_id}}"

model="${MODEL:-$source_run_dir/${topic}.seq8.hidden32.nsrllm}"
tokens="${TOKENS:-$source_run_dir/${topic}.tokens.u16}"
vocab="${VOCAB:-data/processed/simplewiki-expository-v1/boundary-v1/v4096-boundary.vocab.tsv}"
prompt="${PROMPT:-the earth is an ancient planet}"

paragraph_sentences="${PARAGRAPH_SENTENCES:-3}"
candidates_per_sentence="${PARAGRAPH_CANDIDATES:-8}"
paragraph_prompt_mode="${PARAGRAPH_PROMPT_MODE:-full}"
max_new_tokens="${MAX_NEW_TOKENS:-32}"
min_words="${MIN_SENTENCE_WORDS:-8}"
max_words="${MAX_SENTENCE_WORDS:-26}"
top_k="${TOP_K:-32}"
sample_seed="${SAMPLE_SEED:-97}"
local_frequency_hard_cap="${LOCAL_FREQUENCY_HARD_CAP:-2}"
max_local_frequency_rejections="${MAX_LOCAL_FREQUENCY_REJECTIONS:-64}"

paragraph_text="$out_dir/paragraph.txt"
choices_tsv="$out_dir/choices.tsv"
manifest="$out_dir/manifest.json"

mkdir -p "$out_dir/candidates"

for path in "$model" "$tokens" "$vocab"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
done

printf '%s\n' "$prompt" > "$paragraph_text"
printf '%s\n' "sentence	candidate	seed	score	stopped	words	sentence_marks	local_frequency_rejections	repeated_bigram_count	repeated_trigram_count	source_exact_span	source_longest_run	source_bigram_per_mille	source_trigram_per_mille	fragment	trace	preview" > "$choices_tsv"

case "$paragraph_prompt_mode" in
  full | last | hybrid) ;;
  *)
    echo "PARAGRAPH_PROMPT_MODE must be full, last, or hybrid" >&2
    exit 1
    ;;
esac

generate_candidate() {
  local sentence_index="$1"
  local candidate_index="$2"
  local seed="$3"
  local prompt_text="$4"
  local fragment="$5"
  local trace="$6"

  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$model" \
    --vocab "$vocab" \
    --tokens "$tokens" \
    --prompt "$prompt_text" \
    --max-new-tokens "$max_new_tokens" \
    --decode sample \
    --sample-seed "$seed" \
    --top-k "$top_k" \
    --repeat-window 32 \
    --repeat-penalty-shift 2 \
    --max-repeat-run 2 \
    --no-repeat-ngram 3 \
    --corpus-prior \
    --corpus-prior-order 3 \
    --corpus-prior-logit-shift 6 \
    --prompt-topic-radius 64 \
    --prompt-topic-min-q15 4096 \
    --prompt-topic-logit-shift 4 \
    --decode-memory-order 8 \
    --decode-memory-min-order 1 \
    --decode-memory-logit-shift 0 \
    --strict-adjacency \
    --strict-topic \
    --quality-weight-profile prose-aware \
    --decode-frequency-cap 4096 \
    --decode-frequency-min-q15 2048 \
    --decode-frequency-logit-shift 5 \
    --decode-local-frequency-hard-cap "$local_frequency_hard_cap" \
    --generated-only \
    --stop-on-sentence-terminal \
    --text-out "$fragment" \
    --trace "$trace"
}

score_candidate() {
  local sentence_index="$1"
  local candidate_index="$2"
  local seed="$3"
  local fragment="$4"
  local trace="$5"

  perl -Mstrict -Mwarnings -MJSON::PP -0777 -e '
    my (
      $sentence_index, $candidate_index, $seed, $fragment_path, $trace_path,
      $tokens_path, $min_words, $max_words
    ) = @ARGV;

    open my $fh, "<", $fragment_path or die "open $fragment_path: $!";
    local $/;
    my $text = <$fh>;
    close $fh;

    open my $tfh, "<", $trace_path or die "open $trace_path: $!";
    my $json = decode_json(<$tfh>);
    close $tfh;

    my @words = ($text =~ /[a-z][a-z'\''-]*/gi);
    my $sentence_marks = () = $text =~ /[.!?]/g;
    my $metrics = $json->{metrics} // {};
    my $generation = $json->{generation} // {};
    my @generated_tokens = @{$generation->{tokens} // []};
    my $rejects = (($metrics->{total_rejected_candidates} // {})->{local_frequency}) // 0;
    my $stopped = $generation->{stopped_on_sentence_terminal} ? 1 : 0;
    my $repeated_bigram = $metrics->{repeated_bigram_count} // 0;
    my $repeated_trigram = $metrics->{repeated_trigram_count} // 0;
    my $cruft = () = $text =~ /\b(?:nsrlpageboundary|category|file|http|isbn|redirect|template|stub)\b/ig;
    my $bad_sentence_start =
      $sentence_index > 1 && $text =~ /^(?:and|or|but|of|like|because|which|that)\b|^[,;:]/i;
    my ($source_exact_span, $source_longest_run, $source_bigram_per_mille, $source_trigram_per_mille) =
      source_grounding_metrics($tokens_path, \@generated_tokens);

    my $word_count = scalar(@words);
    my $score = 0;
    $score += $stopped ? 100_000 : -100_000;
    $score += ($word_count >= $min_words && $word_count <= $max_words)
      ? 5_000
      : -abs($word_count - (($min_words + $max_words) / 2)) * 100;
    $score += ($word_count >= 8 && $word_count <= 20) ? 2_000 : 0;
    $score -= $word_count < 6 ? 1_500 : 0;
    $score += $sentence_marks == 1 ? 1_000 : -abs($sentence_marks - 1) * 1_500;
    $score -= $repeated_bigram * 2_000;
    $score -= $repeated_trigram * 4_000;
    $score -= int($rejects / 2);
    $score -= $cruft * 5_000;
    $score -= $bad_sentence_start ? 3_000 : 0;
    $score += $source_exact_span ? 4_000 : 0;
    $score += $source_longest_run * 120;
    $score += int($source_trigram_per_mille * 2);
    $score -= (!$source_exact_span && $source_trigram_per_mille < 900) ? 3_000 : 0;

    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+\z//g;
    $text =~ s/\t/ /g;

    print join("\t",
      $sentence_index,
      $candidate_index,
      $seed,
      int($score),
      $stopped,
      $word_count,
      $sentence_marks,
      $rejects,
      $repeated_bigram,
      $repeated_trigram,
      $source_exact_span,
      $source_longest_run,
      $source_bigram_per_mille,
      $source_trigram_per_mille,
      $fragment_path,
      $trace_path,
      $text,
    ), "\n";

    sub source_grounding_metrics {
      my ($tokens_path, $generated_tokens) = @_;
      return (0, 0, 0, 0) unless @$generated_tokens;

      open my $tfh, "<:raw", $tokens_path or die "open $tokens_path: $!";
      local $/;
      my $bytes = <$tfh>;
      close $tfh;
      die "topic token stream must be u16 little-endian\n" if length($bytes) % 2;
      my @source = unpack("v*", $bytes);

      my $exact = source_contains(\@source, $generated_tokens, 0, scalar(@$generated_tokens)) ? 1 : 0;
      my $longest = 0;
      for (my $len = scalar(@$generated_tokens); $len >= 1; $len--) {
        my $found = 0;
        for (my $start = 0; $start + $len <= scalar(@$generated_tokens); $start++) {
          if (source_contains(\@source, $generated_tokens, $start, $len)) {
            $found = 1;
            last;
          }
        }
        if ($found) {
          $longest = $len;
          last;
        }
      }

      my $bigram = ngram_coverage_per_mille(\@source, $generated_tokens, 2);
      my $trigram = ngram_coverage_per_mille(\@source, $generated_tokens, 3);
      return ($exact, $longest, $bigram, $trigram);
    }

    sub source_contains {
      my ($source, $needle, $start, $len) = @_;
      return 0 if $len <= 0 || $len > @$source;
      SOURCE:
      for (my $index = 0; $index + $len <= @$source; $index++) {
        for (my $offset = 0; $offset < $len; $offset++) {
          next SOURCE if $source->[$index + $offset] != $needle->[$start + $offset];
        }
        return 1;
      }
      return 0;
    }

    sub ngram_coverage_per_mille {
      my ($source, $generated_tokens, $order) = @_;
      my $total = scalar(@$generated_tokens) - $order + 1;
      return 0 if $total <= 0;
      my %seen;
      for (my $index = 0; $index + $order <= @$source; $index++) {
        my @key = @{$source}[$index .. $index + $order - 1];
        $seen{join(",", @key)} = 1;
      }
      my $hits = 0;
      for (my $index = 0; $index + $order <= @$generated_tokens; $index++) {
        my @key = @{$generated_tokens}[$index .. $index + $order - 1];
        $hits++ if $seen{join(",", @key)};
      }
      return int(($hits * 1000 + int($total / 2)) / $total);
    }
  ' "$sentence_index" "$candidate_index" "$seed" "$fragment" "$trace" "$tokens" "$min_words" "$max_words"
}

append_fragment() {
  local fragment="$1"
  local tmp="$out_dir/.paragraph.tmp"
  perl -Mstrict -Mwarnings -0777 -e '
    my ($paragraph_path, $fragment_path, $out_path) = @ARGV;
    open my $pfh, "<", $paragraph_path or die "open $paragraph_path: $!";
    my $paragraph = <$pfh>;
    close $pfh;
    open my $ffh, "<", $fragment_path or die "open $fragment_path: $!";
    my $fragment = <$ffh>;
    close $ffh;

    $paragraph =~ s/^\s+|\s+\z//g;
    $fragment =~ s/^\s+|\s+\z//g;
    if (length $fragment) {
      if ($fragment =~ /^[,.;:!?]/ || $paragraph =~ /\s\z/) {
        $paragraph .= $fragment;
      } else {
        $paragraph .= " " . $fragment;
      }
    }
    $paragraph =~ s/\s+/ /g;

    open my $ofh, ">", $out_path or die "open $out_path: $!";
    print {$ofh} $paragraph, "\n";
    close $ofh;
  ' "$paragraph_text" "$fragment" "$tmp"
  mv "$tmp" "$paragraph_text"
}

sentence_index=1
last_fragment_text="$prompt"
while [[ "$sentence_index" -le "$paragraph_sentences" ]]; do
  case "$paragraph_prompt_mode" in
    full)
      prompt_text="$(cat "$paragraph_text")"
      ;;
    last)
      prompt_text="$last_fragment_text"
      ;;
    hybrid)
      prompt_text="$prompt $last_fragment_text"
      ;;
  esac
  sentence_dir="$out_dir/candidates/sentence-$sentence_index"
  mkdir -p "$sentence_dir"
  scores="$sentence_dir/scores.tsv"
  sorted_scores="$sentence_dir/scores.sorted.tsv"
  : > "$scores"

  candidate_index=0
  while [[ "$candidate_index" -lt "$candidates_per_sentence" ]]; do
    seed=$((sample_seed + sentence_index * 1009 + candidate_index * 97))
    fragment="$sentence_dir/candidate-$candidate_index.txt"
    trace="$sentence_dir/candidate-$candidate_index.trace.json"
    generate_candidate "$sentence_index" "$candidate_index" "$seed" "$prompt_text" "$fragment" "$trace"
    score_candidate "$sentence_index" "$candidate_index" "$seed" "$fragment" "$trace" >> "$scores"
    candidate_index=$((candidate_index + 1))
  done

  sort -t $'\t' -k4,4nr "$scores" > "$sorted_scores"
  best_line="$(head -n 1 "$sorted_scores")"
  if [[ -z "$best_line" ]]; then
    break
  fi
  printf '%s\n' "$best_line" >> "$choices_tsv"

  IFS=$'\t' read -r _sentence _candidate _seed _score stopped _words _marks local_rejections _rep2 _rep3 source_exact_span source_longest_run _source_bigram _source_trigram fragment _trace _preview <<< "$best_line"
  if [[ "$stopped" != "1" || "$_words" -lt "$min_words" || "$local_rejections" -gt "$max_local_frequency_rejections" || "$source_exact_span" != "1" ]]; then
    break
  fi
  append_fragment "$fragment"
  last_fragment_text="$(perl -Mstrict -Mwarnings -0777 -e 'my $path = shift; open my $fh, "<", $path or die "open $path: $!"; my $text = <$fh>; close $fh; $text =~ s/^\s+|\s+\z//g; $text =~ s/\s+/ /g; print $text;' "$fragment")"
  sentence_index=$((sentence_index + 1))
done

perl -Mstrict -Mwarnings -MJSON::PP -0777 -e '
  my (
    $manifest, $run_id, $out_dir, $model, $tokens, $vocab, $prompt,
    $paragraph_text, $choices_tsv, $paragraph_sentences, $candidates_per_sentence,
    $max_new_tokens, $top_k, $sample_seed, $local_frequency_hard_cap,
    $paragraph_prompt_mode,
    $max_local_frequency_rejections
  ) = @ARGV;
  open my $pfh, "<", $paragraph_text or die "open $paragraph_text: $!";
  my $text = <$pfh>;
  close $pfh;
  my @words = ($text =~ /[a-z][a-z'\''-]*/gi);
  my @sentence_marks = ($text =~ /[.!?]/g);
  my $doc = {
    schema => "nsrl.simplewiki_topic_paragraph_bestof.v1",
    run_id => $run_id,
    out_dir => $out_dir,
    model => $model,
    tokens => $tokens,
    vocab => $vocab,
    prompt => $prompt,
    paragraph_text => $paragraph_text,
    choices_tsv => $choices_tsv,
    decode => {
      paragraph_sentences => int($paragraph_sentences),
      candidates_per_sentence => int($candidates_per_sentence),
      max_new_tokens => int($max_new_tokens),
      paragraph_prompt_mode => $paragraph_prompt_mode,
      top_k => int($top_k),
      sample_seed => int($sample_seed),
      local_frequency_hard_cap => int($local_frequency_hard_cap),
      max_local_frequency_rejections => int($max_local_frequency_rejections),
      require_source_exact_span => JSON::PP::true,
      stop_on_sentence_terminal => JSON::PP::true,
    },
    metrics => {
      words => scalar(@words),
      sentence_marks => scalar(@sentence_marks),
    },
  };
  open my $mfh, ">", $manifest or die "open $manifest: $!";
  print {$mfh} JSON::PP->new->canonical->pretty->encode($doc);
' "$manifest" "$run_id" "$out_dir" "$model" "$tokens" "$vocab" "$prompt" \
  "$paragraph_text" "$choices_tsv" "$paragraph_sentences" "$candidates_per_sentence" \
  "$max_new_tokens" "$top_k" "$sample_seed" "$local_frequency_hard_cap" \
  "$paragraph_prompt_mode" "$max_local_frequency_rejections"

echo "paragraph_text=$paragraph_text"
echo "choices_tsv=$choices_tsv"
echo "manifest=$manifest"
