#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
base_dir="${BASE_DIR:-data/processed/simplewiki-expository-v1}"
boundary_dir="${BOUNDARY_DIR:-$base_dir/boundary-v1}"
topic="${TOPIC:-earth}"
out_dir="${OUT_DIR:-$base_dir/topic-${topic}-curriculum-${run_id}}"

source_tokens="${SOURCE_TOKENS:-$boundary_dir/v4096-boundary.tokens.u16}"
vocab="${VOCAB:-$boundary_dir/v4096-boundary.vocab.tsv}"
embedding="${EMBEDDING:-$boundary_dir/v4096-boundary.embedding.nsrllex}"
anchor_terms="${ANCHOR_TERMS:-earth,planet,sun,moon,space,orbit,atmosphere,water,life,soil,gravity,warming,climate,oxygen,rocks,surface,clouds,weather,carbon,nitrogen}"
prompt="${PROMPT:-the earth is an ancient planet}"

max_pages="${MAX_PAGES:-8}"
include_prev="${INCLUDE_PREV:-1}"
include_next="${INCLUDE_NEXT:-2}"
min_sentence_tokens="${MIN_SENTENCE_TOKENS:-4}"
max_sentence_tokens="${MAX_SENTENCE_TOKENS:-120}"
train_windows="${TRAIN_WINDOWS:-4096}"
seq_len="${SEQ_LEN:-8}"
batch_windows="${BATCH_WINDOWS:-8}"
lr_shift="${LR_SHIFT:-19}"
hidden_lr_shift="${HIDDEN_LR_SHIFT:-20}"
max_lr_shift="${MAX_LR_SHIFT:-24}"
hidden_dim="${HIDDEN_DIM:-32}"

max_new_tokens="${MAX_NEW_TOKENS:-128}"
top_k="${TOP_K:-32}"
sample_seed="${SAMPLE_SEED:-11}"
local_frequency_hard_cap="${LOCAL_FREQUENCY_HARD_CAP:-2}"
stop_on_sentence_terminal="${STOP_ON_SENTENCE_TERMINAL:-1}"

mkdir -p "$out_dir"

for path in "$source_tokens" "$vocab" "$embedding"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
done

topic_tokens="$out_dir/${topic}.tokens.u16"
topic_trace="$out_dir/${topic}.tokens.trace.json"
model="$out_dir/${topic}.seq${seq_len}.hidden${hidden_dim}.nsrllm"
train_trace="$out_dir/${topic}.seq${seq_len}.hidden${hidden_dim}.trace.jsonl"
guided_text="$out_dir/${topic}.guided.txt"
guided_trace="$out_dir/${topic}.guided.trace.json"
sample_text="$out_dir/${topic}.sample.txt"
sample_trace="$out_dir/${topic}.sample.trace.json"
metrics="$out_dir/metrics.tsv"
manifest="$out_dir/manifest.json"

echo "run_id=$run_id"
echo "out_dir=$out_dir"
echo "topic=$topic"
echo "anchor_terms=$anchor_terms"
echo "source_tokens=$source_tokens"
echo "vocab=$vocab"
echo "embedding=$embedding"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

date -u +"topic_extract_started_at=%Y-%m-%dT%H:%M:%SZ"
perl -Mstrict -Mwarnings -MJSON::PP -e '
  my (
    $source_tokens, $vocab_path, $out_tokens, $trace_path, $topic, $anchor_csv,
    $max_pages, $include_prev, $include_next, $min_sentence_tokens, $max_sentence_tokens
  ) = @ARGV;
  $max_pages = int($max_pages);
  $include_prev = int($include_prev);
  $include_next = int($include_next);
  $min_sentence_tokens = int($min_sentence_tokens);
  $max_sentence_tokens = int($max_sentence_tokens);

  open my $vfh, "<", $vocab_path or die "open $vocab_path: $!";
  my %lexeme_to_id;
  my %id_to_lexeme;
  while (my $line = <$vfh>) {
    chomp $line;
    next if $. == 1 && $line =~ /^token_id\t/;
    my ($id, $lexeme) = split /\t/, $line, 3;
    next unless defined $id && defined $lexeme && $id =~ /^\d+$/;
    $lexeme_to_id{$lexeme} = int($id);
    $id_to_lexeme{int($id)} = $lexeme;
  }
  close $vfh;

  my $boundary = $lexeme_to_id{nsrlpageboundary};
  die "vocab has no nsrlpageboundary token\n" unless defined $boundary;
  my %terminal;
  for my $punct (qw(. ! ?)) {
    $terminal{$lexeme_to_id{$punct}} = 1 if defined $lexeme_to_id{$punct};
  }
  for my $id (keys %id_to_lexeme) {
    $terminal{$id} = 1 if $id_to_lexeme{$id} =~ /[.!?]\z/;
  }

  my @anchor_terms = grep { length } split /,/, lc($anchor_csv);
  my %anchor_ids;
  my @missing_terms;
  for my $term (@anchor_terms) {
    $term =~ s/^\s+|\s+\z//g;
    next unless length $term;
    if (defined $lexeme_to_id{$term}) {
      $anchor_ids{$lexeme_to_id{$term}} = $term;
    } else {
      push @missing_terms, $term;
    }
  }
  die "no anchor terms found in vocab\n" unless %anchor_ids;

  open my $tfh, "<:raw", $source_tokens or die "open $source_tokens: $!";
  local $/;
  my $bytes = <$tfh>;
  close $tfh;
  die "token stream must be u16 little-endian\n" if length($bytes) % 2;
  my @tokens = unpack("v*", $bytes);

  my @pages;
  my @page;
  for my $token (@tokens) {
    if ($token == $boundary) {
      push @pages, [@page] if @page;
      @page = ();
      next;
    }
    push @page, $token;
  }
  push @pages, [@page] if @page;

  my @page_infos;
  for my $page_index (0 .. $#pages) {
    my @sentences;
    my @sentence;
    for my $token (@{$pages[$page_index]}) {
      push @sentence, $token;
      if ($terminal{$token}) {
        push @sentences, [@sentence] if @sentence;
        @sentence = ();
      }
    }
    push @sentences, [@sentence] if @sentence;

    my $anchor_hits = 0;
    my $anchor_sentences = 0;
    my @clean;
    for my $idx (0 .. $#sentences) {
      my $sentence = $sentences[$idx];
      my $len = scalar(@$sentence);
      next if $len < $min_sentence_tokens || $len > $max_sentence_tokens;
      my $hits = 0;
      for my $token (@$sentence) {
        $hits++ if exists $anchor_ids{$token};
      }
      $anchor_hits += $hits;
      $anchor_sentences++ if $hits > 0;
      push @clean, {
        index => $idx,
        tokens => $sentence,
        anchor_hits => $hits,
      };
    }
    next if $anchor_sentences == 0;
    push @page_infos, {
      page_index => $page_index,
      anchor_hits => $anchor_hits,
      anchor_sentences => $anchor_sentences,
      sentences_seen => scalar(@sentences),
      clean_sentences => \@clean,
      token_count => scalar(@{$pages[$page_index]}),
      score => $anchor_sentences * 1000 + $anchor_hits * 100 + scalar(@clean),
    };
  }

  @page_infos = sort {
    $b->{score} <=> $a->{score}
      || $b->{anchor_hits} <=> $a->{anchor_hits}
      || $a->{page_index} <=> $b->{page_index}
  } @page_infos;
  @page_infos = @page_infos[0 .. $max_pages - 1] if @page_infos > $max_pages;

  my @selected;
  my @page_details;
  for my $page (@page_infos) {
    my @clean = @{$page->{clean_sentences}};
    my %keep;
    for my $i (0 .. $#clean) {
      next unless $clean[$i]{anchor_hits} > 0;
      my $start = $i - $include_prev;
      $start = 0 if $start < 0;
      my $end = $i + $include_next;
      $end = $#clean if $end > $#clean;
      $keep{$_} = 1 for $start .. $end;
    }
    my $before = scalar(@selected);
    push @selected, $boundary;
    for my $i (sort { $a <=> $b } keys %keep) {
      push @selected, @{$clean[$i]{tokens}};
    }
    push @page_details, {
      page_index => $page->{page_index},
      score => $page->{score},
      anchor_hits => $page->{anchor_hits},
      anchor_sentences_kept => $page->{anchor_sentences},
      sentences_seen => $page->{sentences_seen},
      sentences_kept => scalar(keys %keep),
      source_tokens => $page->{token_count},
      tokens_kept => scalar(@selected) - $before,
    };
  }
  push @selected, $boundary if @selected;

  my $hash = 0x811c9dc5;
  for my $token (@selected) {
    $hash ^= ($token & 0xff);
    $hash = ($hash * 0x01000193) & 0xffffffff;
    $hash ^= (($token >> 8) & 0xff);
    $hash = ($hash * 0x01000193) & 0xffffffff;
  }

  open my $ofh, ">:raw", $out_tokens or die "open $out_tokens: $!";
  print {$ofh} pack("v*", @selected);
  close $ofh;

  my $doc = {
    schema => "nsrl.topic_curriculum_trace.v1",
    authority => "deterministic_topic_span_filter",
    topic => $topic,
    source_tokens => $source_tokens,
    vocab => $vocab_path,
    output_tokens => $out_tokens,
    boundary_token => $boundary,
    anchor_terms => \@anchor_terms,
    missing_anchor_terms => \@missing_terms,
    matched_anchor_terms => [sort values %anchor_ids],
    source_pages => scalar(@pages),
    selected_pages => scalar(@page_infos),
    selected_tokens => scalar(@selected),
    selected_token_hash_fnv1a32 => sprintf("0x%08x", $hash),
    rules => {
      keep => "anchor sentence plus neighboring clean sentences",
      include_prev => $include_prev,
      include_next => $include_next,
      sentence_len => "$min_sentence_tokens..$max_sentence_tokens",
      page_rank => "anchor sentence count, anchor hits, clean sentence count",
    },
    page_details => \@page_details,
  };
  open my $jfh, ">", $trace_path or die "open $trace_path: $!";
  print {$jfh} JSON::PP->new->canonical->pretty->encode($doc);
  close $jfh;
' "$source_tokens" "$vocab" "$topic_tokens" "$topic_trace" "$topic" "$anchor_terms" \
  "$max_pages" "$include_prev" "$include_next" "$min_sentence_tokens" "$max_sentence_tokens"

date -u +"training_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens "$topic_tokens" \
  --vocab "$vocab" \
  --model "$embedding" \
  --model-out "$model" \
  --trace "$train_trace" \
  --seq-len "$seq_len" \
  --lexeme-context-features ordered \
  --stride 1 \
  --max-windows "$train_windows" \
  --batch-windows "$batch_windows" \
  --epochs 1 \
  --lr-shift "$lr_shift" \
  --lr-shift-decay-windows "$((train_windows / 2))" \
  --lr-shift-decay-step 1 \
  --max-lr-shift "$max_lr_shift" \
  --max-weight-delta 1 \
  --target-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile prose-aware \
  --lexeme-hidden-dim "$hidden_dim" \
  --lexeme-hidden-lr-shift "$hidden_lr_shift" \
  --max-hidden-weight-delta 1

generate_sample() {
  local label="$1"
  local strategy="$2"
  local seed="$3"
  local text_out="$4"
  local trace_out="$5"
  local stop_args=()
  if [[ "$stop_on_sentence_terminal" != "0" ]]; then
    stop_args+=(--stop-on-sentence-terminal)
  fi

  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$model" \
    --vocab "$vocab" \
    --tokens "$topic_tokens" \
    --prompt "$prompt" \
    --max-new-tokens "$max_new_tokens" \
    --decode "$strategy" \
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
    "${stop_args[@]}" \
    --text-out "$text_out" \
    --trace "$trace_out"
}

date -u +"generation_started_at=%Y-%m-%dT%H:%M:%SZ"
generate_sample guided greedy 0 "$guided_text" "$guided_trace"
generate_sample sample sample "$sample_seed" "$sample_text" "$sample_trace"

printf '%s\n' "sample	words	unique_words	max_word_count	sentences	repeated_bigram_count	repeated_trigram_count	local_frequency_rejections	generated_tokens	stopped_on_sentence_terminal	text" > "$metrics"
for spec in "guided:$guided_text:$guided_trace" "sample:$sample_text:$sample_trace"; do
  IFS=: read -r label text_path trace_path <<< "$spec"
  perl -Mstrict -Mwarnings -MJSON::PP -e '
    my ($label, $text_path, $trace_path) = @ARGV;
    open my $xfh, "<", $text_path or die "open $text_path: $!";
    local $/;
    my $text = lc <$xfh>;
    close $xfh;
    open my $tfh, "<", $trace_path or die "open $trace_path: $!";
    my $json = decode_json(<$tfh>);
    close $tfh;
    my @words = ($text =~ /[a-z][a-z'\''-]*/g);
    my %counts;
    $counts{$_}++ for @words;
    my $max_word_count = 0;
    for my $count (values %counts) {
      $max_word_count = $count if $count > $max_word_count;
    }
    my $m = $json->{metrics} // {};
    my $g = $json->{generation} // {};
    my $rejects = (($m->{total_rejected_candidates} // {})->{local_frequency}) // 0;
    my @sentences = ($text =~ /[.!?]/g);
    print join("\t",
      $label,
      scalar(@words),
      scalar(keys %counts),
      $max_word_count,
      scalar(@sentences),
      $m->{repeated_bigram_count} // 0,
      $m->{repeated_trigram_count} // 0,
      $rejects,
      $g->{new_tokens} // 0,
      ($g->{stopped_on_sentence_terminal} ? 1 : 0),
      $text_path
    ), "\n";
  ' "$label" "$text_path" "$trace_path" >> "$metrics"
done

perl -Mstrict -Mwarnings -MJSON::PP -e '
  my (
    $manifest, $run_id, $topic, $out_dir, $source_tokens, $vocab, $embedding,
    $topic_tokens, $model, $topic_trace, $train_trace, $metrics, $anchor_terms,
    $max_pages, $include_prev, $include_next, $train_windows, $seq_len,
    $batch_windows, $lr_shift, $hidden_lr_shift, $hidden_dim, $local_hard_cap,
    $stop_on_sentence_terminal
  ) = @ARGV;
  my $doc = {
    schema => "nsrl.simplewiki_topic_curriculum_run.v1",
    run_id => $run_id,
    topic => $topic,
    out_dir => $out_dir,
    source_tokens => $source_tokens,
    vocab => $vocab,
    embedding => $embedding,
    topic_tokens => $topic_tokens,
    model => $model,
    topic_trace => $topic_trace,
    train_trace => $train_trace,
    metrics => $metrics,
    anchor_terms => [grep { length } split /,/, $anchor_terms],
    extraction => {
      max_pages => int($max_pages),
      include_prev => int($include_prev),
      include_next => int($include_next),
    },
    training => {
      train_windows => int($train_windows),
      seq_len => int($seq_len),
      batch_windows => int($batch_windows),
      lr_shift => int($lr_shift),
      hidden_lr_shift => int($hidden_lr_shift),
      hidden_dim => int($hidden_dim),
    },
    decode => {
      local_frequency_hard_cap => int($local_hard_cap),
      stop_on_sentence_terminal => int($stop_on_sentence_terminal),
    },
  };
  open my $fh, ">", $manifest or die "open $manifest: $!";
  print {$fh} JSON::PP->new->canonical->pretty->encode($doc);
' "$manifest" "$run_id" "$topic" "$out_dir" "$source_tokens" "$vocab" "$embedding" \
  "$topic_tokens" "$model" "$topic_trace" "$train_trace" "$metrics" "$anchor_terms" \
  "$max_pages" "$include_prev" "$include_next" "$train_windows" "$seq_len" \
  "$batch_windows" "$lr_shift" "$hidden_lr_shift" "$hidden_dim" "$local_frequency_hard_cap" \
  "$stop_on_sentence_terminal"

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "topic_tokens=$topic_tokens"
echo "model=$model"
echo "metrics=$metrics"
echo "manifest=$manifest"
