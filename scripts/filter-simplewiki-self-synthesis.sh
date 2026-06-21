#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${1:?usage: scripts/filter-simplewiki-self-synthesis.sh RUN_DIR}"
keep_count="${KEEP_COUNT:-16}"
min_sample_chars="${MIN_SAMPLE_CHARS:-160}"
max_cruft_count="${MAX_CRUFT_COUNT:-12}"
max_spelling_runs="${MAX_SPELLING_RUNS:-8}"

prompts_path="$run_dir/prompts.txt"
manifest_path="$run_dir/manifest.json"
quality_tsv="$run_dir/quality.tsv"
kept_tsv="$run_dir/quality-kept.tsv"
filtered_corpus="$run_dir/filtered-synthetic-corpus.txt"
summary_json="$run_dir/quality-summary.json"

if [[ ! -d "$run_dir" ]]; then
  echo "missing run dir: $run_dir" >&2
  exit 1
fi
if [[ ! -f "$prompts_path" ]]; then
  echo "missing prompts: $prompts_path" >&2
  exit 1
fi

printf '%s\n' \
  "sample	score	avg_prob_q15	min_prob_q15	avg_candidates	new_tokens	byte_rejects	adjacency_rejects	chars	cruft_count	spelling_runs	word_count	distinct_words	max_word_count	kept" \
  > "$quality_tsv"

for sample_text in "$run_dir"/sample-*.txt; do
  [[ -f "$sample_text" ]] || continue
  sample_base="$(basename "$sample_text" .txt)"
  sample="${sample_base#sample-}"
  sample_trace="$run_dir/${sample_base}.trace.jsonl"
  [[ -f "$sample_trace" ]] || continue

  IFS=$'\t' read -r new_tokens avg_prob min_prob avg_candidates byte_rejects adjacency_rejects < <(
    jq -r '
      [
        .generation.new_tokens,
        (((.steps | map(.predicted_probability_q15) | add) / (.steps | length)) | floor),
        (.steps | map(.predicted_probability_q15) | min),
        (((.steps | map(.candidate_count) | add) / (.steps | length)) | floor),
        (.steps | map(.rejected_candidates.byte_fallback) | add),
        (.steps | map(.rejected_candidates.adjacency) | add)
      ] | @tsv
    ' "$sample_trace"
  )

  IFS=$'\t' read -r chars cruft_count spelling_runs word_count distinct_words max_word_count < <(
    perl -0777 -Mstrict -Mwarnings -e '
      my $text = <STDIN>;
      my $lower = lc $text;
      my $cruft = () = $lower =~ /\b(?:http|https|www|class|align|bgcolor|vcard|colspan|rowspan|style|nbsp|ndash|references)\b/g;
      my $spelled = () = $lower =~ /(?:^|[^a-z])(?:[a-z]\s+){4,}[a-z](?:[^a-z]|$)/g;
      my @words = $lower =~ /[a-z][a-z'"'"']*/g;
      my %counts;
      $counts{$_}++ for @words;
      my $max_count = 0;
      for my $count (values %counts) {
        $max_count = $count if $count > $max_count;
      }
      print join("\t", length($text), $cruft, $spelled, scalar(@words), scalar(keys %counts), $max_count), "\n";
    ' < "$sample_text"
  )

  score=$((avg_prob * 20 + avg_candidates * 25 + distinct_words * 3 - cruft_count * 180 - spelling_runs * 240 - max_word_count * 12))
  kept=1
  if (( chars < min_sample_chars )); then
    kept=0
  fi
  if (( cruft_count > max_cruft_count )); then
    kept=0
  fi
  if (( spelling_runs > max_spelling_runs )); then
    kept=0
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sample" "$score" "$avg_prob" "$min_prob" "$avg_candidates" "$new_tokens" \
    "$byte_rejects" "$adjacency_rejects" "$chars" "$cruft_count" "$spelling_runs" \
    "$word_count" "$distinct_words" "$max_word_count" "$kept" >> "$quality_tsv"
done

{
  head -n 1 "$quality_tsv"
  awk -F '\t' 'NR > 1 && $15 == 1 { print }' "$quality_tsv" \
    | sort -t $'\t' -k2,2nr -k1,1 \
    | head -n "$keep_count"
} > "$kept_tsv"

{
  echo "<|source:synthetic-simplewiki-self-filtered|>"
  echo "<|source_run:$(basename "$run_dir")|>"
  echo
} > "$filtered_corpus"

tail -n +2 "$kept_tsv" | while IFS=$'\t' read -r sample score _rest; do
  sample_index="$((10#$sample + 1))"
  prompt="$(sed -n "${sample_index}p" "$prompts_path")"
  sample_text="$run_dir/sample-${sample}.txt"
  {
    echo "<|sample:${sample}|>"
    echo "<|quality_score:${score}|>"
    echo "$prompt"
    cat "$sample_text"
    echo
    echo
  } >> "$filtered_corpus"
done

kept_actual="$(($(wc -l < "$kept_tsv" | tr -d ' ') - 1))"
sample_count="$(($(wc -l < "$quality_tsv" | tr -d ' ') - 1))"
manifest_json="null"
if [[ -f "$manifest_path" ]]; then
  manifest_json="$(jq -c . "$manifest_path")"
fi

cat > "$summary_json" <<EOF
{
  "schema": "nsrl.simplewiki_self_synthesis_quality.v1",
  "run_dir": "$run_dir",
  "sample_count": $sample_count,
  "kept_count": $kept_actual,
  "keep_count_requested": $keep_count,
  "min_sample_chars": $min_sample_chars,
  "max_cruft_count": $max_cruft_count,
  "max_spelling_runs": $max_spelling_runs,
  "quality_tsv": "$quality_tsv",
  "kept_tsv": "$kept_tsv",
  "filtered_corpus": "$filtered_corpus",
  "source_manifest": $manifest_json
}
EOF

echo "quality_tsv=$quality_tsv"
echo "kept_tsv=$kept_tsv"
echo "filtered_corpus=$filtered_corpus"
echo "summary_json=$summary_json"
echo "kept_count=$kept_actual/$sample_count"
