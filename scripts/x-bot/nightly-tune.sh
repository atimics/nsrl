#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

AWS_REGION="${AWS_REGION:-us-east-1}"
MODEL_S3_URI="${X_BOT_MODEL_S3_URI:-}"
CONTEXT_ARCHIVE_S3_URI="${X_BOT_CONTEXT_ARCHIVE_S3_URI:-}"
HISTORY_S3_URI="${X_BOT_MODEL_HISTORY_S3_URI:-}"
TUNE_DAY="${X_BOT_TUNE_DAY:-}"
WORK_DIR="${X_BOT_NIGHTLY_WORK_DIR:-$SCRIPT_DIR/build/nightly-tune}"
NSRL_TRAIN_BIN="${NSRL_TRAIN_BIN:-target/release/nsrl-train}"
NSRL_CORPUS_BIN="${NSRL_CORPUS_BIN:-target/release/nsrl-corpus}"
MAX_WINDOWS="${X_BOT_NIGHTLY_MAX_WINDOWS:-512}"
LR_SHIFT="${X_BOT_NIGHTLY_LR_SHIFT:-23}"
MAX_LR_SHIFT="${X_BOT_NIGHTLY_MAX_LR_SHIFT:-25}"
CONTEXT_REPEAT_COUNT="${X_BOT_NIGHTLY_CONTEXT_REPEAT_COUNT:-2}"
MIN_CONTEXT_EVENTS="${X_BOT_NIGHTLY_MIN_CONTEXT_EVENTS:-1}"
MIN_PASSING_SAMPLES="${X_BOT_NIGHTLY_MIN_PASSING_SAMPLES:-2}"
PUBLISH="${X_BOT_NIGHTLY_PUBLISH:-true}"

if [[ -z "$MODEL_S3_URI" ]]; then
  echo "Set X_BOT_MODEL_S3_URI to the production model bundle prefix" >&2
  exit 1
fi
if [[ -z "$CONTEXT_ARCHIVE_S3_URI" ]]; then
  echo "Set X_BOT_CONTEXT_ARCHIVE_S3_URI to the daily context archive prefix" >&2
  exit 1
fi
if [[ -z "$TUNE_DAY" ]]; then
  TUNE_DAY="$(python3 - <<'PY'
import datetime as dt
print((dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat())
PY
)"
fi
if [[ -z "$HISTORY_S3_URI" ]]; then
  HISTORY_S3_URI="${MODEL_S3_URI%/}/nightly/$TUNE_DAY"
fi

mkdir -p "$WORK_DIR"
RUN_DIR="$WORK_DIR/$TUNE_DAY"
ARCHIVE_DIR="$RUN_DIR/archive"
BASE_DIR="$RUN_DIR/base"
CANDIDATE_DIR="$RUN_DIR/candidate"
mkdir -p "$ARCHIVE_DIR" "$BASE_DIR" "$CANDIDATE_DIR"

write_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >> "$GITHUB_OUTPUT"
  fi
}

write_output published false
write_output candidate_ready false
write_output tune_day "$TUNE_DAY"
write_output candidate_dir "$CANDIDATE_DIR"
write_output model_s3_uri "$MODEL_S3_URI"
write_output history_s3_uri "$HISTORY_S3_URI"

echo "tune_day=$TUNE_DAY"
echo "model_s3_uri=$MODEL_S3_URI"
echo "context_archive_s3_uri=$CONTEXT_ARCHIVE_S3_URI"
echo "history_s3_uri=$HISTORY_S3_URI"

aws s3 sync "$CONTEXT_ARCHIVE_S3_URI/$TUNE_DAY/" "$ARCHIVE_DIR/" --region "$AWS_REGION" --exclude "*" --include "*.json" || true
event_count="$(find "$ARCHIVE_DIR" -type f -name '*.json' | wc -l | tr -d ' ')"
echo "context_events=$event_count"
write_output context_events "$event_count"
if [[ "$event_count" -lt "$MIN_CONTEXT_EVENTS" ]]; then
  echo "not enough context events; skipping nightly tune"
  exit 0
fi

aws s3 cp "$MODEL_S3_URI/v4096.nsrllm" "$BASE_DIR/v4096.nsrllm" --region "$AWS_REGION"
aws s3 cp "$MODEL_S3_URI/v4096.vocab.tsv" "$BASE_DIR/v4096.vocab.tsv" --region "$AWS_REGION"
aws s3 cp "$MODEL_S3_URI/v4096.tokens.u16" "$BASE_DIR/v4096.tokens.u16" --region "$AWS_REGION"

python3 - "$ARCHIVE_DIR" "$CANDIDATE_DIR/context-corpus.txt" "$CONTEXT_REPEAT_COUNT" <<'PY'
import json
import pathlib
import re
import sys

archive_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
repeat = int(sys.argv[3])
public_mention_re = re.compile(r"@\w+")

def clean(text):
    text = re.sub(r"https?://\S+", " ", text or "")
    text = public_mention_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"^[,.!?;:]\s*", "", text)
    return text

events = []
for path in sorted(archive_dir.glob("*.json")):
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    mention = event.get("mention") or {}
    reply = event.get("reply") or {}
    mention_text = clean(mention.get("text", ""))
    reply_text = clean(reply.get("text", ""))
    if len(mention_text) < 8:
        continue
    events.append((mention_text, reply_text))

with out_path.open("w", encoding="utf-8") as out:
    out.write("<|crowley-bard-nightly-context-v1|>\n")
    for _ in range(max(1, repeat)):
        for mention_text, reply_text in events:
            out.write("\n<|timeline-signal|>\n")
            out.write(f"{mention_text}\n")
            if reply_text:
                out.write("<|crowley-bard-reply|>\n")
                out.write(f"{reply_text}\n")
    out.write("\n<|voice-anchor|>\n")
    out.write("Prophet of sweet errors. I drink Blake, mutter Crowley, and sing WikiBard in integer hymn.\n")
    out.write("The bot is awake, but only in the way a candle is awake.\n")

print(len(events))
PY

if [[ ! -s "$CANDIDATE_DIR/context-corpus.txt" ]]; then
  echo "context corpus is empty; skipping nightly tune"
  exit 0
fi

"$NSRL_CORPUS_BIN" lexeme-tokenize-fixed-vocab \
  --corpus "$CANDIDATE_DIR/context-corpus.txt" \
  --vocab "$BASE_DIR/v4096.vocab.tsv" \
  --tokens-out "$CANDIDATE_DIR/context.tokens.u16" \
  --trace "$CANDIDATE_DIR/context.tokens.trace.jsonl" \
  --seq-len 32 \
  --stride 1

python3 - "$BASE_DIR/v4096.tokens.u16" "$CANDIDATE_DIR/context.tokens.u16" "$CANDIDATE_DIR/v4096.tokens.u16" "$CONTEXT_REPEAT_COUNT" <<'PY'
import pathlib
import sys

base = pathlib.Path(sys.argv[1]).read_bytes()
context = pathlib.Path(sys.argv[2]).read_bytes()
out_path = pathlib.Path(sys.argv[3])
repeat = int(sys.argv[4])
with out_path.open("wb") as out:
    out.write(base)
    for _ in range(max(1, repeat)):
        out.write(context)
PY

"$NSRL_TRAIN_BIN" \
  --mode lexeme-softmax \
  --tokens "$CANDIDATE_DIR/context.tokens.u16" \
  --vocab "$BASE_DIR/v4096.vocab.tsv" \
  --model "$BASE_DIR/v4096.nsrllm" \
  --model-out "$CANDIDATE_DIR/v4096.nsrllm" \
  --trace "$CANDIDATE_DIR/nightly-tune.trace.jsonl" \
  --seq-len 8 \
  --stride 1 \
  --max-windows "$MAX_WINDOWS" \
  --epochs 1 \
  --lr-shift "$LR_SHIFT" \
  --max-lr-shift "$MAX_LR_SHIFT" \
  --max-weight-delta 1 \
  --target-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware

cp "$BASE_DIR/v4096.vocab.tsv" "$CANDIDATE_DIR/v4096.vocab.tsv"

PROMPTS=(
  "what does the timeline want"
  "reply to the feed"
  "the omen today is"
  "the crowd says"
)
: > "$CANDIDATE_DIR/samples.md"
sample_index=0
for prompt in "${PROMPTS[@]}"; do
  sample_index=$((sample_index + 1))
  text_out="$CANDIDATE_DIR/sample-$sample_index.txt"
  trace_out="$CANDIDATE_DIR/sample-$sample_index.trace.jsonl"
  "$NSRL_TRAIN_BIN" \
    --mode lexeme-generate \
    --model "$CANDIDATE_DIR/v4096.nsrllm" \
    --vocab "$CANDIDATE_DIR/v4096.vocab.tsv" \
    --tokens "$CANDIDATE_DIR/v4096.tokens.u16" \
    --prompt "$prompt" \
    --max-new-tokens 48 \
    --decode-profile coherent-prose \
    --sample-seed "$((177 + sample_index))" \
    --top-k 12 \
    --corpus-prior \
    --corpus-prior-logit-shift 7 \
    --corpus-prior-order 2 \
    --repeat-window 80 \
    --repeat-penalty-shift 3 \
    --max-repeat-run 2 \
    --no-repeat-ngram 3 \
    --generated-only \
    --stop-on-sentence-terminal \
    --text-out "$text_out" \
    --trace "$trace_out" >/dev/null
  {
    printf '### sample-%02d\n\n' "$sample_index"
    printf 'prompt: `%s`\n\n' "$prompt"
    cat "$text_out"
    printf '\n\n'
  } >> "$CANDIDATE_DIR/samples.md"
done

python3 - "$CANDIDATE_DIR" "$MIN_PASSING_SAMPLES" <<'PY'
import pathlib
import re
import sys

candidate_dir = pathlib.Path(sys.argv[1])
min_passing = int(sys.argv[2])
weak_start_re = re.compile(r"^(?:against|and|but|down|face|out|thee|well|whose)\b[:;,\s]*", re.I)
broken_phrase_re = re.compile(r"\b(?:by came|brain-indeed|come would|mouths sing ha)\b", re.I)
dangling_end_re = re.compile(r"\b(?:a|an|and|as|but|for|from|in|my|of|or|our|than|that|the|their|thy|to|which|who|whose|with|your)[.!?]?$", re.I)
stopwords = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "get",
    "have",
    "how",
    "into",
    "just",
    "like",
    "not",
    "our",
    "out",
    "the",
    "their",
    "there",
    "this",
    "that",
    "they",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}

def public_tweet_score(text):
    text = re.sub(r"^(?:out|output|reply|tweet)\s*:\s*", "", text.strip(), flags=re.I)
    words = re.findall(r"[A-Za-z][A-Za-z']*", text.lower())
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    max_repeat = max(counts.values(), default=0)
    unique_ratio = len(counts) / max(1, len(words))
    content_words = [word for word in words if len(word) > 2 and word.strip("'") not in stopwords]
    content_ratio = len(content_words) / max(1, len(words))
    heavy_punctuation = len(re.findall(r"[,;:]", text))
    sentence_count = len(re.findall(r"[.!?]", text))
    avg_word_len = sum(len(word.strip("'")) for word in words) / max(1, len(words))
    reasons = []
    score = 50
    if not text:
        return 0, ["empty"], words, unique_ratio, content_ratio, max_repeat, heavy_punctuation, sentence_count
    if "@" in text:
        score -= 100
        reasons.append("contains handle")
    if "http" in text.lower():
        score -= 100
        reasons.append("contains url")
    if len(text) < 32:
        score -= 24
        reasons.append("too short")
    elif len(text) <= 180:
        score += 8
        reasons.append("good length")
    elif len(text) > 230:
        score -= 18
        reasons.append("too long")
    else:
        score -= 6
        reasons.append("long")
    if len(words) < 6:
        score -= 18
        reasons.append("too few words")
    elif len(words) <= 32:
        score += 6
        reasons.append("readable word count")
    if 8 <= len(words) <= 24:
        score += 8
        reasons.append("compact thought")
    if unique_ratio < 0.45:
        score -= 18
        reasons.append("repetitive")
    elif unique_ratio >= 0.7 and len(words) >= 6:
        score += 6
        reasons.append("varied")
    if 0.35 <= content_ratio <= 0.75:
        score += 8
        reasons.append("balanced content")
    elif content_ratio < 0.25:
        score -= 10
        reasons.append("thin content")
    elif content_ratio > 0.85 and len(words) > 10:
        score -= 4
        reasons.append("overpacked")
    if heavy_punctuation <= 2:
        score += 6
        reasons.append("clean punctuation")
    elif heavy_punctuation > 4:
        score -= 10
        reasons.append("overpunctuated")
    if sentence_count == 1:
        score += 6
        reasons.append("single complete thought")
    elif sentence_count > 2:
        score -= 8
        reasons.append("too many sentence breaks")
    if 3.2 <= avg_word_len <= 6.2:
        score += 3
        reasons.append("readable word shape")
    if max_repeat > 3:
        score -= 10
        reasons.append("word repeats")
    if weak_start_re.search(text):
        score -= 12
        reasons.append("weak opening")
    if broken_phrase_re.search(text):
        score -= 18
        reasons.append("broken phrase")
    if dangling_end_re.search(text):
        score -= 12
        reasons.append("dangling ending")
    if text and text[-1] in ".!?":
        score += 4
        reasons.append("complete sentence")
    return max(0, min(100, score)), reasons, words, unique_ratio, content_ratio, max_repeat, heavy_punctuation, sentence_count

passing = 0
reports = []
for path in sorted(candidate_dir.glob("sample-*.txt")):
    text = path.read_text(encoding="utf-8").strip()
    (
        score,
        reasons,
        words,
        unique_ratio,
        content_ratio,
        max_repeat,
        heavy_punctuation,
        sentence_count,
    ) = public_tweet_score(text)
    ok = score >= 48
    passing += int(ok)
    reports.append({
        "sample": path.name,
        "ok": ok,
        "quality_score": score,
        "reasons": reasons,
        "chars": len(text),
        "words": len(words),
        "unique_ratio": round(unique_ratio, 3),
        "content_ratio": round(content_ratio, 3),
        "max_repeat": max_repeat,
        "heavy_punctuation": heavy_punctuation,
        "sentence_count": sentence_count,
        "text": text,
    })

gate_path = candidate_dir / "quality-gate.json"
gate_path.write_text(__import__("json").dumps({
    "schema": "nsrl.x_bot.nightly_quality_gate.v1",
    "passing": passing,
    "min_passing": min_passing,
    "samples": reports,
}, indent=2) + "\n", encoding="utf-8")
print(gate_path.read_text(encoding="utf-8"))
if passing < min_passing:
    raise SystemExit(2)
PY

cat > "$CANDIDATE_DIR/manifest.json" <<EOF
{
  "schema": "nsrl.x_bot.nightly_tune_manifest.v1",
  "tune_day": "$TUNE_DAY",
  "context_events": $event_count,
  "base_model_s3_uri": "$MODEL_S3_URI",
  "history_s3_uri": "$HISTORY_S3_URI",
  "max_windows": $MAX_WINDOWS,
  "lr_shift": $LR_SHIFT,
  "max_lr_shift": $MAX_LR_SHIFT,
  "context_repeat_count": $CONTEXT_REPEAT_COUNT
}
EOF

if [[ "$PUBLISH" == "true" ]]; then
  aws s3 cp "$CANDIDATE_DIR/v4096.nsrllm" "$HISTORY_S3_URI/v4096.nsrllm" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/v4096.vocab.tsv" "$HISTORY_S3_URI/v4096.vocab.tsv" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/v4096.tokens.u16" "$HISTORY_S3_URI/v4096.tokens.u16" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/manifest.json" "$HISTORY_S3_URI/manifest.json" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/samples.md" "$HISTORY_S3_URI/samples.md" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/quality-gate.json" "$HISTORY_S3_URI/quality-gate.json" --region "$AWS_REGION"

  aws s3 cp "$CANDIDATE_DIR/v4096.nsrllm" "$MODEL_S3_URI/v4096.nsrllm" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/v4096.vocab.tsv" "$MODEL_S3_URI/v4096.vocab.tsv" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/v4096.tokens.u16" "$MODEL_S3_URI/v4096.tokens.u16" --region "$AWS_REGION"
  aws s3 cp "$CANDIDATE_DIR/manifest.json" "$MODEL_S3_URI/latest-nightly-manifest.json" --region "$AWS_REGION"
  write_output published true
  echo "published=true"
else
  echo "publish=false; candidate bundle left local at $CANDIDATE_DIR"
fi

write_output candidate_ready true
write_output candidate_dir "$CANDIDATE_DIR"
write_output history_s3_uri "$HISTORY_S3_URI"
