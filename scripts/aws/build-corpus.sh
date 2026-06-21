#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Build a versioned NSRL corpus dataset from S3 raw inputs.

Required:
  NSRL_S3_URI=s3://bucket/prefix

Common knobs:
  NSRL_DATASET_NAME=wikibard
  NSRL_DATASET_VERSION=20260621T000000Z
  NSRL_RAW_S3_URI=s3://bucket/prefix/corpus/raw/wikibard
  NSRL_TOKEN_KIND=both              # byte | lexeme | both
  NSRL_TEXT_CLEAN_PROFILE=mixed     # copy | gutenberg | mixed
  NSRL_MAX_VOCAB=4096
  NSRL_LEXEME_VOCAB_PROFILE=balanced
  NSRL_LEXEME_FREQUENCY_CAP=4096
  NSRL_FROZEN_VOCAB_S3_URI=s3://bucket/path/golden.vocab.tsv

Outputs:
  Local: data/aws-corpus/<dataset>/<version>/
  S3:    $NSRL_S3_URI/corpus/datasets/<dataset>/<version>/
USAGE
  exit 0
fi

if [[ -z "${NSRL_S3_URI:-}" ]]; then
  echo "NSRL_S3_URI is required, e.g. s3://my-bucket/nsrl" >&2
  exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

dataset="${NSRL_DATASET_NAME:-wikibard}"
version="${NSRL_DATASET_VERSION:-$(date -u +%Y%m%dT%H%M%SZ)}"
s3_uri="${NSRL_S3_URI%/}"
raw_s3_uri="${NSRL_RAW_S3_URI:-${s3_uri}/corpus/raw/${dataset}}"
dataset_s3_uri="${NSRL_DATASET_S3_URI:-${s3_uri}/corpus/datasets/${dataset}/${version}}"
work_root="${NSRL_CORPUS_WORK_ROOT:-data/aws-corpus}"
work_dir="${work_root}/${dataset}/${version}"
raw_dir="${work_dir}/raw"
clean_dir="${work_dir}/clean"
token_dir="${work_dir}/tokens"
trace_dir="${work_dir}/traces"
manifest="${work_dir}/manifest.json"
corpus="${clean_dir}/${dataset}-${version}.corpus.txt"
clean_profile="${NSRL_TEXT_CLEAN_PROFILE:-mixed}"
token_kind="${NSRL_TOKEN_KIND:-both}"
frozen_vocab="${NSRL_FROZEN_VOCAB:-}"
simplewiki_extra=()
token_extra=()
if [[ -n "${NSRL_MAX_SIMPLEWIKI_PAGES:-}" ]]; then
  simplewiki_extra=(--max-simplewiki-pages "$NSRL_MAX_SIMPLEWIKI_PAGES")
fi
if [[ -n "${NSRL_TOKEN_MAX_WINDOWS:-}" ]]; then
  token_extra=(--max-windows "$NSRL_TOKEN_MAX_WINDOWS")
fi

mkdir -p "$raw_dir" "$clean_dir" "$token_dir" "$trace_dir"

cargo build --release -p nsrl-corpus

if [[ -z "$frozen_vocab" && -n "${NSRL_FROZEN_VOCAB_S3_URI:-}" ]]; then
  frozen_vocab="${work_dir}/frozen.vocab.tsv"
  aws s3 cp "$NSRL_FROZEN_VOCAB_S3_URI" "$frozen_vocab" --only-show-errors
fi

aws s3 sync "$raw_s3_uri" "$raw_dir" --only-show-errors

clean_text_file() {
  local src="$1"
  local dst="$2"
  local base
  base="$(basename "$src")"
  case "$clean_profile" in
    copy)
      cp "$src" "$dst"
      ;;
    gutenberg)
      cargo run --release -q -p nsrl-corpus -- clean-gutenberg --corpus "$src" --out "$dst"
      ;;
    mixed)
      if [[ "$base" == *gutenberg* || "$base" == *shakespeare* ]]; then
        cargo run --release -q -p nsrl-corpus -- clean-gutenberg --corpus "$src" --out "$dst"
      else
        cp "$src" "$dst"
      fi
      ;;
    *)
      echo "unknown NSRL_TEXT_CLEAN_PROFILE=$clean_profile" >&2
      exit 2
      ;;
  esac
}

found=0
while IFS= read -r -d '' path; do
  found=1
  rel="${path#${raw_dir}/}"
  safe="${rel//\//__}"
  case "$path" in
    *.xml)
      cargo run --release -q -p nsrl-corpus -- extract-simplewiki \
        --simplewiki-xml "$path" \
        --out "${clean_dir}/${safe%.xml}.simplewiki.txt" \
        --trace "${trace_dir}/${safe%.xml}.simplewiki.trace.jsonl" \
        "${simplewiki_extra[@]}"
      ;;
    *.xml.bz2)
      if ! command -v bunzip2 >/dev/null 2>&1; then
        echo "bunzip2 is required for $path" >&2
        exit 2
      fi
      bunzip2 -c "$path" | cargo run --release -q -p nsrl-corpus -- extract-simplewiki \
        --simplewiki-xml - \
        --out "${clean_dir}/${safe%.xml.bz2}.simplewiki.txt" \
        --trace "${trace_dir}/${safe%.xml.bz2}.simplewiki.trace.jsonl" \
        "${simplewiki_extra[@]}"
      ;;
    *.txt)
      clean_text_file "$path" "${clean_dir}/${safe%.txt}.clean.txt"
      ;;
    *)
      echo "skipping unsupported raw input: $path" >&2
      ;;
  esac
done < <(find "$raw_dir" -type f -print0 | sort -z)

if [[ "$found" -eq 0 ]]; then
  echo "no raw files found under $raw_s3_uri" >&2
  exit 2
fi

: > "$corpus"
while IFS= read -r -d '' path; do
  printf '<|source:%s|>\n' "$(basename "$path")" >> "$corpus"
  cat "$path" >> "$corpus"
  printf '\n\n' >> "$corpus"
done < <(find "$clean_dir" -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)

byte_tokens="${token_dir}/${dataset}-${version}.tokens.u8"
byte_trace="${trace_dir}/${dataset}-${version}.tokens.trace.jsonl"
lexeme_tokens="${token_dir}/${dataset}-${version}.lexeme-v${NSRL_MAX_VOCAB:-4096}.tokens.u16"
lexeme_vocab="${token_dir}/${dataset}-${version}.lexeme-v${NSRL_MAX_VOCAB:-4096}.vocab.tsv"
lexeme_trace="${trace_dir}/${dataset}-${version}.lexeme-v${NSRL_MAX_VOCAB:-4096}.tokens.trace.jsonl"

if [[ "$token_kind" == "byte" || "$token_kind" == "both" ]]; then
  cargo run --release -q -p nsrl-corpus -- tokenize \
    --corpus "$corpus" \
    --tokens-out "$byte_tokens" \
    --trace "$byte_trace" \
    --seq-len "${NSRL_BYTE_SEQ_LEN:-4}" \
    --stride "${NSRL_BYTE_STRIDE:-1}" \
    --text-profile "${NSRL_BYTE_TEXT_PROFILE:-identity}" \
    --preview-tokens "${NSRL_PREVIEW_TOKENS:-32}" \
    "${token_extra[@]}"
fi

if [[ "$token_kind" == "lexeme" || "$token_kind" == "both" ]]; then
  if [[ -n "$frozen_vocab" ]]; then
    cargo run --release -q -p nsrl-corpus -- lexeme-tokenize-fixed-vocab \
      --corpus "$corpus" \
      --vocab "$frozen_vocab" \
      --tokens-out "$lexeme_tokens" \
      --trace "$lexeme_trace" \
      --seq-len "${NSRL_LEXEME_SEQ_LEN:-32}" \
      --stride "${NSRL_LEXEME_STRIDE:-1}" \
      --preview-tokens "${NSRL_PREVIEW_TOKENS:-32}" \
      "${token_extra[@]}"
    cp "$frozen_vocab" "$lexeme_vocab"
  else
    cargo run --release -q -p nsrl-corpus -- lexeme-tokenize \
      --corpus "$corpus" \
      --tokens-out "$lexeme_tokens" \
      --vocab-out "$lexeme_vocab" \
      --trace "$lexeme_trace" \
      --seq-len "${NSRL_LEXEME_SEQ_LEN:-32}" \
      --stride "${NSRL_LEXEME_STRIDE:-1}" \
      --max-vocab "${NSRL_MAX_VOCAB:-4096}" \
      --lexeme-vocab-profile "${NSRL_LEXEME_VOCAB_PROFILE:-balanced}" \
      --lexeme-frequency-cap "${NSRL_LEXEME_FREQUENCY_CAP:-4096}" \
      --preview-tokens "${NSRL_PREVIEW_TOKENS:-32}" \
      "${token_extra[@]}"
  fi
fi

MANIFEST_PATH="$manifest" \
WORK_DIR="$work_dir" \
DATASET_S3_URI="$dataset_s3_uri" \
DATASET="$dataset" \
VERSION="$version" \
RAW_S3_URI="$raw_s3_uri" \
TOKEN_KIND="$token_kind" \
CLEAN_PROFILE="$clean_profile" \
FROZEN_VOCAB="$frozen_vocab" \
python3 - <<'PY'
import json, os, pathlib, time
manifest = pathlib.Path(os.environ["MANIFEST_PATH"])
work_dir = pathlib.Path(os.environ["WORK_DIR"])
dataset_s3_uri = os.environ["DATASET_S3_URI"].rstrip("/")
artifacts = []
for path in sorted(work_dir.rglob("*")):
    if path.is_file():
        rel = path.relative_to(work_dir).as_posix()
        artifacts.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "s3_uri": f"{dataset_s3_uri}/{rel}",
        })
manifest.write_text(json.dumps({
    "schema": "nsrl.aws_dataset_manifest.v1",
    "dataset": os.environ["DATASET"],
    "version": os.environ["VERSION"],
    "raw_s3_uri": os.environ["RAW_S3_URI"],
    "dataset_s3_uri": dataset_s3_uri,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "token_kind": os.environ["TOKEN_KIND"],
    "clean_profile": os.environ["CLEAN_PROFILE"],
    "frozen_vocab": os.environ["FROZEN_VOCAB"] or None,
    "artifacts": artifacts,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

aws s3 sync "$work_dir" "$dataset_s3_uri" --only-show-errors
echo "dataset_manifest=$manifest"
echo "dataset_s3_uri=$dataset_s3_uri"
