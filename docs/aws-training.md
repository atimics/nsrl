# AWS Training Lane

NSRL training is CPU-native Rust, so the first cloud lane is a plain EC2 runner
plus S3 artifacts. The scripts here do not create credentials, buckets, public
ACLs, or CloudFront distributions. They assume the instance has an IAM role or
AWS CLI profile that can read/write the chosen S3 prefix.

The intended S3 layout is:

```text
s3://BUCKET/PREFIX/
  corpus/
    raw/<dataset>/...
    datasets/<dataset>/<version>/
      raw/
      clean/
      tokens/
      traces/
      manifest.json
  schedules/
  checkpoints/<checkpoint-name>/
    latest.nsrlmt
    latest.trace.jsonl
    latest.progress.jsonl
    latest.run.json
    latest.checkpoint.json
  runs/<run-name>/
  dashboard/
    index.html
    runs.json
    latest.json
```

S3 is the source of truth for cloud inputs and outputs. Local `data/` paths on
an instance are cache/workspace copies only.

## What Gets Published

`scripts/aws/run-mini-transformer-training.sh` writes:

- `s3://BUCKET/PREFIX/runs/<run-name>/run.json`
- `s3://BUCKET/PREFIX/runs/<run-name>/train.log`
- `s3://BUCKET/PREFIX/runs/<run-name>/<run-name>.progress.jsonl`
- `s3://BUCKET/PREFIX/runs/<run-name>/<run-name>.trace.jsonl`
- `s3://BUCKET/PREFIX/runs/<run-name>/<run-name>.nsrlmt`
- `s3://BUCKET/PREFIX/runs/<run-name>/<run-name>.nsrlswarm` for swarm runs
- `s3://BUCKET/PREFIX/runs/<run-name>/<run-name>.manifest.jsonl` for swarm runs
- `s3://BUCKET/PREFIX/dashboard/index.html`
- `s3://BUCKET/PREFIX/dashboard/runs.json`
- `s3://BUCKET/PREFIX/dashboard/latest.json`

The dashboard is a static HTML file. During a run it shows heartbeat, progress
per mille, elapsed time, command, log tail, current adaptive shifts, movement
counters, and artifact links. After the trace lands, it shows final metrics
parsed from the nested training trace:

- probability-error delta,
- final accuracy per mille,
- rollback and rejected-batch counts,
- invalid-forward count,
- final learning-rate shifts,
- output/MLP/embedding/attention movement.

The trainer emits compact progress JSON at batch intervals through
`--progress-out`. This is intentionally cheaper than live full-corpus
validation, so live probability loss is still a final-trace metric unless a run
explicitly chooses to pay for repeated evaluation.

The EC2 launch API defaults to `mini-transformer-swarm` on Graviton and passes
`RUSTFLAGS=-C target-cpu=native`. Set `swarm_workers` or `NSRL_SWARM_WORKERS`;
`0` means use the instance's online CPU count. The shell runner itself remains
single-worker by default unless `NSRL_MODE=mini-transformer-swarm` is set.
Binary traces are still limited to `mini-transformer-mlp`; swarm runs should
keep `NSRL_TRACE_FORMAT=json` until the binary swarm schema lands.

Mini-transformer runs also expose `NSRL_BATCH_MODE=serial|map-reduce` and
`NSRL_MAP_REDUCE_WORKERS=N`. Map-reduce is the default Lambda cost lane for the
fast linear/NoPE byte path: workers accumulate private i64 gradients from a
frozen batch model, the main thread reduces them deterministically, and one
checked candidate update commits or rejects atomically. Use
`scripts/aws/run-crowley-bard-lambda-mapreduce.sh` for the current Crowley Bard
64k comparison run, or call `scripts/aws/run-lambda-swarm-comparison.mjs`
directly for a different S3 token artifact.

## Lexeme Lambda Sweeps

Crowley Bard quality currently scales better through guarded lexeme continuation
than through naive model averaging. Keep the model shape fixed (`v4096`,
embedding dim 16, context 8) until corpus and schedule sweeps stop producing
cheap wins. Continual lexeme training must use a frozen vocab TSV from the
baseline model; do not continue from `visionary-twitter-bot-demo/v4096.nsrllm`
on token IDs discovered from a different corpus.

The current expanded-corpus sweep retokenizes the balanced-prose corpus against
the Twitter bot vocab:

```sh
cargo run --release -q -p nsrl-corpus -- lexeme-tokenize-fixed-vocab \
  --corpus data/processed/visionary-balanced-prose-balanced-prose-interleave-v6-seq2/corpus.txt \
  --vocab data/processed/visionary-twitter-bot-demo/v4096.vocab.tsv \
  --tokens-out data/processed/visionary-expanded-frozen-v4096/v4096.tokens.u16 \
  --trace data/processed/visionary-expanded-frozen-v4096/v4096.tokens.trace.jsonl \
  --seq-len 32 \
  --stride 1
```

Launch a guarded Lambda sweep as independent continuation candidates from the
current bot model. The 2026-06-22 sweep used 4k/8k/12k/16k softmax windows by
`lr_shift` 24/25 against `visionary-expanded-frozen-v4096`. Track it with:

```sh
scripts/aws/watch-lexeme-sweep-dashboard.mjs \
  --run-dir data/aws-lambda-lexeme/runs/<run-name> \
  --s3-prefix s3://nsrl-training-022118847419-us-east-1/wikibard/lambda-runs/<run-name>

python3 -m http.server 8765 \
  --directory data/aws-lambda-lexeme/runs/<run-name>/dashboard
```

The watcher writes a local auto-refreshing dashboard plus `runs.json`. It polls
worker summaries from S3, downloads completed models, evaluates candidates
against the expanded frozen-vocab tokens, and samples `the world is`.

Latest result:

- Run: `visionary-expanded-frozen-v4096-sweep-20260622T075244Z`
- Local dashboard: `http://127.0.0.1:8765/`
- Best candidate: `w16384-lr24` / `worker-006`
- Eval: `8.660` bits/token versus current baseline `9.901`
- Candidate: `data/aws-lambda-lexeme/candidates/visionary-expanded-frozen-v4096-w16384-lr24.nsrllm`
- S3: `s3://nsrl-training-022118847419-us-east-1/wikibard/candidates/visionary-expanded-frozen-v4096-w16384-lr24.nsrllm`

## Build A Versioned Dataset

Upload raw files to S3 first:

```sh
aws s3 sync data/raw/ s3://BUCKET/PREFIX/corpus/raw/wikibard/
```

Then build a cleaned/tokenized version:

```sh
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_DATASET_NAME=wikibard \
NSRL_DATASET_VERSION=20260621T000000Z \
NSRL_RAW_S3_URI=s3://BUCKET/PREFIX/corpus/raw/wikibard \
NSRL_TOKEN_KIND=both \
NSRL_TEXT_CLEAN_PROFILE=mixed \
NSRL_MAX_VOCAB=4096 \
NSRL_LEXEME_VOCAB_PROFILE=balanced \
NSRL_LEXEME_FREQUENCY_CAP=4096 \
scripts/aws/build-corpus.sh
```

This publishes a versioned dataset manifest and artifacts under:

```text
s3://BUCKET/PREFIX/corpus/datasets/wikibard/20260621T000000Z/
```

For mini-transformer byte training, point `NSRL_TOKENS_S3_URI` at the produced
`.tokens.u8` artifact. For lexeme/concept experiments, use the `.tokens.u16`
and vocab TSV artifacts.

## Freeze Lexeme Vocabularies

Continual lexeme training requires immutable token IDs. Once a run becomes a
golden baseline, treat its vocab TSV as part of the model contract. Future
corpus versions should tokenize against that TSV instead of discovering a new
vocabulary:

```sh
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_DATASET_NAME=wikibard \
NSRL_DATASET_VERSION=20260628T000000Z \
NSRL_RAW_S3_URI=s3://BUCKET/PREFIX/corpus/raw/wikibard-expanded \
NSRL_TOKEN_KIND=lexeme \
NSRL_FROZEN_VOCAB_S3_URI=s3://BUCKET/PREFIX/corpus/datasets/wikibard/20260621T000000Z/tokens/wikibard-20260621T000000Z.lexeme-v4096.vocab.tsv \
scripts/aws/build-corpus.sh
```

The builder uses `nsrl-corpus lexeme-tokenize-fixed-vocab` in this mode and
copies the frozen TSV into the new dataset version. Unknown words fall back
through the byte-token range instead of shifting existing lexeme IDs.

## Instance Bootstrap

Use an EC2 CPU instance with enough local disk for the corpus and traces. A
compute-optimized Graviton instance is a good first choice.

On a fresh Linux instance:

```sh
sudo yum install -y git python3 awscli || sudo apt-get update && sudo apt-get install -y git python3 awscli build-essential
curl https://sh.rustup.rs -sSf | sh -s -- -y
. "$HOME/.cargo/env"
git clone https://github.com/atimics/nsrl.git
cd nsrl
cargo build --release -p nsrl-train
```

Upload the token file once from your workstation:

```sh
aws s3 cp data/processed/wiki-bard-corpus.tokens.u8 \
  s3://BUCKET/PREFIX/input/wiki-bard-corpus.tokens.u8
```

For the full pipeline, prefer using the versioned dataset path produced by
`build-corpus.sh` instead of the older `input/` prefix.

## Smoke Run

Run a short job first to verify IAM, S3 sync, build speed, and dashboard
rendering:

```sh
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8 \
NSRL_TOKENS_S3_URI=s3://BUCKET/PREFIX/corpus/datasets/wikibard/20260621T000000Z/tokens/wikibard-20260621T000000Z.tokens.u8 \
NSRL_RUN_NAME=smoke-8192 \
NSRL_MODE=mini-transformer-swarm \
NSRL_SWARM_WORKERS=0 \
NSRL_MAX_WINDOWS=8192 \
NSRL_SEQ_LEN=4 \
NSRL_STRIDE=36965 \
NSRL_BATCH_WINDOWS=2 \
NSRL_ADAPTIVE_RULE_SHIFTS=1 \
NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS=0 \
scripts/aws/run-mini-transformer-training.sh
```

For the local launch API, the JSON body can set the same parallel knobs:

```sh
curl -X POST http://127.0.0.1:8766/runs \
  -H 'content-type: application/json' \
  -d '{"run_name":"c8g-swarm-65536","instance_type":"c8g.4xlarge","mode":"mini-transformer-swarm","swarm_workers":0,"max_windows":65536}'
```

Open or download:

```sh
aws s3 cp s3://BUCKET/PREFIX/dashboard/index.html /tmp/nsrl-dashboard.html
open /tmp/nsrl-dashboard.html
```

If the bucket is configured for static website hosting, browse to the website
URL for `dashboard/index.html`.

## First Long Run

The current production controller is rule-only adaptive shifts. Holographic
shift memory is implemented and measured, but it is still experimental and
should stay off for the first long run.

```sh
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8 \
NSRL_TOKENS_S3_URI=s3://BUCKET/PREFIX/corpus/datasets/wikibard/20260621T000000Z/tokens/wikibard-20260621T000000Z.tokens.u8 \
NSRL_RUN_NAME=rule-linear-nope-250k-001 \
NSRL_MAX_WINDOWS=250000 \
NSRL_SEQ_LEN=4 \
NSRL_STRIDE=1211 \
NSRL_BATCH_WINDOWS=2 \
NSRL_OUT_SHIFT=18 \
NSRL_MLP_SHIFT=17 \
NSRL_EMBED_SHIFT=13 \
NSRL_ATTENTION_SHIFT=22 \
NSRL_ATTENTION_Q_SHIFT=18 \
NSRL_ATTENTION_QK_SHIFT=16 \
NSRL_ATTENTION=linear \
NSRL_POSITION=nope \
NSRL_ADAPTIVE_RULE_SHIFTS=1 \
NSRL_ADAPTIVE_RULE_INTERVAL_BATCHES=128 \
NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS=0 \
NSRL_SYNC_SECONDS=60 \
scripts/aws/run-mini-transformer-training.sh
```

For a denser local-context run, change `NSRL_SEQ_LEN` to `16` and use
`NSRL_EMBED_SHIFT=12`. Treat that as a separate experiment because it changes
the optimizer regime.

## Resume From A Model

To continue a saved model:

```sh
NSRL_RUN_NAME=rule-linear-nope-continuation-001 \
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_RESUME_CHECKPOINT=wiki-bard-golden \
NSRL_PUBLISH_CHECKPOINT=wiki-bard-golden \
NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8 \
NSRL_TOKENS_S3_URI=s3://BUCKET/PREFIX/input/wiki-bard-corpus.tokens.u8 \
scripts/aws/run-mini-transformer-training.sh
```

`NSRL_RESUME_CHECKPOINT` downloads:

```text
s3://BUCKET/PREFIX/checkpoints/wiki-bard-golden/latest.nsrlmt
```

and passes it to `nsrl-train` as `--resume-from`. `NSRL_PUBLISH_CHECKPOINT`
publishes the successful run back to the same checkpoint prefix as the new
`latest.nsrlmt`, together with the final trace, progress heartbeat, run JSON,
and checkpoint pointer JSON.

For an explicit one-off resume path, use `NSRL_MODEL` with `NSRL_MODEL_S3_URI`
or `NSRL_RESUME_FROM_S3_URI`. The CLI also accepts `--resume-from PATH`
directly; it is an alias for loading the existing `.nsrlmt` weights in
`mini-transformer-mlp` mode.

NSRL's batch accumulators are cleared at batch boundaries, so the checkpoint is
just the model bytes plus the selected shift configuration. There is no Adam
momentum or variance state to preserve.

## Schedule Multiple Runs

Use `scripts/aws/example-schedule.tsv` as the format:

```text
# name KEY=VALUE...
smoke-8192 NSRL_MAX_WINDOWS=8192 NSRL_SEQ_LEN=4 NSRL_STRIDE=36965
seq16-8192 NSRL_MAX_WINDOWS=8192 NSRL_SEQ_LEN=16 NSRL_STRIDE=36965 NSRL_EMBED_SHIFT=12
```

Then run:

```sh
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8 \
NSRL_TOKENS_S3_URI=s3://BUCKET/PREFIX/corpus/datasets/wikibard/20260621T000000Z/tokens/wikibard-20260621T000000Z.tokens.u8 \
NSRL_MAX_PARALLEL=8 \
scripts/aws/run-training-schedule.sh scripts/aws/example-schedule.tsv
```

This is deliberately process-level scheduling. The trainer is currently mostly
single-process CPU work, so a large EC2 instance is best used by running a fleet
of independent ablations next to one long hero job.

## Local Launch API

For interactive work, run the local launch API next to the static dashboard:

```sh
AWS_PROFILE=staging \
scripts/aws/launch-api.py --host 127.0.0.1 --port 8766
```

Health and run index:

```sh
curl http://127.0.0.1:8766/health
curl http://127.0.0.1:8766/runs
```

Dry-run a launch without creating EC2:

```sh
curl -X POST http://127.0.0.1:8766/runs/dry-run \
  -H 'content-type: application/json' \
  --data '{
    "run_name": "qk-rescue-64k-dry-run",
    "max_windows": 65536,
    "instance_type": "c8g.xlarge",
    "publish_checkpoint": "wikibard-linear-nope-qk-rescue"
  }'
```

Launch a real auto-terminating EC2 run:

```sh
curl -X POST http://127.0.0.1:8766/runs \
  -H 'content-type: application/json' \
  --data '{
    "run_name": "qk-rescue-64k-001",
    "max_windows": 65536,
    "seq_len": 4,
    "stride": 1211,
    "batch_windows": 2,
    "attention": "linear",
    "position": "nope",
    "out_shift": 18,
    "mlp_shift": 17,
    "embed_shift": 13,
    "attention_shift": 22,
    "attention_q_shift": 18,
    "attention_qk_shift": 16,
    "adaptive_rule_shifts": true,
    "instance_type": "c8g.xlarge",
    "publish_checkpoint": "wikibard-linear-nope-qk-rescue"
  }'
```

The API writes a pending row into the dashboard, launches AL2023 ARM64 EC2 with
the current S3 source bundle, tags the instance with the run name, and sets both
`NSRL_TERMINATE_ON_EXIT=1` and EC2 instance-initiated shutdown behavior to
`terminate`. User-data also schedules a 180-minute shutdown as a fallback.

## Export Results

Download the dashboard run index and export it:

```sh
aws s3 cp s3://BUCKET/PREFIX/dashboard/runs.json /tmp/nsrl-runs.json
scripts/aws/collect-results.py \
  --runs-json /tmp/nsrl-runs.json \
  --out data/aws-runs/results.tsv \
  --format tsv
scripts/aws/collect-results.py \
  --runs-json /tmp/nsrl-runs.json \
  --out data/aws-runs/results.md \
  --format md
```

## Dashboard Notes

The dashboard refreshes `runs.json` every 30 seconds in the browser. The runner
syncs the dashboard every `NSRL_SYNC_SECONDS`.

If you want a public URL, enable S3 static website hosting or put CloudFront in
front of `s3://BUCKET/PREFIX/dashboard/`. If the bucket is private, the same
files are still useful from the AWS console or with `aws s3 cp`.
