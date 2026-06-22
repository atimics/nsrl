# Crowley Bard X Mention Bot

Scheduled Lambda for the fictional Crowley Bard demo account. It polls direct
`@` mentions, generates one short contextual reply, and posts through X API v2
using OAuth 1.0a user credentials.

This handles public mentions, not Direct Messages.

Replies use live NSRL inference. The Lambda can also perform an experimental
tiny test-time fine-tune on the incoming context: it tokenizes the context with
the frozen Crowley vocab, continues the bundled `.nsrllm` for a small fixed
number of windows in `/tmp`, generates from that temporary adapted model, and
discards it. This is behind `X_CONTEXT_ADAPT=false` by default because the tiny
model can overfit short noisy context.

## Safety Defaults

- `X_DRY_RUN=true` by default.
- First run seeds `last_seen_id` and does not reply to older mentions unless
  `X_BOOTSTRAP_REPLY=true`.
- Reserved Lambda concurrency is set to `1`.
- Default reply caps are intentionally tiny:
  - `1` reply per run
  - `1` reply per 15 minutes
  - `10` replies per day
  - `100` replies per month
- `X_CONTEXT_ADAPT=false` by default. The adapted model is per-invocation and is
  never written back to the base model bundle.

## Secret

Use a local ignored file for setup, then rotate the X credentials and update the
AWS secret with the rotated values.

```sh
mkdir -p scripts/x-bot/.secrets
cp scripts/x-bot/secret.example.json scripts/x-bot/.secrets/x-api.json
$EDITOR scripts/x-bot/.secrets/x-api.json
```

The secret shape is:

```json
{
  "consumer_key": "...",
  "consumer_secret": "...",
  "access_token": "...",
  "access_token_secret": "..."
}
```

## Deploy

```sh
AWS_REGION=us-west-2 \
X_BOT_SECRET_JSON_FILE=scripts/x-bot/.secrets/x-api.json \
X_BOT_HANDLE=CrowleyBard \
X_DRY_RUN=true \
scripts/x-bot/deploy.sh
```

For local Apple-Silicon deploys, build the Lambda Linux ARM64 binary first:

```sh
scripts/x-bot/build-lambda-binary.sh
```

The deploy script creates or updates:

- Secrets Manager secret: `crowley-bard/x-api`
- DynamoDB state table: `crowley-bard-mention-state`
- Lambda function: `crowley-bard-mention-bot`
- EventBridge schedule: `rate(15 minutes)`

If `X_BOT_USER_ID` is not set, the worker resolves it with `/2/users/me` on
startup.

## GitHub Actions Updates

`.github/workflows/x-bot-deploy.yml` rebuilds `nsrl-train`, packages the live
Crowley model bundle, deploys the Lambda, and invokes a live-inference smoke
test. It runs on manual dispatch and on pushes that touch the bot, workflow, or
Rust inference code.

Required repo secret:

- `AWS_ROLE_TO_ASSUME`: IAM role ARN trusted by GitHub OIDC for this repository.

Required repo variable, unless supplied as a manual workflow input:

- `X_BOT_MODEL_S3_URI`: S3 prefix containing:
  - `v4096.nsrllm`
  - `v4096.vocab.tsv`
  - `v4096.tokens.u16`

Optional repo variables:

- `AWS_REGION`, default `us-east-1`
- `X_BOT_FUNCTION_NAME`, default `crowley-bard-mention-bot`
- `X_BOT_SECRET_NAME`, default `crowley-bard/x-api`
- `X_BOT_STATE_TABLE`, default `crowley-bard-mention-state`
- `X_BOT_RULE_NAME`, default `crowley-bard-mention-poll`
- `X_BOT_SCHEDULE_EXPRESSION`, default `rate(15 minutes)`
- `X_BOT_MAX_REPLIES_PER_DAY`, default `10`
- `X_BOT_MAX_REPLIES_PER_MONTH`, default `100`
- `X_BOT_CONTEXT_ADAPT`, default `false`
- `X_BOT_CONTEXT_MAX_CHARS`, default `1800`
- `X_BOT_CONTEXT_ADAPT_MAX_WINDOWS`, default `64`
- `X_BOT_CONTEXT_ADAPT_LR_SHIFT`, default `18`

Sync the current local model bundle to S3 before using the workflow:

```sh
X_BOT_MODEL_S3_URI=s3://BUCKET/PREFIX/crowley-bard/model \
scripts/x-bot/sync-model-to-s3.sh
```

The GitHub-hosted runner builds a native Linux `x86_64` Lambda package. If the
currently deployed function is `arm64`, the deploy script recreates it on the
next Actions deploy. That is expected; DynamoDB state, schedule, and secret stay
the same.

## Nightly Timeline Tuning

`.github/workflows/x-bot-nightly-tune.yml` runs nightly at `10:17 UTC`, or
manually with a selected UTC day. It:

1. Downloads archived mention/reply events from `X_BOT_CONTEXT_ARCHIVE_S3_URI`.
2. Downloads the current model bundle from `X_BOT_MODEL_S3_URI`.
3. Builds a fixed-vocab nightly context corpus.
4. Continues the current `.nsrllm` with conservative integer updates.
5. Generates smoke samples and applies a small quality gate.
6. Publishes a dated bundle to model history and updates the production model
   prefix only if the gate passes.
7. Redeploys Lambda in dry-run mode with the newly published model.

Required repo variable:

- `X_BOT_CONTEXT_ARCHIVE_S3_URI`: S3 prefix where Lambda archives daily
  mention/reply JSON objects.

Recommended repo variable:

- `X_BOT_MODEL_HISTORY_S3_URI`: S3 prefix for dated nightly candidate bundles.

Useful nightly tuning variables:

- `X_BOT_NIGHTLY_MAX_WINDOWS`, default `512`
- `X_BOT_NIGHTLY_LR_SHIFT`, default `23`
- `X_BOT_NIGHTLY_CONTEXT_REPEAT_COUNT`, default `2`
- `X_BOT_NIGHTLY_MIN_CONTEXT_EVENTS`, default `1`
- `X_BOT_NIGHTLY_MIN_PASSING_SAMPLES`, default `2`

The Lambda archive uses deterministic daily keys, so repeated dry-run polls
overwrite the same event object instead of multiplying duplicates.

## Mentions vs Scheduled Posts

The current EventBridge schedule only polls direct mentions. Reply tweets
intentionally prepend the target `@username` so X threads the response to the
right person.

Generated body text is handle-stripped before posting, and the nightly tuning
corpus removes public `@handles` and author usernames before training. If a
standalone scheduled-tweet path is added later, call the body-generation path
without adding a reply username prefix.

Trigger a one-shot standalone post with the same guardrails:

```sh
aws lambda invoke \
  --region us-east-1 \
  --function-name crowley-bard-mention-bot \
  --cli-binary-format raw-in-base64-out \
  --payload '{"post_tweet":true,"dry_run":true,"prompt":"the first omen today is","candidate_count":6,"min_score":48}' \
  /tmp/crowley-bard-first-post.json
```

Set `"dry_run":false` only when the preview text is ready to publish.
Standalone generation scores candidates for handle/URL safety, length,
repetition, balanced content, punctuation, and complete-thought shape, then
posts only the best candidate above `min_score`.

## Context Adaptation

The bot has two context limits:

- The base lexeme model still has an 8-token rolling context.
- The wrapper can adapt the model on up to `X_CONTEXT_MAX_CHARS` characters
  before generation.

This is closer to retrieval-free test-time training than a normal long-context
transformer. It can bias the reply toward the mention/feed vocabulary and mood,
but it can also overfit short weird context. Keep `X_DRY_RUN=true` while tuning:

```sh
aws lambda invoke \
  --region us-east-1 \
  --function-name crowley-bard-mention-bot \
  --cli-binary-format raw-in-base64-out \
  --payload '{"test_generate":"@CrowleyBard the feed is finales denied, hockey panic, AI design ads, and ominous movie trivia","username":"tester","id":"ctx1"}' \
  /tmp/crowley-bard-context-test.json
```

## Verify

Invoke once in dry-run mode:

```sh
aws lambda invoke \
  --region us-west-2 \
  --function-name crowley-bard-mention-bot \
  --payload '{"dry_run":true}' \
  /tmp/crowley-bard-mention-bot.json

cat /tmp/crowley-bard-mention-bot.json
```

If the payload returns `CreditsDepleted`, the AWS side is wired correctly but X
is refusing mention reads until the developer account has credits or spending
enabled.

Watch logs:

```sh
aws logs tail /aws/lambda/crowley-bard-mention-bot \
  --region us-west-2 \
  --follow
```

## Go Live

After the dry-run output looks right and the exposed credentials have been
rotated, update the Lambda config:

```sh
aws lambda update-function-configuration \
  --region us-west-2 \
  --function-name crowley-bard-mention-bot \
  --environment 'Variables={X_SECRET_ID=crowley-bard/x-api,X_STATE_TABLE=crowley-bard-mention-state,X_BOT_HANDLE=CrowleyBard,X_DRY_RUN=false,X_BOOTSTRAP_REPLY=false,X_MAX_MENTIONS_PER_POLL=10,X_MAX_REPLIES_PER_RUN=1,X_MAX_REPLIES_PER_15M=1,X_MAX_REPLIES_PER_DAY=10,X_MAX_REPLIES_PER_MONTH=100}'
```

Keep the first live schedule at `rate(15 minutes)`. Tighten the daily/monthly
caps rather than the EventBridge interval if you want it quieter.

## Local Tests

```sh
python3 -m unittest scripts/x-bot/test_lambda_function.py
python3 -m py_compile scripts/x-bot/lambda_function.py
```
