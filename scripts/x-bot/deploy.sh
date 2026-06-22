#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AWS_REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
AWS_REGION="${AWS_REGION:-us-west-2}"
FUNCTION_NAME="${FUNCTION_NAME:-crowley-bard-mention-bot}"
ROLE_NAME="${ROLE_NAME:-crowley-bard-mention-bot-role}"
POLICY_NAME="${POLICY_NAME:-crowley-bard-mention-bot-policy}"
STATE_TABLE="${STATE_TABLE:-crowley-bard-mention-state}"
SECRET_NAME="${SECRET_NAME:-crowley-bard/x-api}"
X_CONTEXT_ARCHIVE_S3_URI="${X_CONTEXT_ARCHIVE_S3_URI:-}"
RULE_NAME="${RULE_NAME:-crowley-bard-mention-poll}"
SCHEDULE_EXPRESSION="${SCHEDULE_EXPRESSION:-rate(15 minutes)}"
LAMBDA_RUNTIME="${LAMBDA_RUNTIME:-python3.12}"
LAMBDA_ARCHITECTURE="${LAMBDA_ARCHITECTURE:-arm64}"
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-120}"
LAMBDA_MEMORY_MB="${LAMBDA_MEMORY_MB:-1024}"
X_BOT_USER_ID="${X_BOT_USER_ID:-}"
X_BOT_HANDLE="${X_BOT_HANDLE:-}"
X_DRY_RUN="${X_DRY_RUN:-true}"
X_DRY_RUN_ADVANCE_STATE="${X_DRY_RUN_ADVANCE_STATE:-true}"
X_BOOTSTRAP_REPLY="${X_BOOTSTRAP_REPLY:-false}"
X_MAX_MENTIONS_PER_POLL="${X_MAX_MENTIONS_PER_POLL:-10}"
X_MAX_REPLIES_PER_RUN="${X_MAX_REPLIES_PER_RUN:-1}"
X_MAX_REPLIES_PER_15M="${X_MAX_REPLIES_PER_15M:-1}"
X_MAX_REPLIES_PER_DAY="${X_MAX_REPLIES_PER_DAY:-10}"
X_MAX_REPLIES_PER_MONTH="${X_MAX_REPLIES_PER_MONTH:-100}"
X_REPLY_ENGINE="${X_REPLY_ENGINE:-nsrl-live}"
X_NSRL_MAX_NEW_TOKENS="${X_NSRL_MAX_NEW_TOKENS:-60}"
X_NSRL_TOP_K="${X_NSRL_TOP_K:-12}"
X_NSRL_TIMEOUT_SECONDS="${X_NSRL_TIMEOUT_SECONDS:-12}"
X_CONTEXT_ADAPT="${X_CONTEXT_ADAPT:-false}"
X_CONTEXT_MAX_CHARS="${X_CONTEXT_MAX_CHARS:-1800}"
X_CONTEXT_REPEAT_COUNT="${X_CONTEXT_REPEAT_COUNT:-3}"
X_CONTEXT_ADAPT_MAX_WINDOWS="${X_CONTEXT_ADAPT_MAX_WINDOWS:-64}"
X_CONTEXT_ADAPT_LR_SHIFT="${X_CONTEXT_ADAPT_LR_SHIFT:-18}"
X_CONTEXT_ADAPT_TIMEOUT_SECONDS="${X_CONTEXT_ADAPT_TIMEOUT_SECONDS:-20}"
X_STANDALONE_CANDIDATES="${X_STANDALONE_CANDIDATES:-6}"
X_PUBLIC_TWEET_MIN_SCORE="${X_PUBLIC_TWEET_MIN_SCORE:-48}"
X_SIGIL_ENABLED="${X_SIGIL_ENABLED:-true}"
X_SIGIL_CANDIDATES="${X_SIGIL_CANDIDATES:-8}"
X_SIGIL_PASSES="${X_SIGIL_PASSES:-4}"
X_SIGIL_TIMEOUT_SECONDS="${X_SIGIL_TIMEOUT_SECONDS:-60}"
SECRET_JSON_FILE="${X_BOT_SECRET_JSON_FILE:-}"

command -v aws >/dev/null || {
  echo "aws CLI is required" >&2
  exit 1
}
command -v python3 >/dev/null || {
  echo "python3 is required" >&2
  exit 1
}
command -v zip >/dev/null || {
  echo "zip is required" >&2
  exit 1
}

if [[ -n "$SECRET_JSON_FILE" && ! -f "$SECRET_JSON_FILE" ]]; then
  echo "X_BOT_SECRET_JSON_FILE does not exist: $SECRET_JSON_FILE" >&2
  exit 1
fi

echo "Using AWS region: $AWS_REGION"

if [[ -n "$SECRET_JSON_FILE" ]]; then
  if aws secretsmanager describe-secret \
    --region "$AWS_REGION" \
    --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
    aws secretsmanager put-secret-value \
      --region "$AWS_REGION" \
      --secret-id "$SECRET_NAME" \
      --secret-string "file://$SECRET_JSON_FILE" >/dev/null
    echo "Updated Secrets Manager secret: $SECRET_NAME"
  else
    aws secretsmanager create-secret \
      --region "$AWS_REGION" \
      --name "$SECRET_NAME" \
      --secret-string "file://$SECRET_JSON_FILE" >/dev/null
    echo "Created Secrets Manager secret: $SECRET_NAME"
  fi
fi

SECRET_ARN="$(aws secretsmanager describe-secret \
  --region "$AWS_REGION" \
  --secret-id "$SECRET_NAME" \
  --query ARN \
  --output text)"

if ! aws dynamodb describe-table \
  --region "$AWS_REGION" \
  --table-name "$STATE_TABLE" >/dev/null 2>&1; then
  aws dynamodb create-table \
    --region "$AWS_REGION" \
    --table-name "$STATE_TABLE" \
    --billing-mode PAY_PER_REQUEST \
    --attribute-definitions AttributeName=pk,AttributeType=S \
    --key-schema AttributeName=pk,KeyType=HASH >/dev/null
  aws dynamodb wait table-exists \
    --region "$AWS_REGION" \
    --table-name "$STATE_TABLE"
  echo "Created DynamoDB table: $STATE_TABLE"
else
  echo "DynamoDB table exists: $STATE_TABLE"
fi

aws dynamodb update-time-to-live \
  --region "$AWS_REGION" \
  --table-name "$STATE_TABLE" \
  --time-to-live-specification Enabled=true,AttributeName=expires_at >/dev/null 2>&1 || true

TABLE_ARN="$(aws dynamodb describe-table \
  --region "$AWS_REGION" \
  --table-name "$STATE_TABLE" \
  --query Table.TableArn \
  --output text)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
TRUST_JSON="$TMP_DIR/trust.json"
POLICY_JSON="$TMP_DIR/policy.json"
ENV_JSON="$TMP_DIR/env.json"

python3 - "$TRUST_JSON" <<'PY'
import json
import sys

trust = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(trust, handle)
PY

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$TRUST_JSON" >/dev/null
  echo "Created IAM role: $ROLE_NAME"
  sleep 10
else
  echo "IAM role exists: $ROLE_NAME"
fi

ROLE_ARN="$(aws iam get-role \
  --role-name "$ROLE_NAME" \
  --query Role.Arn \
  --output text)"

SECRET_ARN="$SECRET_ARN" TABLE_ARN="$TABLE_ARN" X_CONTEXT_ARCHIVE_S3_URI="$X_CONTEXT_ARCHIVE_S3_URI" python3 - "$POLICY_JSON" <<'PY'
import json
import os
import sys
import urllib.parse

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "arn:aws:logs:*:*:*",
        },
        {
            "Effect": "Allow",
            "Action": ["secretsmanager:GetSecretValue"],
            "Resource": os.environ["SECRET_ARN"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DescribeTable",
            ],
            "Resource": os.environ["TABLE_ARN"],
        },
    ],
}
archive_uri = os.environ.get("X_CONTEXT_ARCHIVE_S3_URI", "")
if archive_uri:
    parsed = urllib.parse.urlparse(archive_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SystemExit(f"invalid X_CONTEXT_ARCHIVE_S3_URI: {archive_uri}")
    prefix = parsed.path.strip("/")
    resource = f"arn:aws:s3:::{parsed.netloc}/{prefix}/*" if prefix else f"arn:aws:s3:::{parsed.netloc}/*"
    policy["Statement"].append({
        "Effect": "Allow",
        "Action": ["s3:PutObject"],
        "Resource": resource,
    })
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(policy, handle)
PY

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document "file://$POLICY_JSON" >/dev/null

ZIP_PATH="$("$SCRIPT_DIR/package.sh")"

X_SECRET_ID="$SECRET_NAME" \
X_STATE_TABLE="$STATE_TABLE" \
X_CONTEXT_ARCHIVE_S3_URI="$X_CONTEXT_ARCHIVE_S3_URI" \
X_BOT_USER_ID="$X_BOT_USER_ID" \
X_BOT_HANDLE="$X_BOT_HANDLE" \
X_DRY_RUN="$X_DRY_RUN" \
X_DRY_RUN_ADVANCE_STATE="$X_DRY_RUN_ADVANCE_STATE" \
X_BOOTSTRAP_REPLY="$X_BOOTSTRAP_REPLY" \
X_MAX_MENTIONS_PER_POLL="$X_MAX_MENTIONS_PER_POLL" \
X_MAX_REPLIES_PER_RUN="$X_MAX_REPLIES_PER_RUN" \
X_MAX_REPLIES_PER_15M="$X_MAX_REPLIES_PER_15M" \
X_MAX_REPLIES_PER_DAY="$X_MAX_REPLIES_PER_DAY" \
X_MAX_REPLIES_PER_MONTH="$X_MAX_REPLIES_PER_MONTH" \
X_REPLY_ENGINE="$X_REPLY_ENGINE" \
X_NSRL_MAX_NEW_TOKENS="$X_NSRL_MAX_NEW_TOKENS" \
X_NSRL_TOP_K="$X_NSRL_TOP_K" \
X_NSRL_TIMEOUT_SECONDS="$X_NSRL_TIMEOUT_SECONDS" \
X_CONTEXT_ADAPT="$X_CONTEXT_ADAPT" \
X_CONTEXT_MAX_CHARS="$X_CONTEXT_MAX_CHARS" \
X_CONTEXT_REPEAT_COUNT="$X_CONTEXT_REPEAT_COUNT" \
X_CONTEXT_ADAPT_MAX_WINDOWS="$X_CONTEXT_ADAPT_MAX_WINDOWS" \
X_CONTEXT_ADAPT_LR_SHIFT="$X_CONTEXT_ADAPT_LR_SHIFT" \
X_CONTEXT_ADAPT_TIMEOUT_SECONDS="$X_CONTEXT_ADAPT_TIMEOUT_SECONDS" \
X_STANDALONE_CANDIDATES="$X_STANDALONE_CANDIDATES" \
X_PUBLIC_TWEET_MIN_SCORE="$X_PUBLIC_TWEET_MIN_SCORE" \
X_SIGIL_ENABLED="$X_SIGIL_ENABLED" \
X_SIGIL_CANDIDATES="$X_SIGIL_CANDIDATES" \
X_SIGIL_PASSES="$X_SIGIL_PASSES" \
X_SIGIL_TIMEOUT_SECONDS="$X_SIGIL_TIMEOUT_SECONDS" \
python3 - "$ENV_JSON" <<'PY'
import json
import os
import sys

names = [
    "X_SECRET_ID",
    "X_STATE_TABLE",
    "X_CONTEXT_ARCHIVE_S3_URI",
    "X_BOT_USER_ID",
    "X_BOT_HANDLE",
    "X_DRY_RUN",
    "X_DRY_RUN_ADVANCE_STATE",
    "X_BOOTSTRAP_REPLY",
    "X_MAX_MENTIONS_PER_POLL",
    "X_MAX_REPLIES_PER_RUN",
    "X_MAX_REPLIES_PER_15M",
    "X_MAX_REPLIES_PER_DAY",
    "X_MAX_REPLIES_PER_MONTH",
    "X_REPLY_ENGINE",
    "X_NSRL_MAX_NEW_TOKENS",
    "X_NSRL_TOP_K",
    "X_NSRL_TIMEOUT_SECONDS",
    "X_CONTEXT_ADAPT",
    "X_CONTEXT_MAX_CHARS",
    "X_CONTEXT_REPEAT_COUNT",
    "X_CONTEXT_ADAPT_MAX_WINDOWS",
    "X_CONTEXT_ADAPT_LR_SHIFT",
    "X_CONTEXT_ADAPT_TIMEOUT_SECONDS",
    "X_STANDALONE_CANDIDATES",
    "X_PUBLIC_TWEET_MIN_SCORE",
    "X_SIGIL_ENABLED",
    "X_SIGIL_CANDIDATES",
    "X_SIGIL_PASSES",
    "X_SIGIL_TIMEOUT_SECONDS",
]
env = {"Variables": {name: os.environ.get(name, "") for name in names}}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(env, handle)
PY

FUNCTION_EXISTS=0
if aws lambda get-function \
  --region "$AWS_REGION" \
  --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  FUNCTION_EXISTS=1
  CURRENT_ARCHITECTURE="$(aws lambda get-function-configuration \
    --region "$AWS_REGION" \
    --function-name "$FUNCTION_NAME" \
    --query 'Architectures[0]' \
    --output text 2>/dev/null || true)"
  if [[ "$CURRENT_ARCHITECTURE" != "$LAMBDA_ARCHITECTURE" ]]; then
    echo "Recreating Lambda function to switch architecture: ${CURRENT_ARCHITECTURE:-unknown} -> $LAMBDA_ARCHITECTURE"
    aws lambda delete-function \
      --region "$AWS_REGION" \
      --function-name "$FUNCTION_NAME"
    FUNCTION_EXISTS=0
  fi
fi

if [[ "$FUNCTION_EXISTS" == "1" ]]; then
  aws lambda update-function-code \
    --region "$AWS_REGION" \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" >/dev/null
  aws lambda wait function-updated \
    --region "$AWS_REGION" \
    --function-name "$FUNCTION_NAME"
  aws lambda update-function-configuration \
    --region "$AWS_REGION" \
    --function-name "$FUNCTION_NAME" \
    --role "$ROLE_ARN" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY_MB" \
    --environment "file://$ENV_JSON" >/dev/null
  echo "Updated Lambda function: $FUNCTION_NAME"
else
  aws lambda create-function \
    --region "$AWS_REGION" \
    --function-name "$FUNCTION_NAME" \
    --runtime "$LAMBDA_RUNTIME" \
    --handler lambda_function.lambda_handler \
    --role "$ROLE_ARN" \
    --architectures "$LAMBDA_ARCHITECTURE" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY_MB" \
    --zip-file "fileb://$ZIP_PATH" \
    --environment "file://$ENV_JSON" >/dev/null
  echo "Created Lambda function: $FUNCTION_NAME"
fi

aws lambda put-function-concurrency \
  --region "$AWS_REGION" \
  --function-name "$FUNCTION_NAME" \
  --reserved-concurrent-executions 1 >/dev/null

FUNCTION_ARN="$(aws lambda get-function \
  --region "$AWS_REGION" \
  --function-name "$FUNCTION_NAME" \
  --query Configuration.FunctionArn \
  --output text)"

aws events put-rule \
  --region "$AWS_REGION" \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE_EXPRESSION" \
  --state ENABLED >/dev/null

RULE_ARN="$(aws events describe-rule \
  --region "$AWS_REGION" \
  --name "$RULE_NAME" \
  --query Arn \
  --output text)"

aws lambda add-permission \
  --region "$AWS_REGION" \
  --function-name "$FUNCTION_NAME" \
  --statement-id "${RULE_NAME}-invoke" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN" >/dev/null 2>&1 || true

aws events put-targets \
  --region "$AWS_REGION" \
  --rule "$RULE_NAME" \
  --targets "Id"="lambda","Arn"="$FUNCTION_ARN" >/dev/null

echo "Scheduled $FUNCTION_NAME with $SCHEDULE_EXPRESSION"
echo "Dry run: $X_DRY_RUN"
echo "Invoke once:"
echo "aws lambda invoke --region $AWS_REGION --function-name $FUNCTION_NAME --payload '{\"dry_run\":true}' /tmp/crowley-bard-mention-bot.json"
