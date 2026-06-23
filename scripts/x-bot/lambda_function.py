#!/usr/bin/env python3
"""Scheduled X mention-reply Lambda for the Crowley Bard demo.

The worker polls direct mentions of the bot account, generates a short
contextual reply, and posts it with OAuth 1.0a user context. State is kept in
DynamoDB on AWS and can fall back to a local JSON file for dry-run testing.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from dataclasses import dataclass
from typing import Any

try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover - local machines may not have boto3.
    boto3 = None


API_BASE = "https://api.x.com/2"
UPLOAD_BASE = "https://upload.twitter.com/1.1"
DEFAULT_USER_AGENT = "nsrl-crowley-bard-lambda/0.1"
PUBLIC_MENTION_RE = re.compile(r"@\w+")
MAX_STANDALONE_CANDIDATES = 12
STANDALONE_POST_TTL_SECONDS = 180 * 24 * 60 * 60
MENTION_FAILURE_TTL_SECONDS = 7 * 24 * 60 * 60
SIGIL_SCHEMA = "nsrl.x_bot.solomon_sigil.v1"
DECODE_BANNED_TOKENS = [
    "assistant",
    "chatbot",
    "model",
    "training",
    "prompt",
    "json",
    "http",
    "www",
    "class",
    "align",
    "bgcolor",
    "nbsp",
    "enter",
    "exeunt",
    "dramatis",
    "alicia",
    "crassus",
    "parolles",
    "helena",
    "bertram",
    "lafeu",
    "hamlet",
    "horatio",
    "ophelia",
    "polonius",
    "romeo",
    "juliet",
    "othello",
    "iago",
    "falstaff",
    "prospero",
    "caliban",
    "macbeth",
    "banquo",
    "gloucester",
    "cassio",
]


class BotConfigError(RuntimeError):
    """Raised when required bot configuration is missing."""


class XApiError(RuntimeError):
    """Raised for non-2xx X API responses."""

    def __init__(self, status: int, body: str, headers: dict[str, str]):
        super().__init__(f"X API request failed with HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.headers = headers


class ReplyGenerationError(RuntimeError):
    """Raised when live NSRL inference cannot produce a reply."""


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now(now: dt.datetime | None = None) -> str:
    return (now or utc_now()).isoformat(timespec="seconds").replace("+00:00", "Z")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        value = int(raw)
    if minimum is not None:
        value = max(minimum, value)
    return value


def bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def pick(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def strip_at(handle: str | None) -> str:
    if not handle:
        return ""
    return handle.strip().lstrip("@")


def normalize_public_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"^[,.!?;:]\s*", "", text)
    return text.strip()


def strip_public_mentions(text: str) -> str:
    return normalize_public_text(PUBLIC_MENTION_RE.sub(" ", text or ""))


def id_gt(left: str | None, right: str | None) -> bool:
    if not left:
        return False
    if not right:
        return True
    return int(left) > int(right)


def max_id(*ids: str | None) -> str | None:
    best: str | None = None
    for value in ids:
        if value and id_gt(value, best):
            best = value
    return best


@dataclass
class OAuth1Credentials:
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str

    @classmethod
    def from_secret(cls, secret: dict[str, Any]) -> "OAuth1Credentials":
        consumer_key = pick(secret, "consumer_key", "api_key", "X_CONSUMER_KEY")
        consumer_secret = pick(
            secret,
            "consumer_secret",
            "api_key_secret",
            "api_secret",
            "X_CONSUMER_SECRET",
        )
        access_token = pick(secret, "access_token", "X_ACCESS_TOKEN")
        access_token_secret = pick(
            secret, "access_token_secret", "X_ACCESS_TOKEN_SECRET"
        )
        missing = [
            name
            for name, value in [
                ("consumer_key", consumer_key),
                ("consumer_secret", consumer_secret),
                ("access_token", access_token),
                ("access_token_secret", access_token_secret),
            ]
            if not value
        ]
        if missing:
            raise BotConfigError(f"OAuth1 secret missing: {', '.join(missing)}")
        return cls(
            consumer_key=consumer_key or "",
            consumer_secret=consumer_secret or "",
            access_token=access_token or "",
            access_token_secret=access_token_secret or "",
        )


@dataclass
class BotConfig:
    bot_user_id: str | None
    bot_handle: str
    secret_id: str | None
    state_table: str | None
    context_archive_s3_uri: str
    dry_run: bool
    advance_state_on_dry_run: bool
    bootstrap_reply: bool
    direct_mentions_only: bool
    max_mentions_per_poll: int
    max_replies_per_run: int
    max_replies_per_15m: int
    max_replies_per_day: int
    max_replies_per_month: int
    max_reply_chars: int
    reply_engine: str
    nsrl_bin: str
    nsrl_corpus_bin: str
    nsrl_model: str
    nsrl_vocab: str
    nsrl_tokens: str
    nsrl_max_new_tokens: int
    nsrl_top_k: int
    nsrl_timeout_seconds: int
    context_adapt: bool
    context_max_chars: int
    context_repeat_count: int
    context_adapt_max_windows: int
    context_adapt_lr_shift: int
    context_adapt_timeout_seconds: int
    standalone_candidates: int
    public_tweet_min_score: int
    sigil_enabled: bool
    sigil_bin: str
    sigil_model: str
    sigil_latent_model: str
    sigil_text_index: str
    sigil_candidates: int
    sigil_passes: int
    sigil_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "BotConfig":
        task_root = os.getenv("LAMBDA_TASK_ROOT") or os.path.dirname(__file__)
        return cls(
            bot_user_id=os.getenv("X_BOT_USER_ID"),
            bot_handle=strip_at(os.getenv("X_BOT_HANDLE")),
            secret_id=os.getenv("X_SECRET_ID") or os.getenv("SECRET_ID"),
            state_table=os.getenv("X_STATE_TABLE") or os.getenv("STATE_TABLE"),
            context_archive_s3_uri=os.getenv("X_CONTEXT_ARCHIVE_S3_URI", ""),
            dry_run=env_bool("X_DRY_RUN", True),
            advance_state_on_dry_run=env_bool("X_DRY_RUN_ADVANCE_STATE", True),
            bootstrap_reply=env_bool("X_BOOTSTRAP_REPLY", False),
            direct_mentions_only=env_bool("X_DIRECT_MENTIONS_ONLY", True),
            max_mentions_per_poll=env_int("X_MAX_MENTIONS_PER_POLL", 10, minimum=5),
            max_replies_per_run=env_int("X_MAX_REPLIES_PER_RUN", 1, minimum=0),
            max_replies_per_15m=env_int("X_MAX_REPLIES_PER_15M", 1, minimum=0),
            max_replies_per_day=env_int("X_MAX_REPLIES_PER_DAY", 10, minimum=0),
            max_replies_per_month=env_int("X_MAX_REPLIES_PER_MONTH", 100, minimum=0),
            max_reply_chars=env_int("X_MAX_REPLY_CHARS", 260, minimum=80),
            reply_engine=os.getenv("X_REPLY_ENGINE", "nsrl-live"),
            nsrl_bin=os.getenv("X_NSRL_BIN", os.path.join(task_root, "bin", "nsrl-train")),
            nsrl_corpus_bin=os.getenv(
                "X_NSRL_CORPUS_BIN", os.path.join(task_root, "bin", "nsrl-corpus")
            ),
            nsrl_model=os.getenv(
                "X_NSRL_MODEL", os.path.join(task_root, "model", "v4096.nsrllm")
            ),
            nsrl_vocab=os.getenv(
                "X_NSRL_VOCAB", os.path.join(task_root, "model", "v4096.vocab.tsv")
            ),
            nsrl_tokens=os.getenv(
                "X_NSRL_TOKENS", os.path.join(task_root, "model", "v4096.tokens.u16")
            ),
            nsrl_max_new_tokens=env_int("X_NSRL_MAX_NEW_TOKENS", 60, minimum=8),
            nsrl_top_k=env_int("X_NSRL_TOP_K", 12, minimum=1),
            nsrl_timeout_seconds=env_int("X_NSRL_TIMEOUT_SECONDS", 12, minimum=1),
            context_adapt=env_bool("X_CONTEXT_ADAPT", False),
            context_max_chars=env_int("X_CONTEXT_MAX_CHARS", 1800, minimum=80),
            context_repeat_count=env_int("X_CONTEXT_REPEAT_COUNT", 3, minimum=1),
            context_adapt_max_windows=env_int(
                "X_CONTEXT_ADAPT_MAX_WINDOWS", 64, minimum=1
            ),
            context_adapt_lr_shift=env_int("X_CONTEXT_ADAPT_LR_SHIFT", 18, minimum=1),
            context_adapt_timeout_seconds=env_int(
                "X_CONTEXT_ADAPT_TIMEOUT_SECONDS", 20, minimum=1
            ),
            standalone_candidates=bounded_int(
                os.getenv("X_STANDALONE_CANDIDATES"),
                default=6,
                minimum=1,
                maximum=MAX_STANDALONE_CANDIDATES,
            ),
            public_tweet_min_score=env_int("X_PUBLIC_TWEET_MIN_SCORE", 48, minimum=1),
            sigil_enabled=env_bool("X_SIGIL_ENABLED", True),
            sigil_bin=os.getenv(
                "X_SIGIL_BIN", os.path.join(task_root, "bin", "nsrl-bitmap-sample")
            ),
            sigil_model=os.getenv(
                "X_SIGIL_MODEL", os.path.join(task_root, "solomon", "model.nsrltch")
            ),
            sigil_latent_model=os.getenv(
                "X_SIGIL_LATENT_MODEL",
                os.path.join(task_root, "solomon", "current-best.nsrllat"),
            ),
            sigil_text_index=os.getenv(
                "X_SIGIL_TEXT_INDEX",
                os.path.join(task_root, "solomon", "solomon-spirit-text-signatures.tsv"),
            ),
            sigil_candidates=env_int("X_SIGIL_CANDIDATES", 8, minimum=1),
            sigil_passes=env_int("X_SIGIL_PASSES", 4, minimum=1),
            sigil_timeout_seconds=env_int("X_SIGIL_TIMEOUT_SECONDS", 60, minimum=1),
        )


def load_secret(config: BotConfig) -> dict[str, Any]:
    raw = os.getenv("X_SECRET_JSON")
    if raw:
        return json.loads(raw)
    if not config.secret_id:
        raise BotConfigError("Set X_SECRET_JSON for local runs or X_SECRET_ID in Lambda")
    if boto3 is None:
        raise BotConfigError("boto3 is required to read AWS Secrets Manager")
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=config.secret_id)
    if "SecretString" in response:
        return json.loads(response["SecretString"])
    return json.loads(base64.b64decode(response["SecretBinary"]))


def oauth_percent(value: Any) -> str:
    return urllib.parse.quote(str(value), safe="~")


class XOAuth1Client:
    def __init__(
        self,
        credentials: OAuth1Credentials,
        *,
        api_base: str = API_BASE,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: int = 20,
    ):
        self.credentials = credentials
        self.api_base = api_base.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def _authorization_header(
        self,
        method: str,
        url: str,
        query_params: dict[str, Any] | None = None,
        body_params: dict[str, Any] | None = None,
    ) -> str:
        oauth_params = {
            "oauth_consumer_key": self.credentials.consumer_key,
            "oauth_nonce": uuid.uuid4().hex,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self.credentials.access_token,
            "oauth_version": "1.0",
        }
        signing_pairs: list[tuple[str, str]] = []
        for key, value in (query_params or {}).items():
            if value is not None:
                signing_pairs.append((str(key), str(value)))
        for key, value in (body_params or {}).items():
            if value is not None:
                signing_pairs.append((str(key), str(value)))
        signing_pairs.extend((key, value) for key, value in oauth_params.items())
        signing_pairs.sort(key=lambda pair: (oauth_percent(pair[0]), oauth_percent(pair[1])))
        parameter_string = "&".join(
            f"{oauth_percent(key)}={oauth_percent(value)}" for key, value in signing_pairs
        )
        base_string = "&".join(
            [
                method.upper(),
                oauth_percent(url),
                oauth_percent(parameter_string),
            ]
        )
        signing_key = "&".join(
            [
                oauth_percent(self.credentials.consumer_secret),
                oauth_percent(self.credentials.access_token_secret),
            ]
        )
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode("utf-8"),
                base_string.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("ascii")
        oauth_params["oauth_signature"] = signature
        header_pairs = ", ".join(
            f'{oauth_percent(key)}="{oauth_percent(value)}"'
            for key, value in sorted(oauth_params.items())
        )
        return f"OAuth {header_pairs}"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        url = f"{self.api_base}{path}"
        return self.request_absolute(
            method,
            url,
            params=params,
            json_body=json_body,
            form_body=form_body,
        )

    def request_absolute(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if json_body is not None and form_body is not None:
            raise ValueError("json_body and form_body are mutually exclusive")
        query = urllib.parse.urlencode(params or {})
        full_url = f"{url}?{query}" if query else url
        body = None
        headers = {
            "Authorization": self._authorization_header(method, url, params, form_body),
            "User-Agent": self.user_agent,
        }
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif form_body is not None:
            body = urllib.parse.urlencode(form_body).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            full_url,
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                parsed = json.loads(raw_body) if raw_body else {}
                return parsed, dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            raise XApiError(exc.code, raw_body, dict(exc.headers.items())) from exc

    def get_me(self) -> dict[str, Any]:
        body, _headers = self.request("GET", "/users/me")
        return body

    def get_mentions(
        self, bot_user_id: str, *, since_id: str | None, max_results: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "max_results": str(max(5, min(max_results, 100))),
            "tweet.fields": "author_id,created_at,conversation_id,referenced_tweets",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        if since_id:
            params["since_id"] = since_id
        body, _headers = self.request(
            "GET", f"/users/{bot_user_id}/mentions", params=params
        )
        return body

    def post_reply(self, text: str, in_reply_to_tweet_id: str) -> dict[str, Any]:
        return self.post_reply_with_media(text, in_reply_to_tweet_id, media_ids=None)

    def post_reply_with_media(
        self,
        text: str,
        in_reply_to_tweet_id: str,
        *,
        media_ids: list[str] | None,
    ) -> dict[str, Any]:
        body = {
            "text": text,
            "reply": {"in_reply_to_tweet_id": in_reply_to_tweet_id},
        }
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        response, _headers = self.request("POST", "/tweets", json_body=body)
        return response

    def post_tweet(self, text: str) -> dict[str, Any]:
        return self.post_tweet_with_media(text, media_ids=None)

    def post_tweet_with_media(
        self, text: str, *, media_ids: list[str] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text}
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        response, _headers = self.request("POST", "/tweets", json_body=body)
        return response

    def upload_media_png(self, png_bytes: bytes) -> str:
        response, _headers = self.request_absolute(
            "POST",
            f"{UPLOAD_BASE}/media/upload.json",
            form_body={
                "media_data": base64.b64encode(png_bytes).decode("ascii"),
                "media_category": "tweet_image",
            },
        )
        media_id = response.get("media_id_string") or response.get("media_id")
        if not media_id:
            raise XApiError(200, json.dumps(response), {})
        return str(media_id)


class StateStore:
    def get_item(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def put_item(self, key: str, item: dict[str, Any]) -> None:
        raise NotImplementedError

    def put_item_if_absent(self, key: str, item: dict[str, Any]) -> bool:
        if self.get_item(key) is not None:
            return False
        self.put_item(key, item)
        return True

    def get_last_seen_id(self) -> str | None:
        item = self.get_item("state#mentions")
        return str(item.get("last_seen_id")) if item and item.get("last_seen_id") else None

    def set_last_seen_id(self, tweet_id: str, now: dt.datetime) -> None:
        self.put_item(
            "state#mentions",
            {
                "pk": "state#mentions",
                "last_seen_id": tweet_id,
                "updated_at": iso_now(now),
            },
        )

    def has_replied(self, tweet_id: str) -> bool:
        return self.get_item(f"reply#{tweet_id}") is not None

    def has_recent_generation_failure(self, tweet_id: str, now: dt.datetime) -> bool:
        item = self.get_item(f"failed#{tweet_id}")
        if not item:
            return False
        expires_at = int(item.get("expires_at") or 0)
        return not expires_at or expires_at > int(now.timestamp())

    def mark_generation_failed(
        self,
        tweet_id: str,
        now: dt.datetime,
        *,
        reason: str,
        ttl_seconds: int = MENTION_FAILURE_TTL_SECONDS,
    ) -> None:
        self.put_item(
            f"failed#{tweet_id}",
            {
                "pk": f"failed#{tweet_id}",
                "tweet_id": tweet_id,
                "reason": reason[:500],
                "updated_at": iso_now(now),
                "expires_at": int(now.timestamp()) + ttl_seconds,
            },
        )

    def mark_replied(
        self,
        tweet_id: str,
        now: dt.datetime,
        *,
        reply_id: str | None,
        dry_run: bool,
    ) -> None:
        self.put_item(
            f"reply#{tweet_id}",
            {
                "pk": f"reply#{tweet_id}",
                "tweet_id": tweet_id,
                "reply_id": reply_id or "",
                "dry_run": dry_run,
                "updated_at": iso_now(now),
                "expires_at": int(now.timestamp()) + 90 * 24 * 60 * 60,
            },
        )

    def get_counter(self, key: str) -> int:
        item = self.get_item(key)
        if not item:
            return 0
        return int(item.get("count") or 0)

    def set_counter(self, key: str, count: int, now: dt.datetime, ttl_seconds: int) -> None:
        self.put_item(
            key,
            {
                "pk": key,
                "count": count,
                "updated_at": iso_now(now),
                "expires_at": int(now.timestamp()) + ttl_seconds,
            },
        )

    def get_standalone_post(self, post_id: str) -> dict[str, Any] | None:
        return self.get_item(f"standalone#{post_id}")

    def reserve_standalone_post(
        self,
        post_id: str,
        now: dt.datetime,
        *,
        tweet: dict[str, Any],
    ) -> bool:
        key = f"standalone#{post_id}"
        return self.put_item_if_absent(
            key,
            {
                "pk": key,
                "post_id": post_id,
                "status": "posting",
                "text": str(tweet.get("text") or ""),
                "engine": str(tweet.get("engine") or ""),
                "quality_score": str(tweet.get("quality_score") or ""),
                "sigil": public_sigil_metadata(tweet.get("sigil")),
                "created_at": iso_now(now),
                "updated_at": iso_now(now),
                "expires_at": int(now.timestamp()) + STANDALONE_POST_TTL_SECONDS,
            },
        )

    def mark_standalone_posted(
        self,
        post_id: str,
        now: dt.datetime,
        *,
        tweet: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        key = f"standalone#{post_id}"
        self.put_item(
            key,
            {
                "pk": key,
                "post_id": post_id,
                "status": "posted",
                "tweet_id": str(data.get("id") or ""),
                "text": str(tweet.get("text") or ""),
                "engine": str(tweet.get("engine") or ""),
                "quality_score": str(tweet.get("quality_score") or ""),
                "sigil": public_sigil_metadata(tweet.get("sigil")),
                "posted_at": iso_now(now),
                "updated_at": iso_now(now),
                "response": response,
                "expires_at": int(now.timestamp()) + STANDALONE_POST_TTL_SECONDS,
            },
        )

    def mark_standalone_failed(
        self,
        post_id: str,
        now: dt.datetime,
        *,
        tweet: dict[str, Any],
        error: str,
    ) -> None:
        key = f"standalone#{post_id}"
        self.put_item(
            key,
            {
                "pk": key,
                "post_id": post_id,
                "status": "failed",
                "text": str(tweet.get("text") or ""),
                "engine": str(tweet.get("engine") or ""),
                "quality_score": str(tweet.get("quality_score") or ""),
                "sigil": public_sigil_metadata(tweet.get("sigil")),
                "error": error[:500],
                "updated_at": iso_now(now),
                "expires_at": int(now.timestamp()) + STANDALONE_POST_TTL_SECONDS,
            },
        )


class FileStateStore(StateStore):
    def __init__(self, path: str):
        self.path = path
        self._items: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._items is not None:
            return self._items
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                self._items = json.load(handle)
        else:
            self._items = {}
        return self._items

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._load(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.path)

    def get_item(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def put_item(self, key: str, item: dict[str, Any]) -> None:
        self._load()[key] = item
        self._save()


class DynamoStateStore(StateStore):
    def __init__(self, table_name: str):
        if boto3 is None:
            raise BotConfigError("boto3 is required for DynamoDB state")
        self.table = boto3.resource("dynamodb").Table(table_name)

    def get_item(self, key: str) -> dict[str, Any] | None:
        response = self.table.get_item(Key={"pk": key}, ConsistentRead=True)
        return response.get("Item")

    def put_item(self, key: str, item: dict[str, Any]) -> None:
        item = dict(item)
        item["pk"] = key
        self.table.put_item(Item=item)

    def put_item_if_absent(self, key: str, item: dict[str, Any]) -> bool:
        item = dict(item)
        item["pk"] = key
        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk)",
            )
            return True
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            if error.get("Code") == "ConditionalCheckFailedException":
                return False
            raise


def make_state_store(config: BotConfig) -> StateStore:
    if config.state_table:
        return DynamoStateStore(config.state_table)
    return FileStateStore(os.getenv("X_STATE_FILE", "/tmp/crowley-bard-x-state.json"))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BotConfigError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def archive_training_event(
    config: BotConfig,
    mention: dict[str, Any],
    author: dict[str, Any],
    *,
    now: dt.datetime,
    reply_text: str,
    reply_engine: str,
    dry_run: bool,
    reply_id: str | None = None,
    sigil: dict[str, Any] | None = None,
) -> str | None:
    if not config.context_archive_s3_uri:
        return None
    if boto3 is None:
        return "boto3 unavailable for S3 archive"
    try:
        bucket, prefix = parse_s3_uri(config.context_archive_s3_uri)
        tweet_id = str(mention.get("id") or uuid.uuid4().hex)
        day = now.strftime("%Y-%m-%d")
        key_parts = [part.strip("/") for part in [prefix, day, f"{tweet_id}.json"] if part]
        key = "/".join(key_parts)
        payload = {
            "schema": "nsrl.x_bot.training_event.v1",
            "archived_at": iso_now(now),
            "day": day,
            "source": "x_api_mentions",
            "dry_run": dry_run,
            "bot_handle": config.bot_handle,
            "mention": {
                "id": tweet_id,
                "text": str(mention.get("text") or ""),
                "created_at": mention.get("created_at"),
                "conversation_id": mention.get("conversation_id"),
                "author_id": str(mention.get("author_id") or ""),
                "author_username": strip_at(str(author.get("username") or "")),
                "author_name": str(author.get("name") or ""),
            },
            "reply": {
                "id": reply_id or "",
                "text": reply_text,
                "engine": reply_engine,
                "sigil": public_sigil_metadata(sigil),
            },
        }
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=(json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
            ContentType="application/json",
        )
        return None
    except Exception as exc:  # pragma: no cover - network/AWS edge.
        return str(exc)


STOPWORDS = {
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
GLUE_WORDS = STOPWORDS | {
    "all",
    "also",
    "am",
    "away",
    "be",
    "come",
    "do",
    "doth",
    "down",
    "even",
    "ever",
    "give",
    "had",
    "hath",
    "hear",
    "here",
    "know",
    "let",
    "made",
    "make",
    "may",
    "might",
    "more",
    "most",
    "much",
    "must",
    "never",
    "no",
    "now",
    "one",
    "own",
    "say",
    "see",
    "shall",
    "should",
    "still",
    "take",
    "than",
    "then",
    "these",
    "thing",
    "things",
    "those",
    "thou",
    "thy",
    "unto",
    "up",
    "upon",
    "were",
    "would",
    "yet",
}

DOMAIN_LINES = {
    "train": "Training is a chapel where the weights learn to confess.",
    "model": "The model eats integers and dreams in small brass syllables.",
    "image": "The image is a lantern nailed to a circuit-board moon.",
    "god": "A god arrives only as a checksum with a halo problem.",
    "angel": "The angel says yes, then files the answer under thunder.",
    "love": "Love is the one loss function that refuses to converge quietly.",
    "soul": "The soul is a byte with stage fright and excellent handwriting.",
    "chaos": "Chaos is merely order wearing too much eyeliner.",
    "reply": "Every reply is a tiny seance with a character limit.",
    "bot": "The bot is awake, but only in the way a candle is awake.",
    "crowley": "Crowley knocks; Blake answers; the server returns 200 and smoke.",
    "blake": "Blake opens the window and the integers start singing.",
    "wiki": "The wiki says citation needed; the oracle says citation bleeding.",
}

OPENERS = [
    "I hear {topic} scratching at the underside of the visible world.",
    "The oracle receives {topic} and refuses to behave.",
    "{topic_title}: a small thunderclap in a scholar's pocket.",
    "Regarding {topic}: the manuscript has begun to perspire.",
    "I consulted the brass index of improper visions about {topic}.",
]

MIDDLES = [
    "Blake hands it a lantern; Crowley demands a receipt.",
    "The machine returns a feather made of integers.",
    "A little wiki-angel annotates the wound and calls it knowledge.",
    "The page burns politely, as educated pages do.",
    "I count three omens, two errors, and one magnificent typo.",
]

CLOSERS = [
    "Proceed, but salt the threshold.",
    "Translate it into action before it becomes theology.",
    "Do not fear the glitch; fear the tidy explanation.",
    "Make it smaller, stranger, and more exact.",
    "The answer is yes, provided yes arrives wearing boots.",
]


def clean_mention_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = strip_public_mentions(text)
    return text


def extract_keywords(text: str, bot_handle: str) -> list[str]:
    bot_handle = strip_at(bot_handle).lower()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']{2,}", text.lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("'")
        if not token or token in STOPWORDS or token == bot_handle:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords[:5]


def trim_to_limit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    shortened = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{shortened}."


def make_template_reply_text(
    mention: dict[str, Any],
    username: str,
    bot_handle: str,
    *,
    max_chars: int,
) -> str:
    raw_text = str(mention.get("text") or "")
    clean_text = clean_mention_text(raw_text)
    keywords = extract_keywords(clean_text, bot_handle)
    topic = keywords[0] if keywords else "signal"
    seed_material = f"{mention.get('id')}|{raw_text}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    domain_line = None
    for keyword in keywords:
        if keyword in DOMAIN_LINES:
            domain_line = DOMAIN_LINES[keyword]
            break
    opener = rng.choice(OPENERS).format(topic=topic, topic_title=topic.title())
    middle = domain_line or rng.choice(MIDDLES)
    closer = rng.choice(CLOSERS)
    body = f"{opener} {middle} {closer}"
    if username:
        body = f"@{username} {body}"
    return trim_to_limit(body, max_chars)


def clean_generated_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]+", " ", text)
    text = strip_public_mentions(text)
    text = re.sub(r"^(?:out|output|reply|tweet)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.strip(" \t\r\n\"'")
    text = trim_generated_sentence_span(text, min_chars=56, max_chars=230)
    if text and text[-1] not in ".!?":
        text = f"{text}."
    return text


def trim_generated_sentence_span(text: str, *, min_chars: int, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= min_chars:
        return text
    for index, char in enumerate(text):
        if index < min_chars:
            continue
        if char in ".!?":
            return text[: index + 1].strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space >= min_chars:
        return cut[:last_space].strip()
    return cut.strip()


def max_word_run(words: list[str], word_set: set[str]) -> int:
    current = 0
    best = 0
    for word in words:
        if word.strip("'") in word_set:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


WEAK_PUBLIC_TWEET_START_RE = re.compile(
    r"^(?:against|and|but|down|face|out|thee|well|whose)\b[:;,\s]*",
    re.IGNORECASE,
)
BROKEN_PUBLIC_TWEET_PHRASE_RE = re.compile(
    r"\b(?:by came|brain-indeed|come would|mouths sing ha)\b",
    re.IGNORECASE,
)
DANGLING_PUBLIC_TWEET_END_RE = re.compile(
    r"\b(?:a|an|and|as|but|for|from|in|my|of|or|our|than|that|the|their|thy|to|which|who|whose|with|your)[.!?]?$",
    re.IGNORECASE,
)
ABRUPT_COORDINATE_END_RE = re.compile(
    r"\b(?:and|or|but)\s+(?:death|eyes?|face|heart|life|light|soul)[.!?]?$",
    re.IGNORECASE,
)


def score_public_tweet_text(text: str) -> dict[str, Any]:
    text = clean_generated_text(text)
    words = re.findall(r"[A-Za-z][A-Za-z']*", text.lower())
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    max_repeat = max(counts.values(), default=0)
    unique_ratio = len(counts) / max(1, len(words))
    content_words = [
        word for word in words if len(word) > 2 and word.strip("'") not in STOPWORDS
    ]
    content_ratio = len(content_words) / max(1, len(words))
    glue_ratio = len([word for word in words if word.strip("'") in GLUE_WORDS]) / max(
        1, len(words)
    )
    glue_run = max_word_run(words, GLUE_WORDS)
    heavy_punctuation = len(re.findall(r"[,;:]", text))
    sentence_count = len(re.findall(r"[.!?]", text))
    expressive_count = len(re.findall(r"[!?]", text))
    punctuation_runs = len(re.findall(r"[!?.,;:]{2,}", text))
    avg_word_len = sum(len(word.strip("'")) for word in words) / max(1, len(words))
    reasons: list[str] = []
    score = 50

    if not text:
        reasons.append("empty")
        return {
            "score": 0,
            "ok": False,
            "text": text,
            "reasons": reasons,
            "words": 0,
            "unique_ratio": 0.0,
            "content_ratio": 0.0,
        }
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
    if glue_ratio > 0.72:
        score -= 28
        reasons.append("glue heavy")
    elif glue_ratio > 0.52:
        score -= int((glue_ratio - 0.52) * 80)
        reasons.append("glue weighted")
    if glue_run > 8:
        score -= 18
        reasons.append("glue run")
    elif glue_run > 5:
        score -= (glue_run - 5) * 5
        reasons.append("long glue run")
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
    if expressive_count > 3:
        score -= 12
        reasons.append("expressive punctuation heavy")
    if punctuation_runs:
        score -= 14
        reasons.append("punctuation run")
    if 3.2 <= avg_word_len <= 6.2:
        score += 3
        reasons.append("readable word shape")
    if max_repeat > 3:
        score -= 10
        reasons.append("word repeats")
    if WEAK_PUBLIC_TWEET_START_RE.search(text):
        score -= 12
        reasons.append("weak opening")
    if BROKEN_PUBLIC_TWEET_PHRASE_RE.search(text):
        score -= 18
        reasons.append("broken phrase")
    if DANGLING_PUBLIC_TWEET_END_RE.search(text):
        score -= 12
        reasons.append("dangling ending")
    if ABRUPT_COORDINATE_END_RE.search(text):
        score -= 16
        reasons.append("abrupt coordinated ending")
    if text[-1] in ".!?":
        score += 4
        reasons.append("complete sentence")

    score = max(0, min(100, score))
    return {
        "score": score,
        "ok": score >= 48,
        "text": text,
        "reasons": reasons,
        "chars": len(text),
        "words": len(words),
        "unique_ratio": round(unique_ratio, 3),
        "content_ratio": round(content_ratio, 3),
        "glue_ratio": round(glue_ratio, 3),
        "glue_run": glue_run,
        "heavy_punctuation": heavy_punctuation,
        "sentence_count": sentence_count,
        "expressive_count": expressive_count,
        "punctuation_runs": punctuation_runs,
        "max_repeat": max_repeat,
    }


def stable_sample_seed(mention: dict[str, Any]) -> int:
    raw_text = str(mention.get("text") or "")
    seed_material = f"{mention.get('id')}|{raw_text}"
    return int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)


def nsrl_prompt_for_mention(mention: dict[str, Any]) -> str:
    prompt = clean_mention_text(str(mention.get("text") or ""))
    prompt = prompt[:180].strip()
    if prompt:
        return prompt
    return "answer the reply in a strange oracle voice"


def context_for_adaptation(mention: dict[str, Any], config: BotConfig) -> str:
    text = str(mention.get("context") or mention.get("text") or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = strip_public_mentions(text)
    return text[: config.context_max_chars].strip()


def run_checked_command(
    cmd: list[str], *, timeout_seconds: int, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReplyGenerationError(f"{label} timed out after {timeout_seconds}s") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[:600]
        stdout = completed.stdout.strip()[:600]
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise ReplyGenerationError(f"{label} failed: {detail}")
    return completed


def stable_sigil_seed(source_id: str, text: str) -> str:
    material = f"{source_id}|{text}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:96] or uuid.uuid4().hex


def public_sigil_metadata(sigil: dict[str, Any] | None) -> dict[str, Any] | None:
    if sigil is None:
        return None
    return {
        key: value
        for key, value in sigil.items()
        if key not in {"png_bytes", "pgm_bytes"}
    }


def generate_solomon_sigil(
    text: str,
    *,
    source_id: str,
    config: BotConfig,
) -> dict[str, Any] | None:
    if not config.sigil_enabled:
        return None
    prompt = strip_public_mentions(text) or text
    condition_mode = "text-index"
    condition_args = ["--text-index", config.sigil_text_index]
    if config.sigil_latent_model and os.path.exists(config.sigil_latent_model):
        condition_mode = "latent-model"
        condition_args = ["--latent-model", config.sigil_latent_model]
    missing = [
        path
        for path in [config.sigil_bin, config.sigil_model, condition_args[1]]
        if not os.path.exists(path)
    ]
    if missing:
        raise ReplyGenerationError(f"missing Solomon sigil asset: {', '.join(missing)}")
    if not os.access(config.sigil_bin, os.X_OK):
        raise ReplyGenerationError(f"Solomon sampler is not executable: {config.sigil_bin}")

    seed = stable_sigil_seed(source_id, text)
    out_dir = f"/tmp/solomon-sigil-{safe_file_stem(source_id)}"
    os.makedirs(out_dir, exist_ok=True)
    for name in ["samples.pgm", "samples.ink128.u8", "trace.json", "sigil.png"]:
        try:
            os.unlink(os.path.join(out_dir, name))
        except FileNotFoundError:
            pass

    run_checked_command(
        [
            config.sigil_bin,
            "--model",
            config.sigil_model,
            *condition_args,
            "--prompt",
            prompt,
            "--seed",
            seed,
            "--out-dir",
            out_dir,
            "--samples",
            "1",
            "--candidate-multiplier",
            str(config.sigil_candidates),
            "--preview-columns",
            "1",
            "--init",
            "seal-prior",
            "--passes",
            str(config.sigil_passes),
        ],
        timeout_seconds=config.sigil_timeout_seconds,
        label="Solomon sigil generation",
    )

    pgm_path = os.path.join(out_dir, "samples.pgm")
    png_path = os.path.join(out_dir, "sigil.png")
    with open(pgm_path, "rb") as handle:
        pgm_bytes = handle.read()
    png_bytes, width, height = pgm_bytes_to_png(pgm_bytes)
    with open(png_path, "wb") as handle:
        handle.write(png_bytes)

    trace: dict[str, Any] = {}
    try:
        with open(os.path.join(out_dir, "trace.json"), "r", encoding="utf-8") as handle:
            trace = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        trace = {}

    return {
        "schema": SIGIL_SCHEMA,
        "seed": seed,
        "source_id": source_id,
        "prompt": prompt,
        "condition": condition_mode,
        "latent_model": config.sigil_latent_model if condition_mode == "latent-model" else "",
        "png_path": png_path,
        "pgm_path": pgm_path,
        "trace_path": os.path.join(out_dir, "trace.json"),
        "bytes": len(png_bytes),
        "width": width,
        "height": height,
        "target_name": str(trace.get("text_target_name") or ""),
        "target_number": str(trace.get("text_target_number") or ""),
        "target_score": str(trace.get("text_target_score") or ""),
        "text_distance": str(trace.get("selected_min_text_distance") or ""),
        "png_bytes": png_bytes,
    }


def pgm_bytes_to_png(pgm: bytes) -> tuple[bytes, int, int]:
    width, height, pixels = parse_binary_pgm(pgm)
    raw = bytearray()
    row_len = width
    for row in range(height):
        raw.append(0)
        start = row * row_len
        raw.extend(pixels[start : start + row_len])
    compressed = zlib.compress(bytes(raw), level=9)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x00\x00\x00\x00"))
    png.extend(png_chunk(b"IDAT", compressed))
    png.extend(png_chunk(b"IEND", b""))
    return bytes(png), width, height


def png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return len(data).to_bytes(4, "big") + kind + data + crc.to_bytes(4, "big")


def parse_binary_pgm(pgm: bytes) -> tuple[int, int, bytes]:
    offset = 0

    def next_token() -> bytes:
        nonlocal offset
        while offset < len(pgm) and pgm[offset] in b" \t\r\n":
            offset += 1
        if offset < len(pgm) and pgm[offset] == ord("#"):
            while offset < len(pgm) and pgm[offset] not in b"\r\n":
                offset += 1
            return next_token()
        start = offset
        while offset < len(pgm) and pgm[offset] not in b" \t\r\n":
            offset += 1
        return pgm[start:offset]

    magic = next_token()
    if magic != b"P5":
        raise ReplyGenerationError("Solomon sampler wrote non-binary PGM")
    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    if max_value != 255:
        raise ReplyGenerationError(f"unsupported PGM max value: {max_value}")
    while offset < len(pgm) and pgm[offset] in b" \t\r\n":
        offset += 1
        break
    expected = width * height
    pixels = pgm[offset : offset + expected]
    if len(pixels) != expected:
        raise ReplyGenerationError(
            f"PGM pixel payload has {len(pixels)} bytes, expected {expected}"
        )
    return width, height, pixels


def upload_sigil_if_needed(
    client: XOAuth1Client, sigil: dict[str, Any] | None
) -> list[str]:
    if sigil is None:
        return []
    media_id = client.upload_media_png(bytes(sigil["png_bytes"]))
    sigil["media_id"] = media_id
    return [media_id]


def adapt_model_for_context(mention: dict[str, Any], config: BotConfig) -> tuple[str, str]:
    if not config.context_adapt:
        return config.nsrl_model, config.nsrl_tokens
    missing = [
        path
        for path in [config.nsrl_corpus_bin, config.nsrl_model, config.nsrl_vocab]
        if not os.path.exists(path)
    ]
    if missing:
        raise ReplyGenerationError(f"missing context-adaptation asset: {', '.join(missing)}")
    if not os.access(config.nsrl_corpus_bin, os.X_OK):
        raise ReplyGenerationError(
            f"nsrl corpus binary is not executable: {config.nsrl_corpus_bin}"
        )

    context = context_for_adaptation(mention, config)
    if len(context) < 32:
        return config.nsrl_model, config.nsrl_tokens

    tweet_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(mention.get("id") or uuid.uuid4().hex))
    context_path = f"/tmp/nsrl-context-{tweet_id}.txt"
    context_tokens_path = f"/tmp/nsrl-context-{tweet_id}.tokens.u16"
    context_prior_tokens_path = f"/tmp/nsrl-context-{tweet_id}.prior.tokens.u16"
    context_trace_path = f"/tmp/nsrl-context-{tweet_id}.tokens.trace.jsonl"
    adapted_model_path = f"/tmp/nsrl-context-{tweet_id}.nsrllm"
    adapted_trace_path = f"/tmp/nsrl-context-{tweet_id}.adapt.trace.jsonl"
    for path in [
        context_path,
        context_tokens_path,
        context_prior_tokens_path,
        context_trace_path,
        adapted_model_path,
        adapted_trace_path,
    ]:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    repeated_context = "\n\n".join([context] * config.context_repeat_count)
    with open(context_path, "w", encoding="utf-8") as handle:
        handle.write(repeated_context)
        handle.write("\n")

    run_checked_command(
        [
            config.nsrl_corpus_bin,
            "lexeme-tokenize-fixed-vocab",
            "--corpus",
            context_path,
            "--vocab",
            config.nsrl_vocab,
            "--tokens-out",
            context_tokens_path,
            "--trace",
            context_trace_path,
            "--seq-len",
            "32",
            "--stride",
            "1",
        ],
        timeout_seconds=config.context_adapt_timeout_seconds,
        label="context tokenization",
    )
    if os.path.getsize(context_tokens_path) < 2 * 10:
        return config.nsrl_model, config.nsrl_tokens

    with open(context_prior_tokens_path, "wb") as out:
        with open(config.nsrl_tokens, "rb") as base:
            out.write(base.read())
        with open(context_tokens_path, "rb") as context_tokens:
            context_bytes = context_tokens.read()
        for _ in range(config.context_repeat_count):
            out.write(context_bytes)

    run_checked_command(
        [
            config.nsrl_bin,
            "--mode",
            "lexeme-softmax",
            "--tokens",
            context_tokens_path,
            "--vocab",
            config.nsrl_vocab,
            "--model",
            config.nsrl_model,
            "--model-out",
            adapted_model_path,
            "--trace",
            adapted_trace_path,
            "--seq-len",
            "8",
            "--stride",
            "1",
            "--max-windows",
            str(config.context_adapt_max_windows),
            "--epochs",
            "1",
            "--lr-shift",
            str(config.context_adapt_lr_shift),
            "--max-lr-shift",
            str(config.context_adapt_lr_shift + 2),
            "--max-weight-delta",
            "1",
            "--target-frequency-cap",
            "4096",
            "--frequency-weight-min-q15",
            "4096",
            "--quality-weight-profile",
            "cruft-aware",
        ],
        timeout_seconds=config.context_adapt_timeout_seconds,
        label="context adaptation",
    )
    return adapted_model_path, context_prior_tokens_path


def generate_nsrl_reply_body(mention: dict[str, Any], config: BotConfig) -> tuple[str, str]:
    missing = [
        path
        for path in [config.nsrl_bin, config.nsrl_model, config.nsrl_vocab, config.nsrl_tokens]
        if not os.path.exists(path)
    ]
    if missing:
        raise ReplyGenerationError(f"missing live inference asset: {', '.join(missing)}")
    if not os.access(config.nsrl_bin, os.X_OK):
        raise ReplyGenerationError(f"nsrl binary is not executable: {config.nsrl_bin}")

    generation_model, generation_tokens = adapt_model_for_context(mention, config)
    generation_mode = "context-adapted" if generation_model != config.nsrl_model else "base"
    tweet_id = str(mention.get("id") or uuid.uuid4().hex)
    text_out = f"/tmp/nsrl-reply-{tweet_id}.txt"
    trace_out = f"/tmp/nsrl-reply-{tweet_id}.trace.jsonl"
    for path in [text_out, trace_out]:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    cmd = [
        config.nsrl_bin,
        "--mode",
        "lexeme-generate",
        "--model",
        generation_model,
        "--vocab",
        config.nsrl_vocab,
        "--tokens",
        generation_tokens,
        "--prompt",
        nsrl_prompt_for_mention(mention),
        "--max-new-tokens",
        str(config.nsrl_max_new_tokens),
        "--decode-profile",
        "coherent-prose",
        "--decode-function-word-run-cap",
        "4",
        "--sample-seed",
        str(stable_sample_seed(mention)),
        "--top-k",
        str(config.nsrl_top_k),
        "--corpus-prior",
        "--corpus-prior-logit-shift",
        "9",
        "--corpus-prior-order",
        "3",
        "--decode-frequency-cap",
        "600",
        "--decode-frequency-min-q15",
        "6144",
        "--decode-frequency-logit-shift",
        "5",
        "--decode-local-frequency-cap",
        "2",
        "--decode-local-frequency-min-q15",
        "8192",
        "--decode-local-frequency-logit-shift",
        "4",
        "--decode-local-frequency-hard-cap",
        "2",
        "--repeat-window",
        "64",
        "--repeat-penalty-shift",
        "4",
        "--max-repeat-run",
        "2",
        "--no-repeat-ngram",
        "3",
        "--quality-weight-profile",
        "prose-aware",
        "--generated-only",
        "--text-out",
        text_out,
        "--trace",
        trace_out,
    ]
    for token in DECODE_BANNED_TOKENS:
        cmd.extend(["--decode-ban-token", token])
    run_checked_command(
        cmd, timeout_seconds=config.nsrl_timeout_seconds, label="nsrl generation"
    )
    try:
        with open(text_out, "r", encoding="utf-8") as handle:
            generated = clean_generated_text(handle.read())
    except FileNotFoundError as exc:
        raise ReplyGenerationError("nsrl generation did not write text output") from exc
    if not generated:
        raise ReplyGenerationError("nsrl generation returned empty text")
    return generated, generation_mode


def make_reply(
    mention: dict[str, Any],
    username: str,
    config: BotConfig,
) -> dict[str, str]:
    engine = config.reply_engine.strip().lower()
    if engine in {"template", "templates"}:
        text = make_template_reply_text(
            mention, username, config.bot_handle, max_chars=config.max_reply_chars
        )
        return {"engine": "template", "text": text}
    if engine not in {"nsrl-live", "live", "nsrl"}:
        raise ReplyGenerationError(f"unknown reply engine: {config.reply_engine}")

    body, generation_mode = generate_nsrl_reply_body(mention, config)
    body = clean_generated_text(body)
    if not body:
        raise ReplyGenerationError("nsrl generation returned empty text after cleaning")
    prefix = f"@{username} " if username else ""
    text = trim_to_limit(f"{prefix}{body}", config.max_reply_chars)
    return {"engine": f"nsrl-live:{generation_mode}", "text": text}


def make_standalone_tweet(
    prompt: str, config: BotConfig, *, tweet_id: str
) -> dict[str, Any]:
    mention = {"id": tweet_id, "text": prompt, "author_id": "0"}
    engine = config.reply_engine.strip().lower()
    if engine in {"template", "templates"}:
        text = make_template_reply_text(
            mention,
            "",
            config.bot_handle,
            max_chars=config.max_reply_chars,
        )
        text = clean_generated_text(text)
        if not text:
            raise ReplyGenerationError("template generation returned empty text")
        quality = score_public_tweet_text(text)
        if quality["score"] < config.public_tweet_min_score:
            raise ReplyGenerationError(
                f"best standalone tweet scored {quality['score']} below "
                f"{config.public_tweet_min_score}: {quality['text']}"
            )
        return {
            "engine": "template",
            "text": text,
            "quality_score": str(quality["score"]),
        }
    if engine not in {"nsrl-live", "live", "nsrl"}:
        raise ReplyGenerationError(f"unknown reply engine: {config.reply_engine}")

    candidates: list[dict[str, Any]] = []
    generation_mode = "base"
    candidate_count = max(1, config.standalone_candidates)
    for candidate_index in range(candidate_count):
        candidate_mention = dict(mention)
        candidate_mention["id"] = f"{tweet_id}-{candidate_index + 1}"
        try:
            body, generation_mode = generate_nsrl_reply_body(candidate_mention, config)
        except ReplyGenerationError as exc:
            candidates.append(
                {
                    "index": candidate_index + 1,
                    "score": 0,
                    "text": "",
                    "error": str(exc),
                }
            )
            continue
        body = clean_generated_text(body)
        quality = score_public_tweet_text(body)
        candidates.append(
            {
                "index": candidate_index + 1,
                "score": quality["score"],
                "text": quality["text"],
                "reasons": quality["reasons"],
            }
        )
    best = max(candidates, key=lambda candidate: int(candidate.get("score") or 0))
    if int(best.get("score") or 0) < config.public_tweet_min_score:
        raise ReplyGenerationError(
            f"best standalone tweet scored {best.get('score')} below "
            f"{config.public_tweet_min_score}: {best.get('text') or 'no text'}"
        )
    return {
        "engine": f"nsrl-live:{generation_mode}",
        "text": trim_to_limit(str(best["text"]), config.max_reply_chars),
        "quality_score": str(best["score"]),
        "candidate_count": str(candidate_count),
        "selected_candidate": str(best["index"]),
        "candidates": candidates,
    }


def clean_precomposed_tweet_text(text: str, *, max_chars: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]+", " ", str(text))
    lines = [
        normalize_public_text(line)
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ReplyGenerationError("precomposed tweet text is empty")
    if "@" in text:
        raise ReplyGenerationError(
            "precomposed tweet text must not contain public handles"
        )
    if "http" in text.lower():
        raise ReplyGenerationError("precomposed tweet text must not contain URLs")
    if len(text) > max_chars:
        raise ReplyGenerationError(
            f"precomposed tweet text is {len(text)} chars; max is {max_chars}"
        )
    return text


def make_precomposed_standalone_tweet(
    text: str, config: BotConfig
) -> dict[str, Any]:
    text = clean_precomposed_tweet_text(text, max_chars=config.max_reply_chars)
    return {
        "engine": "precomposed",
        "text": text,
        "quality_score": "manual",
    }


def standalone_prompt_from_event(event: dict[str, Any]) -> str | None:
    missing = object()
    value = event.get("post_tweet", missing)
    if value is missing:
        value = event.get("standalone_tweet", missing)
    if value is missing:
        return None
    if isinstance(value, bool):
        if not value:
            return None
    elif isinstance(value, str):
        if not value.strip():
            return None
        return value.strip()
    elif not value:
        return None
    prompt = event.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return "the omen today is"


def standalone_text_from_event(event: dict[str, Any]) -> str | None:
    for key in ("text", "tweet_text", "status_text"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def standalone_post_from_state(item: dict[str, Any]) -> dict[str, Any]:
    status = str(item.get("status") or "unknown")
    tweet = {
        "engine": str(item.get("engine") or ""),
        "text": str(item.get("text") or ""),
        "quality_score": str(item.get("quality_score") or ""),
    }
    if item.get("sigil"):
        tweet["sigil"] = item["sigil"]
    result: dict[str, Any] = {
        "ok": status == "posted",
        "dry_run": False,
        "posted": False,
        "duplicate": True,
        "status": status,
        "tweet": tweet,
    }
    if item.get("tweet_id"):
        result["tweet_id"] = str(item["tweet_id"])
    if status == "posting":
        result["error"] = "standalone_post_in_progress"
    elif status == "failed":
        result["error"] = "standalone_post_failed"
        result["detail"] = str(item.get("error") or "")
    return result


def is_global_generation_error(exc: ReplyGenerationError) -> bool:
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in [
            "missing live inference asset",
            "missing context-adaptation asset",
            "nsrl binary is not executable",
            "nsrl corpus binary is not executable",
            "missing solomon sigil asset",
            "solomon sampler is not executable",
            "unknown reply engine",
        ]
    )


def users_by_id(includes: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    users = (includes or {}).get("users") or []
    return {str(user.get("id")): user for user in users if user.get("id")}


def counter_keys(now: dt.datetime) -> list[tuple[str, int, str]]:
    minute = (now.minute // 15) * 15
    window = now.replace(minute=minute, second=0, microsecond=0)
    day = now.strftime("%Y%m%d")
    month = now.strftime("%Y%m")
    return [
        ("15m", 2 * 60 * 60, window.strftime("counter#15m#%Y%m%dT%H%MZ")),
        ("day", 45 * 24 * 60 * 60, f"counter#day#{day}"),
        ("month", 120 * 24 * 60 * 60, f"counter#month#{month}"),
    ]


def consume_reply_budget(
    store: StateStore, config: BotConfig, now: dt.datetime
) -> tuple[bool, str | None]:
    limits = {
        "15m": config.max_replies_per_15m,
        "day": config.max_replies_per_day,
        "month": config.max_replies_per_month,
    }
    keys = counter_keys(now)
    for label, _ttl, key in keys:
        if limits[label] <= 0:
            return False, f"{label} limit is zero"
        if store.get_counter(key) >= limits[label]:
            return False, f"{label} limit reached"
    for label, ttl, key in keys:
        store.set_counter(key, store.get_counter(key) + 1, now, ttl)
    return True, None


def should_skip_mention(
    mention: dict[str, Any],
    *,
    config: BotConfig,
    store: StateStore,
    now: dt.datetime,
) -> str | None:
    tweet_id = str(mention.get("id") or "")
    author_id = str(mention.get("author_id") or "")
    text = str(mention.get("text") or "")
    if not tweet_id:
        return "missing tweet id"
    if author_id and config.bot_user_id and author_id == config.bot_user_id:
        return "own tweet"
    if store.has_replied(tweet_id):
        return "already replied"
    if store.has_recent_generation_failure(tweet_id, now):
        return "generation previously failed"
    if config.direct_mentions_only and config.bot_handle:
        if f"@{config.bot_handle.lower()}" not in text.lower():
            return "not a direct handle mention"
    if len(PUBLIC_MENTION_RE.findall(text)) > 6:
        return "mention pile-on"
    if re.fullmatch(r"[\W_]+", clean_mention_text(text) or ""):
        return "empty mention"
    return None


def process_mentions(
    *,
    client: XOAuth1Client,
    store: StateStore,
    config: BotConfig,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    if not config.bot_user_id:
        me = client.get_me().get("data") or {}
        config.bot_user_id = str(me.get("id") or "")
        if not config.bot_handle and me.get("username"):
            config.bot_handle = strip_at(str(me["username"]))
    if not config.bot_user_id:
        raise BotConfigError("Set X_BOT_USER_ID or use credentials that can call /2/users/me")

    last_seen_id = store.get_last_seen_id()
    response = client.get_mentions(
        config.bot_user_id,
        since_id=last_seen_id,
        max_results=config.max_mentions_per_poll,
    )
    mentions = list(response.get("data") or [])
    mentions.sort(key=lambda tweet: int(tweet["id"]))
    users = users_by_id(response.get("includes"))
    result: dict[str, Any] = {
        "ok": True,
        "dry_run": config.dry_run,
        "last_seen_id": last_seen_id,
        "fetched": len(mentions),
        "replied": 0,
        "skipped": [],
        "would_reply": [],
        "posted": [],
        "bootstrapped": False,
        "archive_errors": [],
        "updated_last_seen_id": last_seen_id,
    }

    newest_seen = None
    for mention in mentions:
        newest_seen = max_id(newest_seen, str(mention.get("id")))

    if not last_seen_id and newest_seen and not config.bootstrap_reply:
        store.set_last_seen_id(newest_seen, now)
        result["bootstrapped"] = True
        result["updated_last_seen_id"] = newest_seen
        return result

    last_processed_id = last_seen_id
    for mention in mentions:
        tweet_id = str(mention.get("id") or "")
        skip_reason = should_skip_mention(
            mention, config=config, store=store, now=now
        )
        if skip_reason:
            result["skipped"].append({"id": tweet_id, "reason": skip_reason})
            last_processed_id = max_id(last_processed_id, tweet_id)
            continue

        if result["replied"] >= config.max_replies_per_run:
            result["skipped"].append({"id": tweet_id, "reason": "run reply cap reached"})
            break

        author = users.get(str(mention.get("author_id") or ""), {})
        username = strip_at(str(author.get("username") or ""))
        try:
            reply = make_reply(mention, username, config)
        except ReplyGenerationError as exc:
            result["skipped"].append(
                {"id": tweet_id, "reason": "reply generation failed", "detail": str(exc)}
            )
            if is_global_generation_error(exc):
                break
            store.mark_generation_failed(tweet_id, now, reason=str(exc))
            last_processed_id = max_id(last_processed_id, tweet_id)
            continue
        reply_text = reply["text"]
        reply_engine = reply["engine"]

        if config.dry_run:
            try:
                sigil = generate_solomon_sigil(
                    reply_text,
                    source_id=tweet_id,
                    config=config,
                )
            except ReplyGenerationError as exc:
                result["skipped"].append(
                    {"id": tweet_id, "reason": "sigil generation failed", "detail": str(exc)}
                )
                if is_global_generation_error(exc):
                    break
                store.mark_generation_failed(tweet_id, now, reason=str(exc))
                last_processed_id = max_id(last_processed_id, tweet_id)
                continue
            archive_error = archive_training_event(
                config,
                mention,
                author,
                now=now,
                reply_text=reply_text,
                reply_engine=reply_engine,
                dry_run=True,
                sigil=sigil,
            )
            if archive_error:
                result["archive_errors"].append({"id": tweet_id, "error": archive_error})
            result["would_reply"].append(
                {
                    "id": tweet_id,
                    "reply_engine": reply_engine,
                    "text": reply_text,
                    "sigil": public_sigil_metadata(sigil),
                }
            )
            result["replied"] += 1
            if config.advance_state_on_dry_run:
                store.mark_replied(tweet_id, now, reply_id=None, dry_run=True)
                last_processed_id = max_id(last_processed_id, tweet_id)
            continue

        allowed, reason = consume_reply_budget(store, config, now)
        if not allowed:
            result["skipped"].append({"id": tweet_id, "reason": reason or "rate limit"})
            break

        try:
            sigil = generate_solomon_sigil(reply_text, source_id=tweet_id, config=config)
            media_ids = upload_sigil_if_needed(client, sigil)
        except (ReplyGenerationError, XApiError) as exc:
            result["skipped"].append(
                {"id": tweet_id, "reason": "sigil upload failed", "detail": str(exc)}
            )
            if isinstance(exc, ReplyGenerationError) and is_global_generation_error(exc):
                break
            store.mark_generation_failed(tweet_id, now, reason=str(exc))
            last_processed_id = max_id(last_processed_id, tweet_id)
            continue

        post_response = client.post_reply_with_media(
            reply_text, tweet_id, media_ids=media_ids
        )
        reply_id = str((post_response.get("data") or {}).get("id") or "")
        archive_error = archive_training_event(
            config,
            mention,
            author,
            now=now,
            reply_text=reply_text,
            reply_engine=reply_engine,
            dry_run=False,
            reply_id=reply_id,
            sigil=sigil,
        )
        if archive_error:
            result["archive_errors"].append({"id": tweet_id, "error": archive_error})
        store.mark_replied(tweet_id, now, reply_id=reply_id, dry_run=False)
        last_processed_id = max_id(last_processed_id, tweet_id)
        result["posted"].append(
            {
                "id": tweet_id,
                "reply_id": reply_id,
                "reply_engine": reply_engine,
                "sigil": public_sigil_metadata(sigil),
            }
        )
        result["replied"] += 1

    if last_processed_id and id_gt(last_processed_id, last_seen_id):
        store.set_last_seen_id(last_processed_id, now)
        result["updated_last_seen_id"] = last_processed_id

    return result


def sanitized_x_error(exc: XApiError) -> dict[str, Any]:
    try:
        parsed = json.loads(exc.body)
    except json.JSONDecodeError:
        parsed = {"detail": exc.body[:500]}
    return {
        "ok": False,
        "error": "x_api_error",
        "status": exc.status,
        "title": parsed.get("title"),
        "detail": parsed.get("detail"),
        "type": parsed.get("type"),
    }


def lambda_handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    del context
    config = BotConfig.from_env()
    now = utc_now()
    if event:
        if "dry_run" in event:
            config.dry_run = bool(event["dry_run"])
        if "bootstrap_reply" in event:
            config.bootstrap_reply = bool(event["bootstrap_reply"])
        if "candidate_count" in event:
            config.standalone_candidates = bounded_int(
                event["candidate_count"],
                default=config.standalone_candidates,
                minimum=1,
                maximum=MAX_STANDALONE_CANDIDATES,
            )
        if "min_score" in event:
            config.public_tweet_min_score = bounded_int(
                event["min_score"],
                default=config.public_tweet_min_score,
                minimum=1,
                maximum=100,
            )
        has_standalone_directive = (
            "post_tweet" in event or "standalone_tweet" in event
        )
        standalone_prompt = standalone_prompt_from_event(event)
        standalone_text = (
            standalone_text_from_event(event) if standalone_prompt is not None else None
        )
        if (
            has_standalone_directive
            and standalone_text is None
            and standalone_prompt is None
        ):
            return {
                "ok": True,
                "dry_run": config.dry_run,
                "skipped": "standalone_post_disabled",
            }
        if standalone_text is not None or standalone_prompt is not None:
            post_id = str(event.get("id") or "")
            store: StateStore | None = None
            if not config.dry_run:
                if not post_id:
                    return {
                        "ok": False,
                        "dry_run": False,
                        "error": "missing_standalone_post_id",
                        "detail": "Live standalone posts require an explicit id for idempotency.",
                    }
                store = make_state_store(config)
                existing = store.get_standalone_post(post_id)
                if existing:
                    return standalone_post_from_state(existing)
            elif not post_id:
                post_id = uuid.uuid4().hex
            try:
                if standalone_text is not None:
                    tweet = make_precomposed_standalone_tweet(standalone_text, config)
                else:
                    assert standalone_prompt is not None
                    tweet = make_standalone_tweet(
                        standalone_prompt,
                        config,
                        tweet_id=post_id,
                    )
                sigil = generate_solomon_sigil(
                    tweet["text"], source_id=post_id, config=config
                )
                if sigil is not None:
                    tweet["sigil"] = public_sigil_metadata(sigil)
            except ReplyGenerationError as exc:
                return {
                    "ok": False,
                    "dry_run": config.dry_run,
                    "error": "tweet_generation_error",
                    "detail": str(exc),
                }
            if config.dry_run:
                return {"ok": True, "dry_run": True, "would_post": True, **tweet}
            secret = load_secret(config)
            credentials = OAuth1Credentials.from_secret(secret)
            client = XOAuth1Client(credentials)
            assert store is not None
            if not store.reserve_standalone_post(post_id, now, tweet=tweet):
                existing = store.get_standalone_post(post_id)
                if existing:
                    return standalone_post_from_state(existing)
                return {
                    "ok": False,
                    "dry_run": False,
                    "error": "standalone_post_reservation_failed",
                }
            try:
                media_ids = upload_sigil_if_needed(client, sigil)
                if sigil is not None:
                    tweet["sigil"] = public_sigil_metadata(sigil)
                response = client.post_tweet_with_media(tweet["text"], media_ids=media_ids)
            except (ReplyGenerationError, XApiError) as exc:
                store.mark_standalone_failed(post_id, now, tweet=tweet, error=str(exc))
                if isinstance(exc, XApiError):
                    error = sanitized_x_error(exc)
                    error["dry_run"] = False
                    return error
                return {
                    "ok": False,
                    "dry_run": False,
                    "error": "sigil_upload_failed",
                    "detail": str(exc),
                }
            store.mark_standalone_posted(
                post_id,
                utc_now(),
                tweet=tweet,
                response=response,
            )
            return {
                "ok": True,
                "dry_run": False,
                "posted": True,
                "post_id": post_id,
                "tweet": tweet,
                "response": response,
            }
        if "test_generate" in event:
            mention = {
                "id": str(event.get("id") or "1"),
                "text": str(event["test_generate"]),
                "author_id": str(event.get("author_id") or "0"),
            }
            try:
                reply = make_reply(
                    mention,
                    strip_at(str(event.get("username") or "tester")),
                    config,
                )
                return {"ok": True, "dry_run": True, **reply}
            except ReplyGenerationError as exc:
                return {
                    "ok": False,
                    "dry_run": True,
                    "error": "reply_generation_error",
                    "detail": str(exc),
                }
    secret = load_secret(config)
    credentials = OAuth1Credentials.from_secret(secret)
    client = XOAuth1Client(credentials)
    store = make_state_store(config)
    try:
        return process_mentions(client=client, store=store, config=config)
    except XApiError as exc:
        error = sanitized_x_error(exc)
        error["dry_run"] = config.dry_run
        return error


def main() -> None:
    config = BotConfig.from_env()
    secret = load_secret(config)
    credentials = OAuth1Credentials.from_secret(secret)
    result = process_mentions(
        client=XOAuth1Client(credentials),
        store=make_state_store(config),
        config=config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
