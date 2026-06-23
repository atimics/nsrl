#!/usr/bin/env python3
import datetime as dt
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import lambda_function as bot


class FakeClient:
    def __init__(self, mentions, users):
        self.mentions = mentions
        self.users = users
        self.replies = []
        self.tweets = []
        self.media_uploads = []

    def get_me(self):
        return {"data": {"id": "42", "username": "CrowleyBard"}}

    def get_mentions(self, bot_user_id, *, since_id, max_results):
        self.last_request = {
            "bot_user_id": bot_user_id,
            "since_id": since_id,
            "max_results": max_results,
        }
        return {"data": self.mentions, "includes": {"users": self.users}}

    def post_reply(self, text, in_reply_to_tweet_id):
        return self.post_reply_with_media(text, in_reply_to_tweet_id, media_ids=None)

    def post_reply_with_media(self, text, in_reply_to_tweet_id, *, media_ids=None):
        self.replies.append(
            {
                "text": text,
                "in_reply_to_tweet_id": in_reply_to_tweet_id,
                "media_ids": media_ids or [],
            }
        )
        return {"data": {"id": f"reply-{in_reply_to_tweet_id}", "text": text}}

    def post_tweet(self, text):
        return self.post_tweet_with_media(text, media_ids=None)

    def post_tweet_with_media(self, text, *, media_ids=None):
        tweet_id = f"tweet-{len(self.tweets) + 1}"
        self.tweets.append({"id": tweet_id, "text": text, "media_ids": media_ids or []})
        return {"data": {"id": tweet_id, "text": text}}

    def upload_media_png(self, png_bytes):
        media_id = f"media-{len(self.media_uploads) + 1}"
        self.media_uploads.append({"id": media_id, "bytes": png_bytes})
        return media_id


def config(**overrides):
    base = bot.BotConfig(
        bot_user_id="42",
        bot_handle="CrowleyBard",
        secret_id=None,
        state_table=None,
        context_archive_s3_uri="",
        dry_run=True,
        advance_state_on_dry_run=True,
        bootstrap_reply=False,
        direct_mentions_only=True,
        max_mentions_per_poll=10,
        max_replies_per_run=1,
        max_replies_per_15m=1,
        max_replies_per_day=10,
        max_replies_per_month=100,
        max_reply_chars=260,
        reply_engine="template",
        nsrl_bin="/missing/nsrl-train",
        nsrl_corpus_bin="/missing/nsrl-corpus",
        nsrl_model="/missing/v4096.nsrllm",
        nsrl_vocab="/missing/v4096.vocab.tsv",
        nsrl_tokens="/missing/v4096.tokens.u16",
        nsrl_max_new_tokens=48,
        nsrl_top_k=12,
        nsrl_timeout_seconds=12,
        context_adapt=True,
        context_max_chars=1800,
        context_repeat_count=3,
        context_adapt_max_windows=64,
        context_adapt_lr_shift=18,
        context_adapt_timeout_seconds=20,
        standalone_candidates=6,
        public_tweet_min_score=48,
        sigil_enabled=False,
        sigil_bin="/missing/nsrl-bitmap-sample",
        sigil_model="/missing/model.nsrltch",
        sigil_text_index="/missing/solomon-spirit-text-signatures.tsv",
        sigil_candidates=16,
        sigil_passes=8,
        sigil_timeout_seconds=12,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class MentionReplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = bot.FileStateStore(os.path.join(self.tmp.name, "state.json"))
        self.now = dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_run_bootstraps_without_replying(self):
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard what is the model?"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        result = bot.process_mentions(
            client=client, store=self.state, config=config(), now=self.now
        )
        self.assertTrue(result["bootstrapped"])
        self.assertEqual(result["replied"], 0)
        self.assertEqual(self.state.get_last_seen_id(), "100")
        self.assertEqual(client.replies, [])

    def test_dry_run_reply_is_contextual_and_advances_by_default(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard what is the model?"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        result = bot.process_mentions(
            client=client, store=self.state, config=config(), now=self.now
        )
        self.assertEqual(result["replied"], 1)
        reply_text = result["would_reply"][0]["text"]
        self.assertIn("@signal_summoner", reply_text)
        self.assertIn("model eats integers", reply_text)
        self.assertLessEqual(len(reply_text), 260)
        self.assertEqual(self.state.get_last_seen_id(), "100")
        self.assertTrue(self.state.has_replied("100"))

    def test_dry_run_can_be_configured_not_to_advance(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard what is the model?"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        result = bot.process_mentions(
            client=client,
            store=self.state,
            config=config(advance_state_on_dry_run=False),
            now=self.now,
        )
        self.assertEqual(result["replied"], 1)
        self.assertEqual(self.state.get_last_seen_id(), "99")

    def test_generated_text_strips_public_mentions(self):
        self.assertEqual(
            bot.clean_generated_text("@stray_handle, behold @second_handle"),
            "behold.",
        )
        self.assertEqual(bot.clean_generated_text("out: behold the omen"), "behold the omen.")

    def test_nsrl_reply_keeps_target_handle_only(self):
        with mock.patch.object(
            bot,
            "generate_nsrl_reply_body",
            return_value=("@intruder, behold @other_handle", "base"),
        ):
            reply = bot.make_reply(
                {"id": "101", "text": "@CrowleyBard say the thing"},
                "signal_summoner",
                config(reply_engine="nsrl-live"),
            )
        self.assertEqual(reply["text"], "@signal_summoner behold.")
        self.assertNotIn("@intruder", reply["text"])
        self.assertNotIn("@other_handle", reply["text"])

    def test_standalone_tweet_has_no_public_mentions(self):
        with mock.patch.object(
            bot,
            "generate_nsrl_reply_body",
            return_value=(
                "@intruder, morning light opens softly and the window finally answers @other_handle",
                "base",
            ),
        ):
            tweet = bot.make_standalone_tweet(
                "@CrowleyBard wake up",
                config(reply_engine="nsrl-live"),
                tweet_id="first-post",
            )
        self.assertEqual(
            tweet["text"], "morning light opens softly and the window finally answers."
        )
        self.assertNotIn("@", tweet["text"])

    def test_public_tweet_score_penalizes_glue_word_collapse(self):
        clean = bot.score_public_tweet_text(
            "silent delight returns through dark stars with a little fire."
        )
        glue = bot.score_public_tweet_text(
            "so let no love shall have you now upon thy life will let thee so shall love."
        )
        self.assertGreater(clean["score"], glue["score"])
        self.assertGreater(glue["glue_ratio"], clean["glue_ratio"])
        self.assertIn("glue", " ".join(glue["reasons"]))

    def test_false_post_tweet_event_does_not_select_standalone_prompt(self):
        self.assertIsNone(bot.standalone_prompt_from_event({"post_tweet": False}))
        self.assertIsNone(
            bot.standalone_prompt_from_event(
                {"post_tweet": False, "prompt": "do not publish this"}
            )
        )
        self.assertEqual(
            bot.standalone_prompt_from_event(
                {"post_tweet": True, "prompt": "publish this"}
            ),
            "publish this",
        )

    def test_false_post_tweet_event_is_noop(self):
        with (
            mock.patch.object(
                bot.BotConfig,
                "from_env",
                return_value=config(dry_run=False, reply_engine="nsrl-live"),
            ),
            mock.patch.object(bot, "process_mentions") as process_mentions,
        ):
            result = bot.lambda_handler(
                {
                    "post_tweet": False,
                    "dry_run": False,
                    "prompt": "do not publish",
                    "text": "do not publish this precomposed text",
                },
                None,
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["skipped"], "standalone_post_disabled")
        process_mentions.assert_not_called()

    def test_live_standalone_post_requires_id(self):
        with mock.patch.object(
            bot.BotConfig,
            "from_env",
            return_value=config(dry_run=False, reply_engine="nsrl-live"),
        ):
            result = bot.lambda_handler(
                {"post_tweet": True, "dry_run": False, "prompt": "publish this"},
                None,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_standalone_post_id")

    def test_precomposed_standalone_dry_run_uses_supplied_text(self):
        text = "Solomon checkpoint improved:\nEval top1 200/1000. #NSRL"
        with (
            mock.patch.object(
                bot.BotConfig,
                "from_env",
                return_value=config(dry_run=True, reply_engine="nsrl-live"),
            ),
            mock.patch.object(bot, "generate_nsrl_reply_body") as generate,
        ):
            result = bot.lambda_handler(
                {
                    "post_tweet": True,
                    "dry_run": True,
                    "id": "solomon-checkpoint-eval-200",
                    "text": text,
                },
                None,
            )
        self.assertTrue(result["would_post"])
        self.assertEqual(result["engine"], "precomposed")
        self.assertEqual(result["quality_score"], "manual")
        self.assertEqual(result["text"], text)
        generate.assert_not_called()

    def test_precomposed_standalone_rejects_public_handles(self):
        with mock.patch.object(
            bot.BotConfig,
            "from_env",
            return_value=config(dry_run=True, reply_engine="nsrl-live"),
        ):
            result = bot.lambda_handler(
                {
                    "post_tweet": True,
                    "dry_run": True,
                    "id": "bad-checkpoint",
                    "text": "Solomon checkpoint improved for @someone",
                },
                None,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "tweet_generation_error")
        self.assertIn("public handles", result["detail"])

    def test_live_precomposed_standalone_post_is_idempotent(self):
        state = bot.FileStateStore(os.path.join(self.tmp.name, "precomposed.json"))
        client = FakeClient([], [])
        event = {
            "post_tweet": True,
            "dry_run": False,
            "id": "solomon-checkpoint-eval-200",
            "text": "Solomon checkpoint improved:\nEval top1 200/1000. #NSRL",
        }
        with (
            mock.patch.object(
                bot.BotConfig,
                "from_env",
                return_value=config(dry_run=False, reply_engine="nsrl-live"),
            ),
            mock.patch.object(bot, "make_state_store", return_value=state),
            mock.patch.object(bot, "load_secret", return_value={
                "consumer_key": "ck",
                "consumer_secret": "cs",
                "access_token": "at",
                "access_token_secret": "ats",
            }),
            mock.patch.object(bot, "XOAuth1Client", return_value=client),
            mock.patch.object(bot, "generate_nsrl_reply_body") as generate,
        ):
            first = bot.lambda_handler(event, None)
            second = bot.lambda_handler(event, None)
        self.assertTrue(first["posted"])
        self.assertEqual(first["tweet"]["engine"], "precomposed")
        self.assertEqual(client.tweets[0]["text"], event["text"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(client.tweets), 1)
        generate.assert_not_called()

    def test_live_standalone_post_is_idempotent(self):
        state = bot.FileStateStore(os.path.join(self.tmp.name, "standalone.json"))
        client = FakeClient([], [])
        generated = "morning light opens softly and the window finally answers"
        event = {
            "post_tweet": True,
            "dry_run": False,
            "prompt": "publish this",
            "id": "launch-1",
            "candidate_count": 1,
        }
        with (
            mock.patch.object(
                bot.BotConfig,
                "from_env",
                return_value=config(dry_run=False, reply_engine="nsrl-live"),
            ),
            mock.patch.object(bot, "make_state_store", return_value=state),
            mock.patch.object(bot, "load_secret", return_value={
                "consumer_key": "ck",
                "consumer_secret": "cs",
                "access_token": "at",
                "access_token_secret": "ats",
            }),
            mock.patch.object(bot, "XOAuth1Client", return_value=client),
            mock.patch.object(
                bot,
                "generate_nsrl_reply_body",
                return_value=(generated, "base"),
            ),
        ):
            first = bot.lambda_handler(event, None)
            second = bot.lambda_handler(event, None)
        self.assertTrue(first["posted"])
        self.assertEqual(first["post_id"], "launch-1")
        self.assertEqual(len(client.tweets), 1)
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["status"], "posted")
        self.assertEqual(len(client.tweets), 1)

    def test_candidate_count_is_capped(self):
        generated = "morning light opens softly and the window finally answers"
        with (
            mock.patch.object(
                bot.BotConfig,
                "from_env",
                return_value=config(dry_run=True, reply_engine="nsrl-live"),
            ),
            mock.patch.object(
                bot,
                "generate_nsrl_reply_body",
                return_value=(generated, "base"),
            ) as generate,
        ):
            result = bot.lambda_handler(
                {
                    "post_tweet": True,
                    "dry_run": True,
                    "prompt": "publish this",
                    "id": "cap-test",
                    "candidate_count": 999,
                },
                None,
            )
        self.assertTrue(result["would_post"])
        self.assertEqual(result["candidate_count"], str(bot.MAX_STANDALONE_CANDIDATES))
        self.assertEqual(generate.call_count, bot.MAX_STANDALONE_CANDIDATES)

    def test_generation_failure_marks_mention_and_continues(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [
                {"id": "100", "author_id": "7", "text": "@CrowleyBard break"},
                {"id": "101", "author_id": "8", "text": "@CrowleyBard continue"},
            ],
            [
                {"id": "7", "username": "first_user"},
                {"id": "8", "username": "second_user"},
            ],
        )
        with mock.patch.object(
            bot,
            "make_reply",
            side_effect=[
                bot.ReplyGenerationError("empty model output"),
                {"engine": "template", "text": "@second_user continuing"},
            ],
        ):
            result = bot.process_mentions(
                client=client,
                store=self.state,
                config=config(dry_run=False),
                now=self.now,
            )
        self.assertEqual(result["replied"], 1)
        self.assertEqual(result["posted"][0]["id"], "101")
        self.assertEqual(self.state.get_last_seen_id(), "101")
        self.assertTrue(self.state.has_recent_generation_failure("100", self.now))

    def test_live_mode_posts_and_marks_reply_once(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard reply in thunder"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        result = bot.process_mentions(
            client=client, store=self.state, config=config(dry_run=False), now=self.now
        )
        self.assertEqual(result["replied"], 1)
        self.assertEqual(client.replies[0]["in_reply_to_tweet_id"], "100")
        self.assertTrue(self.state.has_replied("100"))
        self.assertEqual(self.state.get_last_seen_id(), "100")

        result_again = bot.process_mentions(
            client=client, store=self.state, config=config(dry_run=False), now=self.now
        )
        self.assertEqual(result_again["replied"], 0)

    def test_binary_pgm_converts_to_png(self):
        png, width, height = bot.pgm_bytes_to_png(b"P5\n2 2\n255\n\x00\x7f\x80\xff")
        self.assertEqual((width, height), (2, 2))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", png)
        self.assertIn(b"IDAT", png)

    def test_dry_run_reply_includes_sigil_metadata_when_enabled(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard what is love?"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        sigil = {"seed": "abc", "target_name": "Asmoday", "png_bytes": b"png"}
        with mock.patch.object(bot, "generate_solomon_sigil", return_value=sigil):
            result = bot.process_mentions(
                client=client,
                store=self.state,
                config=config(sigil_enabled=True),
                now=self.now,
            )
        self.assertEqual(result["replied"], 1)
        self.assertEqual(result["would_reply"][0]["sigil"]["seed"], "abc")
        self.assertNotIn("png_bytes", result["would_reply"][0]["sigil"])

    def test_live_reply_uploads_and_attaches_sigil(self):
        self.state.set_last_seen_id("99", self.now)
        client = FakeClient(
            [{"id": "100", "author_id": "7", "text": "@CrowleyBard reply in thunder"}],
            [{"id": "7", "username": "signal_summoner"}],
        )
        sigil = {"seed": "abc", "target_name": "Asmoday", "png_bytes": b"png"}
        with mock.patch.object(bot, "generate_solomon_sigil", return_value=sigil):
            result = bot.process_mentions(
                client=client,
                store=self.state,
                config=config(dry_run=False, sigil_enabled=True),
                now=self.now,
            )
        self.assertEqual(result["replied"], 1)
        self.assertEqual(client.media_uploads[0]["bytes"], b"png")
        self.assertEqual(client.replies[0]["media_ids"], ["media-1"])
        self.assertEqual(result["posted"][0]["sigil"]["media_id"], "media-1")


if __name__ == "__main__":
    unittest.main()
