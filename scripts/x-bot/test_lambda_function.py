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
        self.replies.append({"text": text, "in_reply_to_tweet_id": in_reply_to_tweet_id})
        return {"data": {"id": f"reply-{in_reply_to_tweet_id}", "text": text}}

    def post_tweet(self, text):
        tweet_id = f"tweet-{len(self.tweets) + 1}"
        self.tweets.append({"id": tweet_id, "text": text})
        return {"data": {"id": tweet_id, "text": text}}


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
                {"post_tweet": False, "dry_run": False, "prompt": "do not publish"},
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


if __name__ == "__main__":
    unittest.main()
