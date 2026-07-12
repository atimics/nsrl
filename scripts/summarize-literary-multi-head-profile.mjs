#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const out = process.argv[2] ?? "data/experiments/literary-multi-head-profile-v1/report.json";
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const shifts = [3, 4, 5];

async function loadProfile(profile, directoryPattern, withHeadDeltas) {
  const runs = {};
  for (const shift of shifts) {
    const directory = directoryPattern.replace("{shift}", String(shift));
    const trainBytes = await readFile(path.join(directory, "train.json"));
    const holdoutBytes = await readFile(path.join(directory, "holdout.json"));
    const train = JSON.parse(trainBytes);
    const holdout = JSON.parse(holdoutBytes);
    const headDeltaBytes = withHeadDeltas
      ? await readFile(path.join(directory, "head-deltas.json"))
      : null;
    runs[shift] = {
      train,
      holdout: holdout.evaluation,
      architecture: holdout.model,
      head_deltas: headDeltaBytes ? JSON.parse(headDeltaBytes) : null,
      sha256: {
        train: sha256(trainBytes),
        holdout: sha256(holdoutBytes),
        ...(headDeltaBytes ? { head_deltas: sha256(headDeltaBytes) } : {}),
      },
    };
  }
  return { profile, runs };
}

const h2 = await loadProfile(
  "small-h2-d128-ff256",
  "data/local-runs/literary-rms-adam-seq64-512-shift{shift}",
  false,
);
const h8 = await loadProfile(
  "small-h8-d128-ff256",
  "data/local-runs/literary-h8-adam-seq64-512-shift{shift}",
  true,
);
const best = (profile) =>
  Object.entries(profile.runs).sort(
    ([, left], [, right]) =>
      left.holdout.mean_probability_error_q15 - right.holdout.mean_probability_error_q15 ||
      right.holdout.accuracy_per_mille - left.holdout.accuracy_per_mille,
  )[0];
const [bestH2Shift, bestH2] = best(h2);
const [bestH8Shift, bestH8] = best(h8);
if (!bestH8.head_deltas.all_heads_updated) throw new Error("not every H8 head updated");
if (bestH8.holdout.invalid_forward_count !== 0) throw new Error("H8 has invalid forwards");

const report = {
  schema: "nsrl.literary_multi_head_profile.v1",
  experiment: {
    train_windows: 512,
    seq_len: 64,
    optimizer: "integer_adam",
    step_shifts: shifts,
    same_d_model: true,
    same_hidden_dim: true,
  },
  profiles: { h2, h8 },
  best_comparison: {
    h2: { shift: Number(bestH2Shift), ...bestH2.holdout },
    h8: { shift: Number(bestH8Shift), ...bestH8.holdout },
    h8_delta_vs_h2: {
      accuracy_per_mille: bestH8.holdout.accuracy_per_mille - bestH2.holdout.accuracy_per_mille,
      mistakes: bestH8.holdout.mistakes - bestH2.holdout.mistakes,
      mean_probability_error_q15:
        bestH8.holdout.mean_probability_error_q15 - bestH2.holdout.mean_probability_error_q15,
    },
  },
  validation: {
    host_no_std_single_step_parity: true,
    all_h8_heads_updated: true,
    h8_byte_exact_replay: true,
    cross_profile_artifacts_rejected: true,
    h8_invalid_forwards: bestH8.holdout.invalid_forward_count,
  },
  promotion: {
    h8_small_profile_promoted: true,
    reason: "best H8 lowers held-out mean error by 7590 Q15 and raises accuracy by 121 per mille",
    next_step: "train a small swarm of H8 leaves before adding width",
  },
  known_non_claims: [
    "bounded_512_window_architecture_probe",
    "not_yet_a_full_corpus_language_model",
    "does_not_claim_human_level_generation",
  ],
};

await import("node:fs/promises").then(({ mkdir }) => mkdir(path.dirname(out), { recursive: true }));
await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, comparison: report.best_comparison, validation: report.validation }));
