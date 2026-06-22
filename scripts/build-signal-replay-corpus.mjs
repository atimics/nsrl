#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = {
  outDir: "data/processed/signal-replay-corpus",
  signalRoot: "/Users/ratimics/develop/signal",
  buildDir: "/Users/ratimics/develop/signal/build",
  seed: "signal-replay-v1",
  replaySeeds: "2037,2038,2047",
  scripts: "none,mine-fracture,asteroid-death,planned-outpost,station-jostle,player-ram,npc-ram,thrown-rock-hit,fracture-claim",
  horizons: "12,36",
  candidates: "NONE,W,A,D,S,WA,WD,SA,SD",
  skipFailed: "true",
  buildReplay: "true",
};

const stationNames = new Map([
  [0, "Prospect Ref"],
  [1, "Kepler Yard"],
  [2, "Helios Works"],
  [3, "Freeport"],
]);

const scriptLabels = new Map([
  ["none", "open flight"],
  ["mine-fracture", "fracture mining"],
  ["asteroid-death", "spent rock"],
  ["planned-outpost", "planned outpost"],
  ["station-jostle", "station jostle"],
  ["player-ram", "ram warning"],
  ["npc-ram", "traffic ram"],
  ["thrown-rock-hit", "thrown rock"],
  ["fracture-claim", "fracture claim"],
  ["buy-sell", "dock trade"],
  ["pod-tow-sell", "pod tow"],
]);

function usage() {
  console.log(`Usage: node scripts/build-signal-replay-corpus.mjs [options]

Options:
  --out-dir PATH        Output directory [${defaults.outDir}]
  --signal-root PATH    Signal checkout root [${defaults.signalRoot}]
  --build-dir PATH      Signal CMake build directory [${defaults.buildDir}]
  --seed TEXT           Stable id seed [${defaults.seed}]
  --replay-seeds LIST   Comma-separated replay seeds [${defaults.replaySeeds}]
  --scripts LIST        Comma-separated replay scripts [${defaults.scripts}]
  --horizons LIST       Comma-separated horizon ticks [${defaults.horizons}]
  --candidates LIST     Comma-separated replay candidates [${defaults.candidates}]
  --skip-failed BOOL    Skip replay setups that fail [${defaults.skipFailed}]
  --build-replay BOOL   Build signal_replay target first [${defaults.buildReplay}]
`);
}

function parseArgs(argv) {
  const options = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in options)) {
      throw new Error(`unknown option: ${arg}`);
    }
    const value = argv[++index];
    if (value === undefined) {
      throw new Error(`${arg} requires a value`);
    }
    options[key] = value;
  }
  return options;
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function splitList(value) {
  return String(value || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function parseBool(value, optionName) {
  const text = String(value).toLowerCase();
  if (["1", "true", "yes", "on"].includes(text)) return true;
  if (["0", "false", "no", "off"].includes(text)) return false;
  throw new Error(`${optionName} must be true or false`);
}

function parsePositiveInts(value, optionName) {
  return splitList(value).map((part) => {
    const parsed = Number.parseInt(part, 10);
    if (!Number.isFinite(parsed) || parsed < 1) {
      throw new Error(`${optionName} must contain positive integers`);
    }
    return parsed;
  });
}

function cleanAscii(text) {
  return String(text ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    encoding: "utf8",
    stdio: options.capture ? ["ignore", "pipe", "pipe"] : ["ignore", "inherit", "pipe"],
  });
  if (result.status !== 0 && !options.allowFailure) {
    const stderr = result.stderr ? `\n${result.stderr.trim()}` : "";
    throw new Error(`${command} failed with status ${result.status}${stderr}`);
  }
  return result;
}

function parseJsonl(text, label) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${label}:${index + 1}: ${error.message}`);
      }
    });
}

function stationName(index) {
  return stationNames.get(Number(index)) || `Station ${index}`;
}

function scriptLabel(name) {
  return scriptLabels.get(name) || name.replace(/-/g, " ");
}

function fixed(value, digits = 1) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
}

function signedRound(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "0";
  const rounded = Math.round(number);
  return rounded > 0 ? `+${rounded}` : String(rounded);
}

function privateState(row) {
  const station = stationName(row.station);
  const label = scriptLabel(row.provenance_script);
  const candidate = row.candidate_name;
  const events = [];
  if (row.damage_events) events.push(`${row.damage_events} damage`);
  if (row.death_events) events.push(`${row.death_events} death`);
  if (row.dock_events) events.push(`${row.dock_events} dock`);
  if (row.launch_events) events.push(`${row.launch_events} launch`);
  if (row.pickup_events) events.push(`${row.pickup_events} pickup`);
  if (row.buy_events) events.push(`${row.buy_events} buy`);
  if (row.sell_events) events.push(`${row.sell_events} sell`);
  if (row.fracture_events) events.push(`${row.fracture_events} fracture`);
  if (row.outpost_placed_events) events.push(`${row.outpost_placed_events} outpost`);
  if (row.scaffold_ready_events) events.push(`${row.scaffold_ready_events} scaffold`);
  return cleanAscii([
    `Pilot N${Number(row.seed) % 97} is near ${station} in ${label}.`,
    `Control vector ${candidate} holds for ${row.horizon_ticks} ticks after ${row.prefix_ticks} prefix ticks.`,
    `Distance changes from ${fixed(row.start_dist)} to ${fixed(row.end_dist)} with progress ${signedRound(row.progress)}.`,
    `Hull moves from ${fixed(row.start_hull)} to ${fixed(row.end_hull)}; cargo moves from ${fixed(row.start_cargo)} to ${fixed(row.end_cargo)}.`,
    `End speed is ${fixed(row.end_speed)} and docked is ${row.end_docked ? "yes" : "no"}.`,
    `Events: ${events.length ? events.join(", ") : "quiet flight"}.`,
    `Nearby traffic needs a clipped radio call from this moment.`,
  ].join(" "));
}

function expectedOutput(row) {
  const station = stationName(row.station);
  const candidate = row.candidate_name;
  const label = scriptLabel(row.provenance_script);
  const progress = Number(row.progress || 0);
  const endSpeed = Number(row.end_speed || 0);

  if (Number(row.death_events || 0) > 0) {
    return `Mayday on ${label}; hull lost, clear the lane.`;
  }
  if (Number(row.damage_events || 0) > 0 || Number(row.damage_amount || 0) > 0) {
    return `Hull hit on ${label}; widening spacing.`;
  }
  if (Number(row.dock_events || 0) > 0) {
    return `Dock mark closed at ${station}.`;
  }
  if (Number(row.launch_events || 0) > 0) {
    return `Launch clear from ${station}; nose steady.`;
  }
  if (Number(row.sell_events || 0) > 0) {
    const credits = Number(row.sell_base || 0) + Number(row.sell_bonus || 0);
    return `${credits} credits landed at ${station}.`;
  }
  if (Number(row.buy_events || 0) > 0) {
    return `${row.buy_quantity || 0} units bought at ${station}; outbound.`;
  }
  if (Number(row.pickup_events || 0) > 0 || Number(row.pickup_fragments || 0) > 0) {
    return `${row.pickup_fragments || row.pickup_events} fragments aboard; ore reads ${fixed(row.pickup_ore)}.`;
  }
  if (Number(row.fracture_events || 0) > 0) {
    return `Fracture mark open near ${station}; hold ${candidate} vector.`;
  }
  if (Number(row.outpost_placed_events || 0) > 0) {
    return `Outpost mark placed; ${station} lane grows wider.`;
  }
  if (Number(row.scaffold_ready_events || 0) > 0) {
    return `Scaffold ready near ${station}; haulers can commit.`;
  }
  if (progress > 0.5) {
    return `${candidate} vector gains ${Math.round(progress)} toward ${station}.`;
  }
  if (progress < -0.5) {
    return `${candidate} vector loses ${Math.abs(Math.round(progress))}; correcting.`;
  }
  if (endSpeed > 20) {
    return `${candidate} vector fast at ${Math.round(endSpeed)}; hull clean.`;
  }
  return `Holding near ${station}; controls quiet.`;
}

function makeFrame(options, row) {
  const output = cleanAscii(expectedOutput(row));
  const state = privateState(row);
  const id = idFor([
    options.seed,
    row.seed,
    row.provenance_script,
    row.horizon_ticks,
    row.candidate,
    row.prefix_state_hash,
    row.state_hash,
  ]);
  return {
    id,
    source: "signal_replay",
    source_id: `${row.seed}:${row.provenance_script}:h${row.horizon_ticks}:c${row.candidate}`,
    domain: "signal",
    schema: "nsrl.signal_replay_frame.v1",
    speaker: `PILOT N${Number(row.seed) % 97}`,
    kind: "replay_branch",
    private_state: state,
    expected_output: output,
    line: output,
    target: output,
    prompt: `STATE: ${state}\nRADIO: `,
    fields: {
      station: stationName(row.station),
      provenance_script: row.provenance_script,
      candidate: row.candidate_name,
      horizon_ticks: String(row.horizon_ticks),
      progress: String(row.progress),
      utility: String(row.utility),
      end_speed: String(row.end_speed),
      end_hull: String(row.end_hull),
      end_cargo: String(row.end_cargo),
      event_hash: row.event_hash,
    },
    grounding_terms: [
      stationName(row.station),
      row.provenance_script,
      row.candidate_name,
      scriptLabel(row.provenance_script),
    ],
  };
}

function buildReplayRows(options, replayPath, failures) {
  const replaySeeds = parsePositiveInts(options.replaySeeds, "--replay-seeds");
  const scripts = splitList(options.scripts);
  const horizons = parsePositiveInts(options.horizons, "--horizons");
  const candidates = splitList(options.candidates).join(",");
  const skipFailed = parseBool(options.skipFailed, "--skip-failed");
  const rows = [];

  for (const replaySeed of replaySeeds) {
    for (const script of scripts) {
      for (const horizon of horizons) {
        const result = run(replayPath, [
          "--seed", String(replaySeed),
          "--provenance-script", script,
          "--horizon-ticks", String(horizon),
          "--candidates", candidates,
        ], { capture: true, allowFailure: true });
        if (result.status !== 0) {
          const failure = {
            seed: replaySeed,
            script,
            horizon,
            status: result.status,
            stderr: cleanAscii(result.stderr),
          };
          failures.push(failure);
          if (!skipFailed) {
            throw new Error(`signal_replay failed for ${JSON.stringify(failure)}`);
          }
          continue;
        }
        rows.push(...parseJsonl(result.stdout, `${replayPath}:${replaySeed}:${script}:${horizon}`));
      }
    }
  }
  return rows;
}

function writeJsonl(filePath, records) {
  fs.writeFileSync(filePath, records.map((record) => `${JSON.stringify(record)}\n`).join(""), "utf8");
}

function trainingText(frame) {
  return `${frame.private_state}\n${frame.expected_output}\nEND\n`;
}

function trainingPair(frame) {
  return {
    id: frame.id,
    speaker: frame.speaker,
    kind: frame.kind,
    domain: frame.domain,
    private_state: frame.private_state,
    expected_output: frame.expected_output,
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const replayPath = path.join(options.buildDir, "signal_replay");
  if (parseBool(options.buildReplay, "--build-replay") || !fs.existsSync(replayPath)) {
    run("cmake", ["--build", options.buildDir, "--target", "signal_replay", "--parallel", "4"], {
      cwd: options.signalRoot,
    });
  }
  if (!fs.existsSync(replayPath)) {
    throw new Error(`missing signal_replay binary: ${replayPath}`);
  }

  const failures = [];
  const states = buildReplayRows(options, replayPath, failures);
  const frames = states.map((row) => makeFrame(options, row));

  const statesPath = path.join(outDir, "states.jsonl");
  const framesPath = path.join(outDir, "frames.jsonl");
  const trainingPairsPath = path.join(outDir, "training-pairs.jsonl");
  const corpusPath = path.join(outDir, "corpus.txt");
  const voicePath = path.join(outDir, "voice.txt");
  const manifestPath = path.join(outDir, "manifest.json");

  writeJsonl(statesPath, states);
  writeJsonl(framesPath, frames);
  writeJsonl(trainingPairsPath, frames.map(trainingPair));
  fs.writeFileSync(corpusPath, `SIGNAL_REPLAY_CORPUS_V1\n\n${frames.map(trainingText).join("\n")}`, "utf8");
  fs.writeFileSync(voicePath, `${frames.map((frame) => frame.expected_output).join("\n")}\n`, "utf8");
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.signal_replay_corpus.v1",
    created_at: new Date().toISOString(),
    signal_root: options.signalRoot,
    build_dir: options.buildDir,
    replay_binary_path: replayPath,
    states_path: statesPath,
    frames_path: framesPath,
    training_pairs_path: trainingPairsPath,
    corpus_path: corpusPath,
    voice_path: voicePath,
    replay_seeds: parsePositiveInts(options.replaySeeds, "--replay-seeds"),
    scripts: splitList(options.scripts),
    horizons: parsePositiveInts(options.horizons, "--horizons"),
    candidates: splitList(options.candidates),
    state_rows: states.length,
    frames: frames.length,
    failures,
    notes: [
      "Generated by running Signal's C signal_replay deterministic branch tool.",
      "states.jsonl preserves raw replay rows; training-pairs.jsonl maps each branch to private_state and a clipped pilot radio line.",
      "Some provenance scripts can fail for a given seed/default setup; skipped failures are recorded in this manifest.",
    ],
  }, null, 2)}\n`, "utf8");

  console.log(`states=${states.length}`);
  console.log(`frames=${frames.length}`);
  console.log(`failures=${failures.length}`);
  console.log(`states_path=${statesPath}`);
  console.log(`frames_path=${framesPath}`);
  console.log(`training_pairs=${trainingPairsPath}`);
  console.log(`manifest=${manifestPath}`);
}

try {
  main();
} catch (error) {
  console.error(`build-signal-replay-corpus: ${error.message}`);
  process.exit(1);
}
