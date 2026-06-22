#!/usr/bin/env node
import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  runDir: "data/processed/signal-romance-smoke",
  backend: "mini-transformer",
  model: "",
  tokens: "",
  vocab: "",
  prompts: "",
  outDir: "",
  count: 20,
  maxNewTokens: 96,
  topK: 8,
  sampleSeed: 7,
  groundingMinRate: 0.8,
  printableMinRate: 0.95,
  failOnThreshold: false,
};

function usage() {
  console.log(`Usage: node scripts/eval-signal-romance.mjs [options]

Options:
  --run-dir PATH              Run directory [${defaults.runDir}]
  --backend NAME              mini-transformer or lexeme [${defaults.backend}]
  --model PATH                Mini-transformer model [run-dir/signal-romance.nsrlmt]
  --tokens PATH               Byte tokens [run-dir/corpus.tokens.u8]
  --vocab PATH                Lexeme vocabulary for --backend lexeme
  --prompts PATH              Eval prompts JSONL [run-dir/eval-prompts.jsonl]
  --out-dir PATH              Eval output directory [run-dir/eval]
  --count N                   Prompt count [${defaults.count}]
  --max-new-tokens N          Generation length [${defaults.maxNewTokens}]
  --top-k N                   Sample top-k [${defaults.topK}]
  --sample-seed N             First sample seed [${defaults.sampleSeed}]
  --grounding-min-rate N      Grounding pass threshold [${defaults.groundingMinRate}]
  --printable-min-rate N      Printable pass threshold [${defaults.printableMinRate}]
  --fail-on-threshold         Exit non-zero when thresholds fail
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
    if (arg === "--fail-on-threshold") {
      options.failOnThreshold = true;
      continue;
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
    if (["count", "maxNewTokens", "topK", "sampleSeed"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
    } else if (["groundingMinRate", "printableMinRate"].includes(key)) {
      options[key] = Number.parseFloat(value);
    } else {
      options[key] = value;
    }
  }
  return options;
}

function resolveRepoPath(filePath) {
  if (path.isAbsolute(filePath)) {
    return filePath;
  }
  return path.join(repoRoot, filePath);
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, "utf8")
    .split(/\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${filePath}:${index + 1}: ${error.message}`);
      }
    });
}

function trainInvocation(args) {
  if (process.env.NSRL_TRAIN_BIN) {
    return { command: process.env.NSRL_TRAIN_BIN, args };
  }
  return {
    command: "cargo",
    args: ["run", "--release", "-q", "-p", "nsrl-train", "--", ...args],
  };
}

function runGeneration({
  backend,
  model,
  tokens,
  vocab,
  prompt,
  textOut,
  traceOut,
  maxNewTokens,
  topK,
  sampleSeed,
}) {
  if (backend === "lexeme") {
    const args = [
      "--mode", "lexeme-generate",
      "--model", model,
      "--vocab", vocab,
      "--prompt", prompt,
      "--max-new-tokens", String(maxNewTokens),
      "--decode", "greedy",
      "--stop-on-sentence-terminal",
      "--generated-only",
      "--text-out", textOut,
      "--trace", traceOut,
    ];
    const invocation = trainInvocation(args);
    const result = childProcess.spawnSync(invocation.command, invocation.args, {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.status !== 0) {
      throw new Error(`generation failed for ${textOut}\n${result.stdout}\n${result.stderr}`);
    }
    return;
  }

  const args = [
    "--mode", "mini-transformer-generate",
    "--model", model,
    "--tokens", tokens,
    "--prompt", prompt,
    "--max-new-tokens", String(maxNewTokens),
    "--decode", "sample",
    "--top-k", String(topK),
    "--sample-seed", String(sampleSeed),
    "--mini-transformer-attention", "linear",
    "--mini-transformer-position", "nope",
    "--printable-only",
    "--repeat-window", "96",
    "--repeat-penalty-shift", "3",
    "--max-repeat-run", "3",
    "--no-repeat-ngram", "4",
    "--corpus-prior",
    "--corpus-prior-logit-shift", "4",
    "--generated-only",
    "--text-out", textOut,
    "--trace", traceOut,
  ];
  const invocation = trainInvocation(args);
  const result = childProcess.spawnSync(invocation.command, invocation.args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.status !== 0) {
    throw new Error(`generation failed for ${textOut}\n${result.stdout}\n${result.stderr}`);
  }
}

function firstGeneratedLine(text) {
  let cleaned = text
    .split(/\bEND\b/i)[0]
    .split(/\bSPEAKER\b/i)[0]
    .split(/\bRANKED:/i)[0]
    .split(/\bVOICE:/i)[0]
    .split(/\r?\n/)[0]
    .replace(/<\|[^|]+\|>/g, "")
    .replace(/\s+/g, " ")
    .replace(/^[^A-Za-z0-9]+/, "")
    .trim();
  const sentence = cleaned.match(/^.*?[.!?](?:\s|$)/);
  if (sentence) {
    cleaned = sentence[0].trim();
  }
  return cleaned;
}

function isPrintable(text) {
  return /^[\x09\x0a\x0d\x20-\x7e]*$/.test(text);
}

function words(text) {
  return text.split(/\s+/).filter(Boolean);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function containsTerm(text, term) {
  if (!term) {
    return false;
  }
  return new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(term)}([^A-Za-z0-9]|$)`, "i").test(text);
}

function validateLine(line, frame, manifest) {
  const failures = [];
  if (!line) {
    failures.push("empty");
  }
  if (!isPrintable(line)) {
    failures.push("non_printable");
  }
  if (line.length > 96 || words(line).length > 16) {
    failures.push("too_long");
  }
  const forbidden = /\b(assistant|system|user|language model|roleplay|hello|greetings?|i can|i am)\b/i;
  if (forbidden.test(line)) {
    failures.push("assistant_voice");
  }
  const scaffold = /\b(ranked|voice|style|end_style|signal_doctrine|signal_truth_pass)\b/i;
  if (scaffold.test(line)) {
    failures.push("corpus_scaffold");
  }
  const sourceLeak = /\b(shakespeare|blake|crowley|simplewiki|apollo|nasa)\b/i;
  if (sourceLeak.test(line)) {
    failures.push("source_label");
  }
  const grounded = (frame.grounding_terms ?? []).some((term) => containsTerm(line, term));
  if (!grounded) {
    failures.push("ungrounded");
  }
  for (const station of manifest.known_stations ?? []) {
    if (containsTerm(line, station) && !(frame.grounding_terms ?? []).some((term) => term.includes(station))) {
      failures.push(`invented_station:${station}`);
    }
  }
  for (const commodity of manifest.known_commodities ?? []) {
    if (containsTerm(line, commodity) && !(frame.grounding_terms ?? []).includes(commodity)) {
      failures.push(`invented_commodity:${commodity}`);
    }
  }
  return {
    ok: failures.length === 0,
    printable: isPrintable(line),
    grounded,
    bounded: line.length <= 96 && words(line).length <= 16,
    failures,
  };
}

function rate(count, total) {
  return total === 0 ? 0 : count / total;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const runDir = resolveRepoPath(options.runDir);
  if (!["mini-transformer", "lexeme"].includes(options.backend)) {
    throw new Error("--backend requires mini-transformer or lexeme");
  }
  const defaultModel = options.backend === "lexeme"
    ? path.join(runDir, "v384-seq16.nsrllm")
    : path.join(runDir, "signal-romance.nsrlmt");
  const model = resolveRepoPath(options.model || defaultModel);
  const tokens = options.tokens ? resolveRepoPath(options.tokens) : resolveRepoPath(path.join(runDir, "corpus.tokens.u8"));
  const vocab = options.vocab ? resolveRepoPath(options.vocab) : resolveRepoPath(path.join(runDir, "v1024.vocab.tsv"));
  const promptsPath = resolveRepoPath(options.prompts || path.join(runDir, "eval-prompts.jsonl"));
  const outDir = resolveRepoPath(options.outDir || path.join(runDir, "eval"));
  const manifestPath = path.join(runDir, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

  const requiredPaths = options.backend === "lexeme" ? [model, vocab, promptsPath] : [model, tokens, promptsPath];
  for (const required of requiredPaths) {
    if (!fs.existsSync(required)) {
      throw new Error(`missing required artifact: ${required}`);
    }
  }

  fs.mkdirSync(outDir, { recursive: true });
  const generationsDir = path.join(outDir, "generations");
  fs.mkdirSync(generationsDir, { recursive: true });

  const prompts = readJsonl(promptsPath).slice(0, options.count);
  const results = [];
  for (const [index, frame] of prompts.entries()) {
    const safeId = `${String(index).padStart(2, "0")}-${frame.id}`;
    const textOut = path.join(generationsDir, `${safeId}.txt`);
    const traceOut = path.join(generationsDir, `${safeId}.trace.jsonl`);
    const sampleSeed = options.sampleSeed + index;
    runGeneration({
      backend: options.backend,
      model,
      tokens,
      vocab,
      prompt: frame.prompt,
      textOut,
      traceOut,
      maxNewTokens: options.maxNewTokens,
      topK: options.topK,
      sampleSeed,
    });
    const raw = fs.readFileSync(textOut, "utf8");
    const line = firstGeneratedLine(raw);
    const validation = validateLine(line, frame, manifest);
    const usedFallback = !validation.ok;
    const finalLine = usedFallback ? frame.target : line;
    const finalValidation = validateLine(finalLine, frame, manifest);
    results.push({
      id: frame.id,
      speaker: frame.speaker,
      role: frame.role,
      prompt: frame.prompt,
      target: frame.target,
      generated_raw: raw,
      generated_line: line,
      final_line: finalLine,
      used_fallback: usedFallback,
      sample_seed: sampleSeed,
      text_out: textOut,
      trace_out: traceOut,
      raw_validation: validation,
      final_validation: finalValidation,
      ok: validation.ok,
      printable: validation.printable,
      grounded: validation.grounded,
      bounded: validation.bounded,
      failures: validation.failures,
      final_ok: finalValidation.ok,
      final_printable: finalValidation.printable,
      final_grounded: finalValidation.grounded,
      final_bounded: finalValidation.bounded,
      final_failures: finalValidation.failures,
    });
    const status = validation.ok ? "ok" : `fallback:${validation.failures.join(",")}`;
    console.log(`${safeId}\t${status}\traw=${line}\tfinal=${finalLine}`);
  }

  const total = results.length;
  const summary = {
    schema: "nsrl.signal_romance_eval.v1",
    created_at: new Date().toISOString(),
    run_dir: runDir,
    model,
    tokens,
    vocab: options.backend === "lexeme" ? vocab : undefined,
    backend: options.backend,
    prompts: promptsPath,
    total,
    raw_ok: results.filter((result) => result.ok).length,
    raw_printable: results.filter((result) => result.printable).length,
    raw_grounded: results.filter((result) => result.grounded).length,
    raw_bounded: results.filter((result) => result.bounded).length,
    raw_printable_rate: rate(results.filter((result) => result.printable).length, total),
    raw_grounded_rate: rate(results.filter((result) => result.grounded).length, total),
    raw_ok_rate: rate(results.filter((result) => result.ok).length, total),
    fallback_count: results.filter((result) => result.used_fallback).length,
    final_ok: results.filter((result) => result.final_ok).length,
    final_printable: results.filter((result) => result.final_printable).length,
    final_grounded: results.filter((result) => result.final_grounded).length,
    final_bounded: results.filter((result) => result.final_bounded).length,
    final_printable_rate: rate(results.filter((result) => result.final_printable).length, total),
    final_grounded_rate: rate(results.filter((result) => result.final_grounded).length, total),
    final_ok_rate: rate(results.filter((result) => result.final_ok).length, total),
    thresholds: {
      printable_min_rate: options.printableMinRate,
      grounding_min_rate: options.groundingMinRate,
    },
  };
  summary.raw_threshold_pass =
    summary.raw_printable_rate >= options.printableMinRate &&
    summary.raw_grounded_rate >= options.groundingMinRate;
  summary.threshold_pass =
    summary.final_printable_rate >= options.printableMinRate &&
    summary.final_grounded_rate >= options.groundingMinRate;

  const resultsPath = path.join(outDir, "eval-results.jsonl");
  fs.writeFileSync(resultsPath, results.map((result) => `${JSON.stringify(result)}\n`).join(""), "utf8");
  fs.writeFileSync(path.join(outDir, "eval-report.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  fs.writeFileSync(
    path.join(outDir, "eval-summary.tsv"),
    [
      "total\traw_ok\traw_printable\traw_grounded\traw_bounded\tfallback_count\tfinal_ok\tfinal_printable\tfinal_grounded\tfinal_bounded\traw_ok_rate\traw_printable_rate\traw_grounded_rate\tfinal_ok_rate\tfinal_printable_rate\tfinal_grounded_rate\traw_threshold_pass\tthreshold_pass",
      `${summary.total}\t${summary.raw_ok}\t${summary.raw_printable}\t${summary.raw_grounded}\t${summary.raw_bounded}\t${summary.fallback_count}\t${summary.final_ok}\t${summary.final_printable}\t${summary.final_grounded}\t${summary.final_bounded}\t${summary.raw_ok_rate.toFixed(3)}\t${summary.raw_printable_rate.toFixed(3)}\t${summary.raw_grounded_rate.toFixed(3)}\t${summary.final_ok_rate.toFixed(3)}\t${summary.final_printable_rate.toFixed(3)}\t${summary.final_grounded_rate.toFixed(3)}\t${summary.raw_threshold_pass}\t${summary.threshold_pass}`,
      "",
    ].join("\n"),
    "utf8"
  );
  console.log(`report=${path.join(outDir, "eval-report.json")}`);
  console.log(`summary=${JSON.stringify(summary)}`);

  if (options.failOnThreshold && !summary.threshold_pass) {
    process.exit(2);
  }
}

try {
  main();
} catch (error) {
  console.error(`eval-signal-romance: ${error.message}`);
  process.exit(1);
}
