#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "data/processed/signal-sim-log-corpus",
  repeat: 1,
  seed: "signal-sim-log-v1",
};

const stations = ["Prospect Ref", "Kepler Yard", "Helios Works", "Freeport"];
const commodities = ["FE", "CU", "CR", "FR", "CO", "LN", "FM", "LM", "TM", "RK"];
const modules = ["Frame Press", "Laser Fab", "Tractor Fab", "Signal Relay", "Shipyard"];
const roles = ["miner", "hauler"];
const states = ["prospecting", "mining", "returning", "outbound", "unloading", "holding"];
const memoryKinds = [
  "ore pressure",
  "demand",
  "supply",
  "route risk",
  "route danger",
  "route success",
  "route reputation",
  "delivery receipt",
  "station trust",
  "station risk",
];

function usage() {
  console.log(`Usage: node scripts/build-signal-sim-log-corpus.mjs [options]

Options:
  --out-dir PATH   Output directory [${defaults.outDir}]
  --repeat N       Repeat deterministic sweep with shifted ticks [${defaults.repeat}]
  --seed TEXT      Stable id seed [${defaults.seed}]
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
    if (key === "repeat") {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 1) {
        throw new Error(`${arg} requires a positive integer`);
      }
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

function cleanAscii(text) {
  return String(text ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function routePairs() {
  const pairs = [];
  for (const source of stations) {
    for (const dest of stations) {
      if (source !== dest) {
        pairs.push({ source, dest, route: `${source}>${dest}` });
      }
    }
  }
  return pairs;
}

function lineTemplates(kind, role) {
  if (kind === "ore pressure") {
    return [
      ({ commodity, source }) => `${source}, copy ${commodity} pressure bright.`,
      ({ commodity, source }) => `${commodity} ore mark holds at ${source}, over.`,
      ({ commodity, source }) => `${source} reports ${commodity} pocket steady.`,
      ({ commodity, source, state }) => `${commodity} ${state} crew holding near ${source}.`,
    ];
  }
  if (kind === "supply") {
    return [
      ({ commodity, source }) => `${source} has ${commodity} supply warm, copy.`,
      ({ commodity, source }) => `${commodity} stock moving out of ${source}.`,
      ({ commodity, source }) => `Stand by for ${commodity} supply at ${source}.`,
    ];
  }
  if (kind === "demand") {
    return [
      ({ commodity, source }) => `${source} calling for ${commodity}, readback requested.`,
      ({ commodity, source, quantity }) => `${source} has ${quantity} ${commodity} haul open.`,
      ({ commodity, source }) => `${commodity} demand board awake at ${source}.`,
    ];
  }
  if (kind === "route risk" || kind === "route danger") {
    return [
      ({ commodity, route }) => `Caution ${commodity} traffic on ${route}.`,
      ({ source, dest }) => `${source}>${dest} reads rough, keep spacing.`,
      ({ commodity, dest }) => `Hazard near ${dest} for ${commodity} haul.`,
      ({ commodity, route }) => `Negative clean route for ${commodity} on ${route}.`,
    ];
  }
  if (kind === "route success" || kind === "route reputation") {
    return [
      ({ commodity, route }) => `${commodity} lane ${route} runs clean, copy.`,
      ({ source, dest }) => `${source}>${dest} pays clean, proceed.`,
      ({ commodity, route }) => `${commodity} route ${route} confirmed trusted.`,
      ({ commodity, dest }) => `${commodity} clean arrival mark at ${dest}.`,
    ];
  }
  if (kind === "delivery receipt") {
    return [
      ({ commodity, route, quantity }) => `Readback: ${quantity} ${commodity} landed ${route}.`,
      ({ commodity, route }) => `${commodity} receipt closed on ${route}.`,
      ({ commodity, dest }) => `${commodity} delivery confirmed at ${dest}.`,
    ];
  }
  if (kind === "station trust") {
    return [
      ({ commodity, source }) => `${commodity} desk at ${source} pays clean, copy.`,
      ({ commodity, source }) => `${source} clears ${commodity} work today.`,
      ({ commodity, source }) => `${source} ${commodity} mark confirmed clean.`,
    ];
  }
  if (kind === "station risk") {
    return [
      ({ commodity, source }) => `${commodity} desk at ${source} reads sharp.`,
      ({ commodity, source }) => `${source} ${commodity} work needs escort.`,
      ({ commodity, source }) => `Risk mark posted for ${commodity} at ${source}.`,
    ];
  }
  return role === "miner"
    ? [({ source }) => `Miner channel holding at ${source}.`]
    : [({ route }) => `Hauler channel holding on ${route}.`];
}

function moduleTemplates() {
  return [
    ({ module, dest }) => `${dest} requests ${module} scaffold check.`,
    ({ module, source, dest }) => `${module} kit moving ${source}>${dest}.`,
    ({ module, dest }) => `${module} build signal holding at ${dest}.`,
    ({ module, source, dest }) => `Readback ${module} shell for ${source}>${dest}.`,
  ];
}

function pilotPrivateState({ kind, commodity, module, routeText, sourceStation, destStation, quantity }) {
  if (kind === "route risk" || kind === "route danger") {
    return `${routeText} may be unsafe; warn ${commodity || "traffic"} before committing.`;
  }
  if (kind === "route success" || kind === "route reputation") {
    return `${routeText} is usable; confirm the clean lane and keep moving.`;
  }
  if (kind === "delivery receipt") {
    return `${quantity || "load"} ${commodity || "cargo"} has landed; close the receipt clearly.`;
  }
  if (kind === "ore pressure") {
    return `${sourceStation} has a useful ${commodity || "ore"} mark; report it without embellishment.`;
  }
  if (kind === "demand") {
    return `${sourceStation} needs ${commodity || "cargo"}; call the open haul plainly.`;
  }
  if (kind === "supply") {
    return `${sourceStation} has ${commodity || "cargo"} supply; keep the channel concise.`;
  }
  if (kind === "station risk") {
    return `${sourceStation} looks sharp; warn the next pilot before docking.`;
  }
  if (kind === "station trust") {
    return `${sourceStation} is paying clean; mark trust for the next run.`;
  }
  return module
    ? `${module} movement needs a clear readback between ${sourceStation} and ${destStation}.`
    : `${routeText} is active; hold the channel and listen for reply.`;
}

function makeFrame({ seed, source, kind, role, state, commodity, module, route, sourceStation, destStation, quantity, tick, line }) {
  const cleanLine = cleanAscii(line);
  const routeText = route || (sourceStation && destStation ? `${sourceStation}>${destStation}` : "");
  const fields = {
    role,
    state,
    memory: kind,
    commodity: commodity || "",
    module: module || "",
    source_station: sourceStation || "",
    dest_station: destStation || "",
    route: routeText,
    quantity: String(quantity || ""),
    tick: String(tick || ""),
  };
  const groundingTerms = [
    commodity,
    module,
    sourceStation,
    destStation,
    routeText,
    kind,
    state,
  ].filter(Boolean);
  const id = idFor([seed, source, kind, role, state, commodity, module, routeText, quantity, tick, cleanLine]);
  return {
    id,
    source: "signal_sim_log_template",
    source_id: source,
    speaker: `PILOT N${tick % 4}`,
    role,
    state,
    fields,
    private_state: cleanAscii(pilotPrivateState({
      kind,
      commodity,
      module,
      routeText,
      sourceStation,
      destStation,
      quantity,
    })),
    line: cleanLine,
    prompt: `RANKED: ${cleanLine}\nVOICE: `,
    target: cleanLine,
    expected_output: cleanLine,
    grounding_terms: [...new Set(groundingTerms)],
    stations: [...new Set([sourceStation, destStation].filter(Boolean))],
    commodities: commodity ? [commodity] : [],
  };
}

function buildFrames(options) {
  const frames = [];
  const routes = routePairs();
  for (let pass = 0; pass < options.repeat; pass += 1) {
    let tick = 12000 + pass * 5000;
    for (const [routeIndex, routeInfo] of routes.entries()) {
      for (const [commodityIndex, commodity] of commodities.entries()) {
        for (const [kindIndex, kind] of memoryKinds.entries()) {
          const role = kind === "ore pressure" ? "miner" : roles[(routeIndex + commodityIndex + kindIndex) % roles.length];
          const state = states[(routeIndex + commodityIndex + kindIndex + pass) % states.length];
          const quantity = 4 + ((routeIndex * 7 + commodityIndex * 3 + kindIndex + pass) % 128);
          const context = {
            ...routeInfo,
            commodity,
            role,
            state,
            quantity,
          };
          for (const [variant, template] of lineTemplates(kind, role).entries()) {
            tick += 1;
            frames.push(makeFrame({
              seed: options.seed,
              source: `pass${pass}:route${routeIndex}:commodity${commodityIndex}:kind${kindIndex}:variant${variant}`,
              kind,
              role,
              state,
              commodity,
              route: routeInfo.route,
              sourceStation: routeInfo.source,
              destStation: routeInfo.dest,
              quantity,
              tick,
              line: template(context),
            }));
          }
        }
      }
    }

    for (const [routeIndex, routeInfo] of routes.entries()) {
      for (const [moduleIndex, module] of modules.entries()) {
        const state = states[(routeIndex + moduleIndex + pass) % states.length];
        const role = module === "Shipyard" ? "hauler" : roles[(routeIndex + moduleIndex) % roles.length];
        const context = { ...routeInfo, module, state, quantity: moduleIndex + 1 };
        for (const [variant, template] of moduleTemplates().entries()) {
          tick += 1;
          frames.push(makeFrame({
            seed: options.seed,
            source: `pass${pass}:route${routeIndex}:module${moduleIndex}:variant${variant}`,
            kind: "scaffold pressure",
            role,
            state,
            module,
            route: routeInfo.route,
            sourceStation: routeInfo.source,
            destStation: routeInfo.dest,
            quantity: moduleIndex + 1,
            tick,
            line: template(context),
          }));
        }
      }
    }
  }
  const unique = new Map();
  for (const frame of frames) {
    unique.set(frame.line, frame);
  }
  return [...unique.values()].sort((a, b) => a.id.localeCompare(b.id));
}

function writeJsonl(filePath, records) {
  fs.writeFileSync(filePath, records.map((record) => `${JSON.stringify(record)}\n`).join(""), "utf8");
}

function trainingPair(frame) {
  return {
    id: frame.id,
    speaker: frame.speaker,
    kind: frame.role,
    private_state: frame.private_state,
    expected_output: frame.expected_output,
  };
}

function trainingText(frame) {
  return `${frame.prompt}${frame.target}\nEND\n`;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });
  const frames = buildFrames(options);
  const framesPath = path.join(outDir, "sim-log-frames.jsonl");
  const trainingPairsPath = path.join(outDir, "training-pairs.jsonl");
  const corpusPath = path.join(outDir, "corpus.txt");
  const voicePath = path.join(outDir, "sim-log-voice.txt");
  const manifestPath = path.join(outDir, "manifest.json");
  writeJsonl(framesPath, frames);
  writeJsonl(trainingPairsPath, frames.map(trainingPair));
  fs.writeFileSync(corpusPath, `SIGNAL_SIM_LOG_CORPUS_V1\n\n${frames.map(trainingText).join("\n")}`, "utf8");
  fs.writeFileSync(voicePath, `${frames.map((frame) => frame.line).join("\n")}\n`, "utf8");
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.signal_sim_log_corpus.v1",
    created_at: new Date().toISOString(),
    corpus_path: corpusPath,
    frames_path: framesPath,
    training_pairs_path: trainingPairsPath,
    voice_path: voicePath,
    frames: frames.length,
    repeat: options.repeat,
    stations,
    commodities,
    modules,
    memory_kinds: memoryKinds,
    notes: [
      "Deterministic sentence templates over Signal simulation vocabulary.",
      "Binary chain logs are not treated as prose; they motivate the structured fields only.",
      "training-pairs.jsonl is the paired private_state to expected_output dataset; sim-log-voice.txt is output-only for the tiniest raw models.",
    ],
  }, null, 2)}\n`, "utf8");
  console.log(`frames=${frames.length}`);
  console.log(`corpus=${corpusPath}`);
  console.log(`frames_path=${framesPath}`);
  console.log(`training_pairs=${trainingPairsPath}`);
  console.log(`voice_path=${voicePath}`);
  console.log(`manifest=${manifestPath}`);
}

main();
