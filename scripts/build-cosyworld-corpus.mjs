#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "data/processed/cosyworld-corpus",
  itemCsv: "/Users/ratimics/develop/cosyworld/items_export.csv",
  maxCsvItems: 24,
  repeat: 1,
  seed: "cosyworld-v1",
};

const avatarSeeds = [
  {
    name: "Miri Buttonwake",
    description: "a gentle tea-cartographer with brass spectacles and a pocket full of route ribbons",
    personality: "patient, curious, and delighted by useful shortcuts",
  },
  {
    name: "Tallow Finch",
    description: "a candle-mender who can hear when a room is lonely",
    personality: "soft-spoken, practical, and brave in small ways",
  },
  {
    name: "Brindle Mosscup",
    description: "a moss-garden keeper who keeps weather notes in tiny clay bells",
    personality: "warm, observant, and quietly mischievous",
  },
  {
    name: "Pippa Glimmerloam",
    description: "a cheerful archivist who files dreams by scent and season",
    personality: "bright, tidy, and very protective of lost stories",
  },
  {
    name: "Orin Teakettle",
    description: "a hearthside tinkerer with soot on one cheek and a reliable repair kit",
    personality: "steady, inventive, and fond of second chances",
  },
  {
    name: "Sable Nook",
    description: "a moonlit concierge who remembers every guest's favorite blanket",
    personality: "gracious, watchful, and impossible to hurry",
  },
];

const locationSeeds = [
  {
    name: "Candlelit Map Room",
    description: "cedar tables, pinned thread-maps, and rain ticking softly on the skylight",
  },
  {
    name: "Mossglass Pantry",
    description: "green jars, warm scones, and a window where the herb labels rearrange themselves politely",
  },
  {
    name: "Rainbell Conservatory",
    description: "ferns, hanging chimes, and little puddles that ring when moonlight touches them",
  },
  {
    name: "Pennywhistle Quay",
    description: "small boats, brass lanterns, and rope ladders leading to cloudberry stalls",
  },
  {
    name: "Teacup Observatory",
    description: "a round hilltop room where star charts curl around porcelain telescopes",
  },
  {
    name: "Blanket Fort Commons",
    description: "pillows, council cushions, and a treaty table made from an old bakery door",
  },
];

const fallbackItems = [
  { name: "Pocket Hearth Lantern", type: "artifact", rarity: "common" },
  { name: "Tea-Thread Compass", type: "quest", rarity: "uncommon" },
  { name: "Moonberry Spoon", type: "consumable", rarity: "common" },
  { name: "Rainmoth Cloak", type: "armor", rarity: "rare" },
  { name: "Kindling Key", type: "key", rarity: "uncommon" },
  { name: "Dandelion Repair Kit", type: "artifact", rarity: "common" },
];

const itemTypes = new Set(["weapon", "armor", "consumable", "quest", "key", "artifact"]);
const rarities = new Set(["common", "uncommon", "rare", "legendary", "mythic"]);
const itemCharms = [
  "keeps soup warm on long walks",
  "points home by firefly light",
  "finds misplaced buttons before breakfast",
  "turns nervous silence into a useful hum",
  "stores one kind memory and returns it when needed",
  "glows when a promise has been kept",
  "marks safe paths with a line of gold dust",
  "rings softly when a friend is nearby",
];

function usage() {
  console.log(`Usage: node scripts/build-cosyworld-corpus.mjs [options]

Options:
  --out-dir PATH        Output directory [${defaults.outDir}]
  --item-csv PATH       Optional CosyWorld item export CSV [${defaults.itemCsv}]
  --max-csv-items N     Maximum item names to borrow from CSV [${defaults.maxCsvItems}]
  --repeat N            Repeat deterministic frame sweep [${defaults.repeat}]
  --seed TEXT           Stable id seed [${defaults.seed}]
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
    if (["maxCsvItems", "repeat"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 0) {
        throw new Error(`${arg} requires a non-negative integer`);
      }
    } else {
      options[key] = value;
    }
  }
  if (options.repeat < 1) {
    throw new Error("--repeat must be positive");
  }
  return options;
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function cleanAscii(text) {
  return String(text ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function compact(text) {
  return cleanAscii(text).replace(/\s+/g, " ").trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function keyFor(name) {
  return compact(name).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (ch === '"') {
      if (inQuotes && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && text[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  row.push(field);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

function csvRecords(filePath, maxItems) {
  const resolved = resolveRepoPath(filePath);
  if (!filePath || !fs.existsSync(resolved) || maxItems === 0) {
    return [];
  }
  const rows = parseCsv(fs.readFileSync(resolved, "utf8"));
  if (rows.length < 2) return [];
  const headers = rows[0].map((header) => compact(header));
  const out = [];
  for (const row of rows.slice(1)) {
    const record = Object.fromEntries(headers.map((header, index) => [header, compact(row[index] || "")]));
    if (record.name) out.push(record);
    if (out.length >= maxItems) break;
  }
  return out;
}

function normalizeItem(record, index) {
  const base = record || fallbackItems[index % fallbackItems.length];
  const name = compact(base.name) || fallbackItems[index % fallbackItems.length].name;
  const type = itemTypes.has(String(base.type || "").toLowerCase())
    ? String(base.type).toLowerCase()
    : fallbackItems[index % fallbackItems.length].type;
  const rarity = rarities.has(String(base.rarity || "").toLowerCase())
    ? String(base.rarity).toLowerCase()
    : fallbackItems[index % fallbackItems.length].rarity;
  const charm = itemCharms[index % itemCharms.length];
  return {
    key: keyFor(name),
    name,
    type,
    rarity,
    description: `${name} is a ${rarity} ${type} that ${charm}.`,
    properties: {
      effects: [{ type: "utility", value: 1, duration: 1 }],
      cosy_charm: charm,
    },
  };
}

function makeFrame({ seed, sourceId, kind, name, line, fields, terms, privateState }) {
  const cleanLine = compact(line);
  const id = idFor([seed, kind, sourceId, name, cleanLine]);
  return {
    id,
    source: "cosyworld_template",
    source_id: sourceId,
    domain: "cosyworld",
    speaker: kind === "avatar" ? name : "COSYWORLD",
    role: kind,
    state: "generated",
    fields: Object.fromEntries(
      Object.entries(fields || {}).map(([key, value]) => [key, compact(value)])
    ),
    private_state: compact(privateState || ""),
    line: cleanLine,
    prompt: `RANKED: ${cleanLine}\nVOICE: `,
    target: cleanLine,
    expected_output: cleanLine,
    grounding_terms: [...new Set([kind, name, ...(terms || [])].filter(Boolean).map(compact))],
  };
}

function trainingText(frame) {
  return `${frame.prompt}${frame.target}\nEND\n`;
}

function buildCatalog(options) {
  const csvItems = csvRecords(options.itemCsv, options.maxCsvItems);
  const rawItems = csvItems.length > 0 ? csvItems : fallbackItems;
  const items = rawItems.map(normalizeItem);
  return {
    avatars: avatarSeeds.map((avatar) => ({
      ...avatar,
      status: "alive",
      model: "auto",
      imagePrompt: avatar.description,
    })),
    locations: locationSeeds.map((location) => ({
      ...location,
      type: "thread",
      imagePrompt: `${location.name}: ${location.description} Overhead RPG Map Style`,
    })),
    items,
  };
}

function buildFrames(options, catalog) {
  const frames = [];
  for (let pass = 0; pass < options.repeat; pass += 1) {
    for (const [index, avatar] of catalog.avatars.entries()) {
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:avatar${index}:description`,
        kind: "avatar",
        name: avatar.name,
        fields: {
          name: avatar.name,
          description: avatar.description,
          personality: avatar.personality,
          model: avatar.model,
        },
        terms: [avatar.personality],
        privateState: `${avatar.name} wants the room to feel safe before offering help.`,
        line: `${avatar.name} is ${avatar.description}, ${avatar.personality}.`,
      }));
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:avatar${index}:greeting`,
        kind: "avatar",
        name: avatar.name,
        fields: {
          name: avatar.name,
          personality: avatar.personality,
        },
        terms: [avatar.personality],
        privateState: `${avatar.name} notices who needs comfort and chooses one useful kindness.`,
        line: `${avatar.name} offers ${avatar.personality.split(",")[0]} care and a small useful kindness.`,
      }));
    }

    for (const [index, location] of catalog.locations.entries()) {
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:location${index}:description`,
        kind: "location",
        name: location.name,
        fields: {
          name: location.name,
          type: location.type,
          description: location.description,
        },
        terms: [location.type],
        privateState: `${location.name} holds a quiet welcome for careful travelers.`,
        line: `${location.name} has ${location.description}.`,
      }));
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:location${index}:arrival`,
        kind: "location",
        name: location.name,
        fields: {
          name: location.name,
          description: location.description,
        },
        terms: ["arrival"],
        privateState: `${location.name} makes the threshold feel settled before anything strange happens.`,
        line: `Arriving at ${location.name}, the air feels safe, busy, and quietly enchanted.`,
      }));
    }

    for (const [index, item] of catalog.items.entries()) {
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:item${index}:description`,
        kind: "item",
        name: item.name,
        fields: {
          key: item.key,
          name: item.name,
          type: item.type,
          rarity: item.rarity,
          description: item.description,
        },
        terms: [item.type, item.rarity],
        privateState: `${item.name} waits to be useful without demanding attention.`,
        line: item.description,
      }));
      frames.push(makeFrame({
        seed: options.seed,
        sourceId: `pass${pass}:item${index}:use`,
        kind: "item",
        name: item.name,
        fields: {
          key: item.key,
          type: item.type,
          rarity: item.rarity,
          effect: item.properties.cosy_charm,
        },
        terms: [item.type, item.rarity, item.properties.cosy_charm],
        privateState: `${item.name} answers use with a practical charm, not a speech about itself.`,
        line: `Using ${item.name} ${item.properties.cosy_charm} and leaves the room a little kinder.`,
      }));
    }
  }
  const unique = new Map();
  for (const frame of frames) {
    unique.set([frame.role, frame.line].join("\0"), frame);
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

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const catalog = buildCatalog(options);
  const frames = buildFrames(options, catalog);
  const corpus = [
    "COSYWORLD_CORPUS_V1\n",
    "DIVISION characters=avatars places=locations objects=items voice=nsrl\n",
    ...frames.map(trainingText),
  ].join("\n");

  const framesPath = path.join(outDir, "frames.jsonl");
  const trainingPairsPath = path.join(outDir, "training-pairs.jsonl");
  const corpusPath = path.join(outDir, "corpus.txt");
  const voicePath = path.join(outDir, "voice.txt");
  const catalogPath = path.join(outDir, "catalog.json");
  const manifestPath = path.join(outDir, "manifest.json");

  writeJsonl(framesPath, frames);
  writeJsonl(trainingPairsPath, frames.map(trainingPair));
  fs.writeFileSync(corpusPath, corpus, "utf8");
  fs.writeFileSync(voicePath, `${frames.map((frame) => frame.line).join("\n")}\n`, "utf8");
  fs.writeFileSync(catalogPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.cosyworld_corpus.v1",
    created_at: new Date().toISOString(),
    corpus_path: corpusPath,
    frames_path: framesPath,
    training_pairs_path: trainingPairsPath,
    voice_path: voicePath,
    catalog_path: catalogPath,
    item_csv_path: resolveRepoPath(options.itemCsv),
    item_csv_used: fs.existsSync(resolveRepoPath(options.itemCsv)),
    max_csv_items: options.maxCsvItems,
    repeat: options.repeat,
    avatars: catalog.avatars.length,
    locations: catalog.locations.length,
    items: catalog.items.length,
    frames: frames.length,
    corpus_bytes: Buffer.byteLength(corpus),
    notes: [
      "Dependency-free deterministic corpus for cheap local generation.",
      "CSV item names are softened into cosy descriptions instead of copying long item prose.",
      "Frames use the same RANKED/VOICE shape as Signal so dashboard chat and lexeme training can reuse existing tooling.",
      "training-pairs.jsonl is the paired private_state to expected_output dataset; voice.txt is output-only for the tiniest raw models.",
    ],
  }, null, 2)}\n`, "utf8");

  console.log(`frames=${frames.length}`);
  console.log(`corpus=${corpusPath}`);
  console.log(`frames_path=${framesPath}`);
  console.log(`training_pairs=${trainingPairsPath}`);
  console.log(`catalog=${catalogPath}`);
  console.log(`manifest=${manifestPath}`);
}

try {
  main();
} catch (error) {
  console.error(`build-cosyworld-corpus: ${error.message}`);
  process.exit(1);
}
