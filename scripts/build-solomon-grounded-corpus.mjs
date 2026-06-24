#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  textIndex: "data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv",
  dictionnaireText: "data/raw/dictionnaire-infernal-1863-djvu.txt",
  scotText: "data/raw/scot-discovery-witchcraft-djvu.txt",
  outDir: "data/processed/key-solomon-goetia-grounded-corpus-v1",
  variantsPerRow: 32,
  excerptRadius: 560,
  maxSourceExcerpts: 3,
};

const schema = "nsrl.solomon_grounded_text_signature_corpus.v1";
const sourceUrls = {
  goetia: "https://www.gutenberg.org/ebooks/72679",
  dictionnaire: "https://archive.org/details/dictionnaireinfe00coll_1",
  scot: "https://archive.org/details/discoveryofwitch00scot",
};

function usage() {
  console.log(
    [
      "Usage: build-solomon-grounded-corpus.mjs [--text-index PATH] [--out-dir PATH]",
      "       [--dictionnaire-text PATH] [--scot-text PATH] [--variants-per-row N]",
      "",
      "Builds a source-grounded expanded Solomon text/signature TSV. The first",
      "nine columns match solomon-spirit-text-signatures.tsv so existing tools",
      "can read it; extra columns record variant/source provenance.",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--dictionnaire-text") {
      config.dictionnaireText = requireValue(argv, ++index, arg);
    } else if (arg === "--scot-text") {
      config.scotText = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--variants-per-row") {
      config.variantsPerRow = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--excerpt-radius") {
      config.excerptRadius = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-source-excerpts") {
      config.maxSourceExcerpts = parsePositive(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePositive(value, flag) {
  if (!/^[0-9]+$/.test(value) || Number(value) === 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function readBaseRows(tsvPath) {
  const text = fs.readFileSync(tsvPath, "utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error(`${tsvPath} has no data rows`);
  }
  const header = lines[0].split("\t");
  const required = [
    "number",
    "primary_name",
    "aliases",
    "slice_id",
    "label",
    "source_file",
    "ink_128_u8",
    "signature_8x8",
    "text",
  ];
  for (const column of required) {
    if (!header.includes(column)) {
      throw new Error(`${tsvPath} missing required column ${column}`);
    }
  }
  return lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.number = Number(row.number);
    if (!Number.isInteger(row.number) || row.number < 1 || row.number > 72) {
      throw new Error(`${tsvPath} row ${rowIndex + 2} has invalid number`);
    }
    row.aliasList = unique(
      [row.primary_name, ...String(row.aliases || "").split("|")]
        .map(cleanName)
        .filter(Boolean),
    );
    return row;
  });
}

function readOptionalSource(filePath, sourceName) {
  if (!filePath || !fs.existsSync(filePath)) {
    return null;
  }
  const text = fs.readFileSync(filePath, "utf8");
  return {
    name: sourceName,
    path: filePath,
    text,
    folded: foldWithMap(text),
  };
}

function foldWithMap(text) {
  let folded = "";
  const map = [];
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const replacement = foldText(char);
    for (let offset = 0; offset < replacement.length; offset += 1) {
      folded += replacement[offset];
      map.push(index);
    }
  }
  return { text: folded.toLowerCase(), map };
}

function foldText(text) {
  return String(text)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/æ/g, "ae")
    .replace(/Æ/g, "Ae")
    .replace(/œ/g, "oe")
    .replace(/Œ/g, "Oe")
    .replace(/[^A-Za-z0-9]+/g, " ");
}

function cleanName(value) {
  return String(value || "")
    .replace(/[.]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function unique(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const key = foldText(value).trim().toLowerCase();
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(value);
  }
  return out;
}

function sourceExcerptsForRow(row, sources, config) {
  const excerpts = [
    {
      source: "goetia",
      ref: `${row.number}:${row.primary_name}`,
      text: row.text,
    },
  ];
  for (const source of sources) {
    if (!source) {
      continue;
    }
    for (const snippet of findSourceSnippets(row, source, config)) {
      excerpts.push(snippet);
    }
  }
  return excerpts;
}

function findSourceSnippets(row, source, config) {
  const snippets = [];
  const seen = new Set();
  const aliases = row.aliasList.flatMap((name) => nameVariants(name));
  for (const alias of aliases) {
    const needle = foldText(alias).trim().toLowerCase();
    if (needle.length < 3) {
      continue;
    }
    const re = new RegExp(`(?:^| )${escapeRegExp(needle)}(?: |$)`, "g");
    let match;
    while ((match = re.exec(source.folded.text)) && snippets.length < config.maxSourceExcerpts) {
      const foldedStart = match.index;
      const originalStart = source.folded.map[foldedStart] ?? 0;
      const originalEnd = source.folded.map[Math.min(source.folded.map.length - 1, foldedStart + match[0].length)] ?? originalStart;
      const start = Math.max(0, originalStart - config.excerptRadius);
      const end = Math.min(source.text.length, originalEnd + config.excerptRadius);
      const text = compactText(source.text.slice(start, end));
      const key = foldText(text).slice(0, 180);
      if (!text || seen.has(key) || !looksRelevant(text, row)) {
        continue;
      }
      seen.add(key);
      snippets.push({
        source: source.name,
        ref: `${source.path}:${originalStart}`,
        text,
      });
    }
  }
  return snippets;
}

function nameVariants(name) {
  const base = cleanName(name);
  const variants = [base];
  const lower = base.toLowerCase();
  const manual = {
    bael: ["Bael", "Bael", "Baal", "Beal"],
    agares: ["Agares", "Agarès", "Agreas", "Aguarès"],
    samigina: ["Samigina", "Gamigin", "Gamygin"],
    marbas: ["Marbas", "Barbas"],
    amon: ["Amon", "Aamon", "Ammon"],
    paimon: ["Paimon", "Paymon", "Poymon"],
    sitri: ["Sitri", "Bitru", "Sytry"],
    beleth: ["Beleth", "Byleth", "Bilet"],
    leraje: ["Leraje", "Leraie", "Loray"],
    eligos: ["Eligos", "Abigor"],
    botis: ["Botis", "Otis"],
    bathin: ["Bathin", "Bathym"],
    sallos: ["Sallos", "Saleos", "Zaleos"],
    purson: ["Purson", "Pursan"],
    ipos: ["Ipos", "Ipes", "Ayporos"],
    aim: ["Aim", "Aym", "Haborym"],
    naberius: ["Naberius", "Cerberus"],
    glasyalabolas: ["Glasyalabolas", "Glasya-Labolas", "Caacrinolaas"],
    bune: ["Bune", "Bime"],
    ronove: ["Ronove", "Ronwe"],
    berith: ["Berith", "Beal", "Bofry", "Bolfry"],
    astaroth: ["Astaroth", "Astarot"],
    forneus: ["Forneus", "Foras"],
    marchosias: ["Marchosias", "Marchocias"],
    phenex: ["Phenex", "Phoenix", "Pheynix", "Fenix"],
    stolas: ["Stolas", "Stolos"],
    malphas: ["Malphas", "Malpas"],
    raum: ["Raum", "Raym"],
    focalor: ["Focalor", "Forcalor", "Furcalor"],
    vepar: ["Vepar", "Vephar"],
    vine: ["Vine", "Vinea"],
    bifrons: ["Bifrons", "Bifrous"],
    crocell: ["Crocell", "Crokel", "Procell"],
    furcas: ["Furcas", "Forcas"],
    alloces: ["Alloces", "Alocas"],
    caim: ["Caim", "Camio"],
    murmur: ["Murmur", "Murmus"],
    orobas: ["Orobas", "Orobas"],
    gomory: ["Gomory", "Gemory", "Gamori"],
    ose: ["Ose", "Oze", "Voso"],
    amy: ["Amy", "Avnas"],
    orias: ["Orias", "Oriax"],
    vapula: ["Vapula", "Naphula"],
    andras: ["Andras"],
    haures: ["Haures", "Flauros", "Flavros"],
    andrealphus: ["Andrealphus", "Androalphus"],
    kimaris: ["Kimaris", "Cimeies", "Cimejes"],
    amdusias: ["Amdusias", "Amduscias"],
    belial: ["Belial"],
    decarabia: ["Decarabia", "Carabia"],
    seere: ["Seere", "Sear", "Seir"],
    dantalion: ["Dantalion"],
    andromalius: ["Andromalius"],
  };
  if (manual[lower]) {
    variants.push(...manual[lower]);
  }
  return unique(variants);
}

function compactText(text) {
  return String(text)
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .replace(/[^\S\r\n]+/g, " ")
    .trim();
}

function looksRelevant(text, row) {
  const folded = foldText(text).toLowerCase();
  if (!row.aliasList.some((name) => folded.includes(foldText(name).trim().toLowerCase()))) {
    return false;
  }
  const signalWords = [
    "demon",
    "daemon",
    "spirit",
    "roi",
    "duc",
    "marquis",
    "prince",
    "president",
    "earl",
    "king",
    "legion",
    "enfer",
    "infernal",
    "goetia",
    "pseudomonarchia",
  ];
  return signalWords.some((word) => folded.includes(word));
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function factsForRow(row, excerpts) {
  const joined = excerpts.map((excerpt) => excerpt.text).join(" ");
  const sentences = splitSentences(joined);
  const rank = firstMatch(joined, [
    /\b(?:is|called|named)\s+(?:a|an)\s+([^.;,]{0,60}?\b(?:king|duke|prince|marquis|president|earl|count|knight)\b)/i,
    /\b(?:roi|duc|prince|marquis|president|comte|chevalier)\b[^.;,]*/i,
  ]);
  const legions = firstMatch(joined, [
    /\b(?:ruleth|governeth|commandeth|gouverne|obey|obeissent|obéissent)[^.;]{0,80}?\b([0-9]+|sixty-six|thirty-one|twenty-six|thirty|forty|fifty|twenty|six)\s+legions?\b/i,
    /\b([0-9]+|sixty-six|thirty-one|twenty-six|thirty|forty|fifty|twenty|six)\s+legions?\b/i,
  ]);
  return {
    rank: cleanFact(rank),
    legions: cleanFact(legions),
    appearance: selectSentences(sentences, [
      "appeareth",
      "appears",
      "form",
      "shape",
      "tetes",
      "têtes",
      "heads",
      "voice",
      "riding",
      "carrying",
      "montre",
      "figure",
    ], 3),
    offices: selectSentences(sentences, [
      "office",
      "maketh",
      "teaches",
      "teacheth",
      "giveth",
      "gives",
      "declare",
      "discover",
      "causeth",
      "cause",
      "bring",
      "build",
      "heal",
      "love",
      "lang",
      "science",
      "knowledge",
      "invisible",
      "invisibles",
      "rend",
      "apprend",
    ], 4),
    keywords: keywords(joined, row),
  };
}

function splitSentences(text) {
  return compactText(text)
    .split(/(?<=[.!?])\s+|;\s+|\s+--\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 20 && sentence.length < 360);
}

function firstMatch(text, regexes) {
  for (const regex of regexes) {
    const match = text.match(regex);
    if (match) {
      return match[1] || match[0];
    }
  }
  return "";
}

function cleanFact(text) {
  return compactText(text)
    .replace(/^a\s+/i, "")
    .replace(/[.,;:]+$/g, "")
    .slice(0, 160);
}

function selectSentences(sentences, needles, limit) {
  const scored = [];
  for (const sentence of sentences) {
    const folded = foldText(sentence).toLowerCase();
    let score = 0;
    for (const needle of needles) {
      if (folded.includes(foldText(needle).toLowerCase())) {
        score += 1;
      }
    }
    if (score > 0) {
      scored.push({ score, sentence: compactText(sentence) });
    }
  }
  scored.sort((left, right) => right.score - left.score || left.sentence.length - right.sentence.length);
  return unique(scored.map((item) => item.sentence)).slice(0, limit);
}

function keywords(text, row) {
  const stop = new Set([
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "shall",
    "which",
    "spirit",
    "spirits",
    "called",
    "named",
    "unto",
    "upon",
    "before",
    "after",
    "when",
    "where",
    "their",
    "there",
    "them",
    "they",
    "dans",
    "avec",
    "pour",
    "plus",
    "dont",
    "elle",
    "cette",
    "sont",
    "sous",
  ]);
  const aliasFolded = new Set(row.aliasList.map((name) => foldText(name).trim().toLowerCase()));
  const counts = new Map();
  for (const token of foldText(text).toLowerCase().split(/\s+/)) {
    if (token.length < 4 || stop.has(token) || aliasFolded.has(token)) {
      continue;
    }
    counts.set(token, (counts.get(token) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 18)
    .map(([token]) => token);
}

function variantsForRow(row, excerpts, config) {
  const facts = factsForRow(row, excerpts);
  const sourceNames = unique(excerpts.map((excerpt) => excerpt.source));
  const sourceRefText = sourceNames.join("+");
  const appearance = facts.appearance[0] || `${row.primary_name} has a distinct Goetic appearance`;
  const office = facts.offices[0] || `${row.primary_name} has an office described in the source texts`;
  const keywordsText = facts.keywords.slice(0, 10).join(" ");
  const aliasText = row.aliasList.join(" ");
  const seed = [
    {
      kind: "canonical",
      text: `${row.primary_name} ${aliasText} ${row.text}`,
    },
    {
      kind: "source-summary",
      text: `${row.primary_name} is ${facts.rank || "a Goetic spirit"}; ${appearance}; ${office}; ${facts.legions || "the source records its legions"}.`,
    },
    {
      kind: "short-prompt",
      text: `${row.primary_name} ${facts.rank || "Goetic spirit"} ${keywordsText}`,
    },
    {
      kind: "appearance-prompt",
      text: `${row.primary_name} seal prompt: ${appearance}`,
    },
    {
      kind: "office-prompt",
      text: `${row.primary_name} seal prompt: ${office}`,
    },
    {
      kind: "alias-prompt",
      text: `${aliasText} ${facts.rank || "Goetic spirit"} ${facts.keywords.slice(0, 8).join(" ")}`,
    },
  ];

  for (const excerpt of excerpts) {
    seed.push({
      kind: `source-${excerpt.source}`,
      text: `${row.primary_name} grounded by ${excerpt.source}: ${excerpt.text}`,
    });
  }
  for (const sentence of facts.appearance.slice(1)) {
    seed.push({ kind: "appearance-variant", text: `${row.primary_name}: ${sentence}` });
  }
  for (const sentence of facts.offices.slice(1)) {
    seed.push({ kind: "office-variant", text: `${row.primary_name}: ${sentence}` });
  }

  const variants = [];
  let cursor = 0;
  while (variants.length < config.variantsPerRow) {
    const base = seed[cursor % seed.length];
    const windowStart = (cursor * 3) % Math.max(1, facts.keywords.length);
    const window = rotate(facts.keywords, windowStart).slice(0, 8).join(" ");
    let text = base.text;
    if (cursor >= seed.length && window) {
      text = `${base.text} Keywords: ${window}.`;
    }
    variants.push({
      variantId: `${String(row.number).padStart(2, "0")}-${String(variants.length).padStart(3, "0")}`,
      kind: base.kind,
      sourceLanes: sourceRefText,
      supportTerms: window || keywordsText,
      text: compactText(text),
    });
    cursor += 1;
  }
  return variants;
}

function rotate(values, start) {
  if (values.length === 0) {
    return [];
  }
  const out = [];
  for (let index = 0; index < values.length; index += 1) {
    out.push(values[(start + index) % values.length]);
  }
  return out;
}

function escapeTsv(value) {
  return String(value ?? "")
    .replace(/\t/g, " ")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const baseRows = readBaseRows(config.textIndex);
  const sources = [
    readOptionalSource(config.dictionnaireText, "dictionnaire"),
    readOptionalSource(config.scotText, "scot"),
  ];

  fs.mkdirSync(config.outDir, { recursive: true });

  const allExcerpts = [];
  const expandedRows = [];
  const synthesisRequests = [];
  const sourceCoverage = {};
  for (const row of baseRows) {
    const excerpts = sourceExcerptsForRow(row, sources, config);
    for (const excerpt of excerpts) {
      allExcerpts.push({
        number: row.number,
        primary_name: row.primary_name,
        source: excerpt.source,
        ref: excerpt.ref,
        text: excerpt.text,
      });
      sourceCoverage[excerpt.source] = (sourceCoverage[excerpt.source] || 0) + 1;
    }
    const variants = variantsForRow(row, excerpts, config);
    synthesisRequests.push({
      schema: "nsrl.solomon_grounded_synthesis_request.v1",
      number: row.number,
      primary_name: row.primary_name,
      aliases: row.aliasList,
      target: {
        slice_id: row.slice_id,
        signature_8x8: row.signature_8x8,
      },
      instruction:
        "Generate short, source-grounded English prompt variants. Do not add facts not present in source_excerpts. Keep variants tied to this demon only.",
      source_excerpts: excerpts,
    });
    for (const variant of variants) {
      expandedRows.push({ row, variant });
    }
  }

  const expandedTsvPath = path.join(config.outDir, "grounded-text-signatures.tsv");
  const header = [
    "number",
    "primary_name",
    "aliases",
    "slice_id",
    "label",
    "source_file",
    "ink_128_u8",
    "signature_8x8",
    "text",
    "variant_id",
    "source_lanes",
    "prompt_kind",
    "support_terms",
    "source_urls",
  ];
  const sourceUrlText = Object.entries(sourceUrls)
    .map(([key, value]) => `${key}:${value}`)
    .join("|");
  const tsv = [
    header.join("\t"),
    ...expandedRows.map(({ row, variant }) =>
      [
        row.number,
        row.primary_name,
        row.aliases,
        row.slice_id,
        row.label,
        row.source_file,
        row.ink_128_u8,
        row.signature_8x8,
        variant.text,
        variant.variantId,
        variant.sourceLanes,
        variant.kind,
        variant.supportTerms,
        sourceUrlText,
      ].map(escapeTsv).join("\t"),
    ),
  ].join("\n");
  fs.writeFileSync(expandedTsvPath, `${tsv}\n`, "utf8");

  const excerptsPath = path.join(config.outDir, "source-excerpts.jsonl");
  const requestsPath = path.join(config.outDir, "synthesis-requests.jsonl");
  writeJsonl(excerptsPath, allExcerpts);
  writeJsonl(requestsPath, synthesisRequests);

  const manifestPath = path.join(config.outDir, "manifest.json");
  fs.writeFileSync(
    manifestPath,
    `${JSON.stringify(
      {
        schema,
        base_text_index: config.textIndex,
        expanded_text_index: path.relative(config.outDir, expandedTsvPath),
        source_excerpts: path.relative(config.outDir, excerptsPath),
        synthesis_requests: path.relative(config.outDir, requestsPath),
        rows: baseRows.length,
        expanded_rows: expandedRows.length,
        variants_per_row: config.variantsPerRow,
        source_coverage: sourceCoverage,
        source_urls: sourceUrls,
        optional_sources: sources.filter(Boolean).map((source) => ({
          name: source.name,
          path: source.path,
          bytes: Buffer.byteLength(source.text),
        })),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  console.log(
    JSON.stringify({
      schema,
      tsv: expandedTsvPath,
      excerpts: excerptsPath,
      synthesis_requests: requestsPath,
      rows: baseRows.length,
      expanded_rows: expandedRows.length,
      source_coverage: sourceCoverage,
    }),
  );
}

main();
