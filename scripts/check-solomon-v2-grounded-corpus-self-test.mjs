#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_v2_grounded_corpus_self_test.v1";
const sourceTextTasks = ["explain", "image-to-explain", "text-image-explain", "description-to-image"];
const attributeTasks = ["image-to-attributes"];
const spirits = [
  {
    id: 1,
    name: "Bael",
    text: "Bael teaches hidden geometry waters and governs eastern kings with a crown.",
    excerpt: "teaches hidden geometry waters and governs eastern kings",
  },
  {
    id: 2,
    name: "Agares",
    text: "Agares brings language motion and returns runaways with earthly power.",
    excerpt: "brings language motion and returns runaways",
  },
  {
    id: 3,
    name: "Vassago",
    text: "Vassago discovers things hidden and speaks of past future treasures.",
    excerpt: "discovers things hidden and speaks of past future",
  },
];

const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

function usage() {
  console.log([
    "Usage: check-solomon-v2-grounded-corpus-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic grounded v2 corpus artifacts and proves the checker",
    "rejects weak source overlap, source placeholders, generic attribute ranks,",
    "bad source hashes, non-name explain prompts, non-source description image",
    "prompts, name-leaking attribute prompts, and missing grounded attribute",
    "task coverage.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "", keep: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
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

function writeFixture(root, name, mutate = () => {}) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  const examplesPath = path.join(dir, "examples.jsonl");
  const textIndexPath = path.join(dir, "text-index.tsv");
  const rows = buildExampleRows();
  mutate(rows);
  fs.writeFileSync(
    textIndexPath,
    [
      "number\tprimary_name\ttext",
      ...spirits.map((spirit) => [spirit.id, spirit.name, spirit.text].join("\t")),
      "",
    ].join("\n"),
    "utf8",
  );
  fs.writeFileSync(
    examplesPath,
    `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  );
  return { dir, examplesPath, textIndexPath };
}

function buildExampleRows() {
  const rows = [];
  for (const spirit of spirits) {
    for (const task of sourceTextTasks) {
      rows.push(sourceTaskRow(spirit, task));
    }
    rows.push(attributeRow(spirit));
  }
  return rows;
}

function sourceTaskRow(spirit, task) {
  const text = task === "description-to-image"
    ? spirit.excerpt
    : `Solomon selects ${spirit.name}: ${spirit.excerpt}`;
  return withSourceProvenance(spirit, {
    task,
    spirit_id: spirit.id,
    primary_name: spirit.name,
    prompt: sourceTaskPrompt(spirit, task),
    text,
  });
}

function sourceTaskPrompt(spirit, task) {
  if (task === "explain") return spirit.name;
  if (task === "description-to-image") return spirit.excerpt;
  return `${task} ${spirit.name}`;
}

function attributeRow(spirit) {
  return withSourceProvenance(spirit, {
    task: "image-to-attributes",
    spirit_id: spirit.id,
    primary_name: spirit.name,
    prompt: "seal attributes",
    text: [
      `${spirit.name} rank source-derived office`,
      `appearance ${spirit.excerpt}`,
      `office ${spirit.excerpt}`,
    ].join("; "),
  });
}

function withSourceProvenance(spirit, row) {
  return {
    ...row,
    source_spirit_id: spirit.id,
    source_text_hash: fnv64TextHex(normalizeSourceText(spirit.text)),
    source_excerpt: spirit.excerpt,
    source_excerpt_hash: fnv64TextHex(normalizeSourceText(spirit.excerpt)),
  };
}

function runChecker(fixture) {
  return childProcess.spawnSync(process.execPath, [
    "scripts/check-solomon-v2-grounded-corpus.mjs",
    "--examples",
    fixture.examplesPath,
    "--text-index",
    fixture.textIndexPath,
    "--expect-spirits",
    String(spirits.length),
    "--min-source-overlap-tokens",
    "2",
    "--min-attribute-source-overlap-tokens",
    "4",
    "--max-source-placeholder-rows",
    "0",
    "--max-attribute-generic-rank-rows",
    "0",
    "--require-source-provenance",
    "--require-name-source-explain",
    "--require-description-source-image",
    "--require-image-attribute-generic-prompt",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function readReport(stdout) {
  const start = String(stdout || "").indexOf("{");
  if (start < 0) return null;
  return JSON.parse(String(stdout).slice(start));
}

function caseResult(definition, result, report) {
  const actualOk = result.status === 0 && report?.ok === true;
  const haystack = [
    ...(report?.errors || []),
    result.stdout || "",
    result.stderr || "",
  ].join("\n");
  const requiredErrorOk = definition.requiredError
    ? haystack.includes(definition.requiredError)
    : true;
  return {
    name: definition.name,
    expect_ok: definition.expectOk,
    ok: actualOk === definition.expectOk && requiredErrorOk,
    status: result.status,
    passed: report?.ok === true,
    required_error: definition.requiredError || "",
    errors: report?.errors || [],
    stdout_tail: result.stdout ? tailLines(result.stdout, 20) : "",
    stderr_tail: result.stderr ? tailLines(result.stderr, 20) : "",
  };
}

function tailLines(text, maxLines) {
  const lines = String(text).split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function writeReport(outPath, report) {
  if (!outPath) return;
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function normalizeSourceText(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, " - ")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/[^ -~]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function fnv64TextHex(value) {
  let hash = FNV64_OFFSET;
  for (const byte of Buffer.from(String(value), "utf8")) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-grounded-corpus-self-test-"));
  const cases = [];
  try {
    const definitions = [
      { name: "good", expectOk: true, mutate: () => {} },
      {
        name: "bad-source-overlap",
        expectOk: false,
        requiredError: "source overlap 0 < 2",
        mutate: (rows) => {
          rows.find((row) => row.task === "explain" && row.spirit_id === 1).text =
            "Solomon selects Bael: unrelated vapor silence marble";
        },
      },
      {
        name: "bad-source-placeholder",
        expectOk: false,
        requiredError: "grounded task explain source placeholder rows 1 > 0",
        mutate: (rows) => {
          rows.find((row) => row.task === "explain" && row.spirit_id === 1).text =
            "Solomon selects Bael: described in the source teaches hidden geometry waters";
        },
      },
      {
        name: "bad-attribute-generic-rank",
        expectOk: false,
        requiredError: "attribute text used generic rank placeholder",
        mutate: (rows) => {
          rows.find((row) => row.task === "image-to-attributes" && row.spirit_id === 1).text =
            "Bael rank Goetic spirit; appearance teaches hidden geometry waters; office teaches hidden geometry waters";
        },
      },
      {
        name: "bad-source-provenance-hash",
        expectOk: false,
        requiredError: "source_text_hash 0x0000000000000000 !=",
        mutate: (rows) => {
          rows.find((row) => row.task === "image-to-explain" && row.spirit_id === 2).source_text_hash =
            "0x0000000000000000";
        },
      },
      {
        name: "bad-explain-prompt",
        expectOk: false,
        requiredError: 'explain prompt "seal of Bael" != primary name "Bael"',
        mutate: (rows) => {
          rows.find((row) => row.task === "explain" && row.spirit_id === 1).prompt = "seal of Bael";
        },
      },
      {
        name: "bad-description-prompt",
        expectOk: false,
        requiredError: "description-to-image prompt source overlap 1 < 2",
        mutate: (rows) => {
          rows.find((row) => row.task === "description-to-image" && row.spirit_id === 1).prompt = "seal of Bael";
        },
      },
      {
        name: "bad-attribute-name-prompt",
        expectOk: false,
        requiredError: 'image-to-attributes prompt leaks primary name "Bael"',
        mutate: (rows) => {
          rows.find((row) => row.task === "image-to-attributes" && row.spirit_id === 1).prompt =
            "attributes of Bael";
        },
      },
      {
        name: "bad-missing-attribute-task",
        expectOk: false,
        requiredError: "examples are missing grounded task image-to-attributes",
        mutate: (rows) => {
          for (let index = rows.length - 1; index >= 0; index -= 1) {
            if (rows[index].task === "image-to-attributes") {
              rows.splice(index, 1);
            }
          }
        },
      },
    ];
    for (const definition of definitions) {
      const fixture = writeFixture(root, definition.name, definition.mutate);
      const result = runChecker(fixture);
      const report = readReport(result.stdout);
      cases.push(caseResult(definition, result, report));
    }
    const report = {
      schema,
      ok: cases.every((item) => item.ok),
      root,
      kept: config.keep,
      cases,
    };
    writeReport(config.outPath, report);
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) {
      process.exit(1);
    }
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
