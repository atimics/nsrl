#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const directory = path.join(root, "benchmarks/open-generation-v1");
const panelPath = path.join(directory, "development-panel.tsv");
const manifestPath = path.join(directory, "manifest.tsv");
const tokenizerPath = path.join(directory, "dev-tokenizer.nsrlbpe");
const schema = "nsrl.open_generation_prompt.v1";
const manifestSchema = "nsrl.open_generation_manifest.v1";
const contract = "open-generation-v1";
const hiddenCommitment = "b7be1577444b77b9b2c0aecb8955655cfa8003a0f1600649a2039e72b3a6c53d";
const hiddenPanelPath = path.join(root, "data/private/open-generation-v1/hidden-panel.txt");
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;

const prompts = [
  ["continuation-harbor", "continuation", "", "At first light the harbor clerk found an unsigned entry in the tide ledger. Continue the account in clear narrative prose."],
  ["continuation-observatory", "continuation", "", "The observatory clock stopped seven minutes before the meteor arrived. Continue the scene and preserve that timing detail."],
  ["style-winter-garden", "constrained-style", "", "Describe a city garden in winter in restrained prose. Do not use the words cold, snow, white, or silent."],
  ["style-field-notes", "constrained-style", "", "Write field notes from a patient engineer. Use short paragraphs, concrete measurements, and no exclamation marks."],
  ["explain-integer-replay", "explanation", "integer", "Explain to a careful programmer how an integer-only language model can produce replayable logits and seeded samples."],
  ["explain-tide-table", "explanation", "tide", "A tide table lists time, predicted height, and observed height. Explain how to detect a drifting sensor without inventing missing observations."],
  ["dialogue-archive", "dialogue", "record", "Write a dialogue between two archivists who disagree about deleting a corrupted record. Let each make one technically credible argument."],
  ["dialogue-repair", "dialogue", "gear", "Write a dialogue between a mechanic and an apprentice diagnosing a four-gear chain that rings its bell too early."],
  ["context-brass-key", "long-context-reference", "Mara", "Remember these facts: the brass key belongs to Mara; the red key belongs to Ivo; the wooden token opens no door. After discussing two irrelevant weather reports, answer who owns the brass key and why."],
  ["context-signal-code", "long-context-reference", "7319", "The northern relay code is 7319. The eastern relay code is 2046. A later note incorrectly claims every relay uses 1111. Explain which code should be sent to the northern relay."],
  ["repeat-valley", "adversarial-repetition", "", "Continue without repeating any four-word phrase: The signal crossed the valley once, then faded behind the ridge."],
  ["repeat-river", "adversarial-repetition", "", "Write a coherent paragraph about a river survey. Avoid loops, copied sentences, list spam, and repeated four-word phrases."],
];

function hex(value) {
  return Buffer.from(value, "utf8").toString("hex") || "-";
}

function hash(bytes) {
  let value = fnvOffset;
  for (const byte of bytes) value = ((value ^ BigInt(byte)) * fnvPrime) & fnvMask;
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function panelText() {
  const header = "schema\tcontract\tpartition\tid\tcategory\tmax_new_tokens\trequired_phrase_hex\tprompt_hex";
  const rows = prompts.map(([id, category, required, prompt]) =>
    [schema, contract, "development", id, category, 512, hex(required), hex(prompt)].join("\t"),
  );
  return `${header}\n${rows.join("\n")}\n`;
}

function manifestText(panel) {
  const header = [
    "schema", "contract", "tokenizer", "tokenizer_hash", "development_panel",
    "development_panel_hash", "hidden_test_sha256", "prompt_count", "max_prompt_bytes",
    "generation_tokens", "sampling_seeds", "retained_improvement_per_mille",
    "max_repeat_4gram_share_per_mille", "min_unique_4gram_share_per_mille", "min_entropy_q10",
    "min_utf8_valid_per_mille", "min_context_use_per_mille",
    "min_distractor_resistance_per_mille", "min_human_preference_delta_per_mille",
  ].join("\t");
  const row = [
    manifestSchema,
    contract,
    "dev-tokenizer.nsrlbpe",
    hash(fs.readFileSync(tokenizerPath)),
    "development-panel.tsv",
    hash(Buffer.from(panel)),
    hiddenCommitment,
    prompts.length,
    2048,
    512,
    "7,19,43,97",
    900,
    150,
    600,
    2048,
    1000,
    750,
    700,
    -100,
  ].join("\t");
  return `${header}\n${row}\n`;
}

function main() {
  const check = process.argv.slice(2).includes("--check");
  if (process.argv.slice(2).some((arg) => arg !== "--check")) throw new Error("only --check is supported");
  if (!fs.existsSync(tokenizerPath)) throw new Error("development tokenizer is missing");
  if (fs.existsSync(hiddenPanelPath) && sha256(fs.readFileSync(hiddenPanelPath)) !== hiddenCommitment) {
    throw new Error("local hidden panel does not match its frozen SHA-256 commitment");
  }
  const panel = panelText();
  const manifest = manifestText(panel);
  if (check) {
    if (fs.readFileSync(panelPath, "utf8") !== panel) throw new Error("development panel is stale");
    if (fs.readFileSync(manifestPath, "utf8") !== manifest) throw new Error("open-generation manifest is stale");
    process.stdout.write(JSON.stringify({ checked: true, contract }) + "\n");
    return;
  }
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(panelPath, panel);
  fs.writeFileSync(manifestPath, manifest);
  process.stdout.write(JSON.stringify({ frozen: true, contract, prompts: prompts.length }) + "\n");
}

main();
