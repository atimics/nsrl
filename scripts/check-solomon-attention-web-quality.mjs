#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { SolomonAttentionSampler } from "../web/attention-sampler.js";

let modelPath = "web/assets/solomon-attention.nsrllmm";
let textIndexPath = "web/assets/solomon-spirit-text-signatures.tsv";
let allNames = false;
let summary = false;

for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--model") {
    modelPath = requiredValue(process.argv, ++index, arg);
  } else if (arg === "--text-index") {
    textIndexPath = requiredValue(process.argv, ++index, arg);
  } else if (arg === "--all-names") {
    allNames = true;
  } else if (arg === "--summary") {
    summary = true;
  } else if (!arg.startsWith("-") && modelPath === "web/assets/solomon-attention.nsrllmm") {
    modelPath = arg;
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}

const sampler = new SolomonAttentionSampler(readFileSync(modelPath));

const prompts = allNames ? allNamePrompts(textIndexPath) : [
  {
    prompt: "king solomon seal",
    prefix: "Solomon selects ",
  },
  {
    prompt: "seal of Bael",
    prefix: "Solomon selects Bael: ",
  },
  {
    prompt: "seal of Buer",
    prefix: "Solomon selects Buer: ",
  },
  {
    prompt: "Agares goetic seal",
    prefix: "Solomon selects Agares: ",
  },
  {
    prompt: "seal of Stolas",
    prefix: "Solomon selects Stolas: ",
  },
  {
    prompt: "seal of Marbas",
    prefix: "Solomon selects Marbas: ",
  },
  {
    prompt: "Marbas",
    prefix: "Solomon selects Marbas: ",
  },
];

let checked = 0;
for (const { prompt, prefix } of prompts) {
  const result = sampler.sample(prompt, {
    seed: 13,
    topK: 1,
    maxTextTokens: 220,
  });
  const {
    text_source: textSource,
    image_source: imageSource,
    text_lm_order: textLmOrder,
    text_lm_min_order: textLmMinOrder,
  } = result.metadata;
  const terminalOk = allNames || /[.!?]"?$/.test(result.text);
  const textOk =
    result.text.startsWith(prefix) &&
    terminalOk &&
    !hasWeakRepeat(result.text);
  const sourceOk = textSource === "embedded_text_lm_strict";
  const lmOk = textLmOrder >= 12 && textLmMinOrder >= 3;
  const imageOk = imageSource === "embedded_image_memory_strict";
  if (!textOk || !sourceOk || !lmOk || !imageOk) {
    console.error(
      JSON.stringify({
        prompt,
        text: result.text,
        textSource,
        textLmOrder,
        textLmMinOrder,
        imageSource,
        textOk,
        sourceOk,
        lmOk,
        imageOk,
      }),
    );
    process.exit(1);
  }
  checked += 1;
  if (!summary) {
    console.log(
      JSON.stringify({
        prompt,
        text: result.text,
        textSource,
        textLmOrder,
        textLmMinOrder,
        imageSource,
      }),
    );
  }
}

if (summary) {
  console.log(
    JSON.stringify({
      schema: "nsrl.solomon_attention_web_quality_summary.v1",
      model: modelPath,
      prompts: checked,
      allNames,
    }),
  );
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function allNamePrompts(tsvPath) {
  const lines = readFileSync(tsvPath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  const primaryNameIndex = header.indexOf("primary_name");
  if (primaryNameIndex < 0) {
    throw new Error(`${tsvPath} is missing primary_name`);
  }
  return lines
    .filter(Boolean)
    .map((line) => line.split("\t")[primaryNameIndex])
    .filter(Boolean)
    .map((name) => {
      const normalizedName = normalizeNameForText(name);
      return {
        prompt: `seal of ${name}`,
        prefix: `Solomon selects ${normalizedName}: `,
      };
    });
}

function normalizeNameForText(name) {
  return String(name || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "");
}

function hasWeakRepeat(text) {
  return /(.)\1{5,}/.test(text) || hasRepeatedWordNgram(text, 4);
}

function hasRepeatedWordNgram(text, size) {
  const words = (text.toLowerCase().match(/[a-z']{2,}/g) || []).filter(
    (word) => word !== "thee",
  );
  const seen = new Set();
  for (let index = 0; index + size <= words.length; index += 1) {
    const key = words.slice(index, index + size).join(" ");
    if (seen.has(key)) {
      return true;
    }
    seen.add(key);
  }
  return false;
}
