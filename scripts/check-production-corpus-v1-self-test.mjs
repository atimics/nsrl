#!/usr/bin/env node

import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { build, check, sha256 } from "./production-corpus-v1.mjs";

const workDir = await mkdtemp(path.join(os.tmpdir(), "nsrl-production-corpus-v1-"));

try {
  const prompt = "A uniquely contaminated evaluation sentence must never appear inside the training corpus.";
  const repeated = "Deterministic integer models preserve exact replay across machines while retaining useful context for careful generation.";
  const nearA = "The patient engineer records voltage current pressure temperature and timing before repairing the station relay.";
  const nearB = "The patient engineer records voltage current pressure temperature and timing before repairing the station cable.";
  const pages = [
    repeated,
    repeated,
    nearA,
    nearB,
    `A document begins normally. ${prompt} It then continues with material that must be quarantined.`,
    ...Array.from({ length: 120 }, (_, index) => (
      `Document ${index} explains a distinct mechanism using numbered component ${index}, stable measurement ${index + 17}, and reproducible observation ${index + 31}. `
      + `It contains enough prose to exercise deterministic splitting and artifact hashing without copying evaluation material.`
    )),
  ];
  const sourceText = `<|source:simplewiki|>\n${pages.map((text, index) => `<|page:fixture-${index}|>\n${text}`).join("\n\n")}\n`;
  const sourcePath = path.join(workDir, "source.txt");
  const panelPath = path.join(workDir, "panel.txt");
  const configPath = path.join(workDir, "config.json");
  const outDir = path.join(workDir, "out");
  await writeFile(sourcePath, sourceText);
  await writeFile(panelPath, `${prompt}\n`);
  await writeFile(configPath, `${JSON.stringify({
    schema: "nsrl.production_corpus_config.v1",
    contract_id: "production-corpus-v1",
    document: { min_bytes: 64, max_bytes: 4096 },
    near_dedup: { shingle_words: 5, signature_size: 12, bands: 3, threshold_per_mille: 800 },
    contamination: {
      shingle_words: 5,
      min_direct_bytes: 32,
      prompt_overlap_per_mille: 500,
      panels: [{ id: "fixture", path: panelPath, format: "lines", expected_sha256: sha256(`${prompt}\n`) }],
    },
    split: { seed: "fixture-seed", permyriad: { train: 6000, dev: 2000, test: 2000 }, require_nonempty: true },
    tokenizer_training: { source_split: "train", max_bytes: 8192, target_vocab_size: 512, min_pair_frequency: 2 },
    sources: [{
      id: "fixture-pages",
      input_path: sourcePath,
      expected_sha256: sha256(sourceText),
      format: "nsrl_simplewiki_pages",
      source_url: "https://example.invalid/fixture",
      license_id: "CC0-1.0",
      rights_basis_url: "https://example.invalid/fixture-license",
      rights_status: "approved",
      attribution: "self-test fixture",
    }],
  }, null, 2)}\n`);

  const manifestPath = await build(configPath, outDir);
  await check(manifestPath);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.equal(manifest.counts.exact_duplicates_removed, 1);
  assert.ok(manifest.counts.near_duplicates_removed >= 1);
  assert.equal(manifest.counts.contaminated_quarantined, 1);
  assert.ok(manifest.artifacts.train.documents > 0);
  assert.ok(manifest.artifacts.dev.documents > 0);
  assert.ok(manifest.artifacts.test.documents > 0);
  console.log(JSON.stringify({ schema: "nsrl.production_corpus_self_test.v1", ok: true }));
} finally {
  await rm(workDir, { recursive: true, force: true });
}
