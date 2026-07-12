#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  runDir: "",
  manifest: "",
  minSeedVariants: 3,
  maxIntraPromptDistance: 8192,
  maxTargetDistance: 24576,
  minInterClassDistance: 1024,
  minTargetInkCells: 8,
  maxTargetInkCells: 224,
  minEvalClassTop1: 1,
  expectedTargetSource: "decoded-latent",
};

const expectedPrompts = new Map([
  ["crocell", { prompt: "Crocell", number: 49, name: "Crocell" }],
  ["stolas", { prompt: "Stolas", number: 36, name: "Stolas" }],
  ["bael", { prompt: "Bael", number: 1, name: "Bael" }],
  [
    "hidden-geometry-waters",
    { prompt: "hidden geometry and rushing waters", number: 49, name: "Crocell" },
  ],
  [
    "astronomy-herbs-teacher",
    { prompt: "astronomy and herbs teacher", number: 36, name: "Stolas" },
  ],
]);

const grid = 16;
const gridBins = grid * grid;

function usage() {
  console.log(
    [
      "Usage: check-solomon-prior-smoke.mjs --run-dir PATH [options]",
      "",
      "Verifies the Solomon prior smoke artifacts prove learned target routing,",
      "fixed-prompt seed stability, and non-collapsed generated layouts.",
      "",
      "Options:",
      "  --manifest PATH",
      "  --min-seed-variants N",
      "  --max-intra-prompt-distance N",
      "  --max-target-distance N",
      "  --min-inter-class-distance N",
      "  --min-target-ink-cells N",
      "  --max-target-ink-cells N",
      "  --min-eval-class-top1 N",
      "  --expected-target-source SOURCE",
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
    } else if (arg === "--run-dir") {
      config.runDir = requireValue(argv, ++index, arg);
    } else if (arg === "--manifest") {
      config.manifest = requireValue(argv, ++index, arg);
    } else if (arg === "--min-seed-variants") {
      config.minSeedVariants = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-intra-prompt-distance") {
      config.maxIntraPromptDistance = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-target-distance") {
      config.maxTargetDistance = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-inter-class-distance") {
      config.minInterClassDistance = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-target-ink-cells") {
      config.minTargetInkCells = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-target-ink-cells") {
      config.maxTargetInkCells = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-eval-class-top1") {
      config.minEvalClassTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--expected-target-source") {
      config.expectedTargetSource = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.runDir) {
    throw new Error("--run-dir is required");
  }
  if (!config.manifest) {
    config.manifest = path.join(config.runDir, "manifest.tsv");
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

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function readTsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    throw new Error(`${filePath} is empty`);
  }
  const lines = text.split(/\r?\n/);
  const header = lines[0].split("\t");
  return lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.row_index = rowIndex + 2;
    return row;
  });
}

function readLastJsonLine(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    throw new Error(`${filePath} has no JSON rows`);
  }
  return JSON.parse(lines[lines.length - 1]);
}

function resolveOutDir(outDir, runDir) {
  if (fs.existsSync(path.join(outDir, "trace.json"))) {
    return outDir;
  }
  const marker = "/samples/";
  const markerIndex = outDir.indexOf(marker);
  if (markerIndex >= 0) {
    const suffix = outDir.slice(markerIndex + marker.length);
    const remapped = path.join(runDir, "samples", suffix);
    if (fs.existsSync(path.join(remapped, "trace.json"))) {
      return remapped;
    }
  }
  const remapped = path.join(runDir, "samples", path.basename(outDir));
  if (fs.existsSync(path.join(remapped, "trace.json"))) {
    return remapped;
  }
  return outDir;
}

function readGeneratedSignature(row, runDir) {
  const outDir = resolveOutDir(row.out_dir, runDir);
  const tracePath = path.join(outDir, "trace.json");
  const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  const imageSize = Number(trace.image_size || 128);
  const sampleCount = Number(trace.samples || 1);
  const rawPath = path.join(outDir, `samples.ink${imageSize}.u8`);
  const bytes = fs.readFileSync(rawPath);
  const imageBytes = imageSize * imageSize;
  if (bytes.length !== imageBytes * sampleCount) {
    throw new Error(`${rawPath} has ${bytes.length} bytes, expected ${imageBytes * sampleCount}`);
  }
  const signatures = [];
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    signatures.push(imageSignature(bytes, sampleIndex * imageBytes, imageSize));
  }
  return {
    trace,
    signature: meanSignature(signatures),
  };
}

function imageSignature(bytes, offset, imageSize) {
  const signature = new Array(gridBins).fill(0);
  for (let gy = 0; gy < grid; gy += 1) {
    const y0 = Math.floor((gy * imageSize) / grid);
    const y1 = Math.floor(((gy + 1) * imageSize) / grid);
    for (let gx = 0; gx < grid; gx += 1) {
      const x0 = Math.floor((gx * imageSize) / grid);
      const x1 = Math.floor(((gx + 1) * imageSize) / grid);
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1; y += 1) {
        const row = offset + y * imageSize;
        for (let x = x0; x < x1; x += 1) {
          sum += bytes[row + x];
          count += 1;
        }
      }
      signature[gy * grid + gx] = count > 0 && Math.floor(sum / count) >= 64 ? 255 : 0;
    }
  }
  return signature;
}

function meanSignature(signatures) {
  const out = new Array(gridBins).fill(0);
  for (const signature of signatures) {
    for (let index = 0; index < gridBins; index += 1) {
      out[index] += signature[index];
    }
  }
  return out.map((value) => Math.round(value / Math.max(1, signatures.length)));
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < gridBins; index += 1) {
    distance += Math.abs(left[index] - right[index]);
  }
  return distance;
}

function meanInk(signature) {
  return Math.round(signature.reduce((sum, value) => sum + value, 0) / signature.length);
}

function binarySignature(signature) {
  if (!Array.isArray(signature) || signature.length !== gridBins) {
    throw new Error(`latent_target_signature must contain ${gridBins} bins`);
  }
  return signature.map((value) => (Number(value) >= 64 ? 255 : 0));
}

function inkCells(signature) {
  return signature.reduce((sum, value) => sum + (value >= 128 ? 1 : 0), 0);
}

function metric(metrics, group, field) {
  return metrics?.[group]?.[field] ?? 0;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const rows = readTsv(config.manifest);
  const groups = new Map();
  const failures = [];
  const targetDistances = [];

  for (const row of rows) {
    const slug = row.prompt_slug || row.slug;
    if (!expectedPrompts.has(slug)) {
      failures.push(`unexpected prompt slug '${slug}' at manifest row ${row.row_index}`);
      continue;
    }
    const expected = expectedPrompts.get(slug);
    const sample = readGeneratedSignature(row, config.runDir);
    row.trace = sample.trace;
    row.generated_signature = sample.signature;
    row.generated_mean_ink = meanInk(sample.signature);
    row.target_signature = binarySignature(sample.trace.latent_target_signature);
    row.target_ink_cells = inkCells(row.target_signature);
    row.target_distance = signatureDistance(row.generated_signature, row.target_signature);
    if (sample.trace.latent_target_source !== config.expectedTargetSource) {
      failures.push(
        `${slug}/${row.seed_variant || "seed"} used ${sample.trace.latent_target_source || "<none>"}, expected ${config.expectedTargetSource}`,
      );
    }
    if (sample.trace.model_format !== "NSRLTCH") {
      failures.push(`${slug}/${row.seed_variant || "seed"} used model format ${sample.trace.model_format || "<none>"}, expected NSRLTCH`);
    }
    if (Number(sample.trace.feature_channels || 0) < 30) {
      failures.push(
        `${slug}/${row.seed_variant || "seed"} model feature_channels ${sample.trace.feature_channels || 0} < 30`,
      );
    }
    if (Number(sample.trace.latent_target_number || 0) !== expected.number) {
      failures.push(
        `${slug}/${row.seed_variant || "seed"} routed to ${sample.trace.latent_target_number}, expected ${expected.number}`,
      );
    }
    targetDistances.push({
      slug,
      seed: row.seed_variant || `row-${row.row_index}`,
      distance: row.target_distance,
      target_ink_cells: row.target_ink_cells,
    });
    if (row.target_distance > config.maxTargetDistance) {
      failures.push(
        `${slug}/${row.seed_variant || "seed"} target distance ${row.target_distance} > ${config.maxTargetDistance}`,
      );
    }
    if (row.target_ink_cells < config.minTargetInkCells || row.target_ink_cells > config.maxTargetInkCells) {
      failures.push(
        `${slug}/${row.seed_variant || "seed"} target ink cells ${row.target_ink_cells} outside ${config.minTargetInkCells}..${config.maxTargetInkCells}`,
      );
    }
    if (!groups.has(slug)) {
      groups.set(slug, []);
    }
    groups.get(slug).push(row);
  }

  for (const [slug, expected] of expectedPrompts.entries()) {
    const group = groups.get(slug) || [];
    if (group.length < config.minSeedVariants) {
      failures.push(`${slug} has ${group.length} seed variants, expected at least ${config.minSeedVariants}`);
      continue;
    }
    const targetNumbers = new Set(group.map((row) => String(row.trace.latent_target_number || "")));
    if (targetNumbers.size !== 1 || !targetNumbers.has(String(expected.number))) {
      failures.push(`${slug} target number changed across seeds: ${[...targetNumbers].join(",")}`);
    }
  }

  const intraDistances = [];
  for (const [slug, group] of groups.entries()) {
    for (let left = 0; left < group.length; left += 1) {
      for (let right = left + 1; right < group.length; right += 1) {
        const distance = signatureDistance(
          group[left].generated_signature,
          group[right].generated_signature,
        );
        intraDistances.push({
          slug,
          left: group[left].seed_variant || `row-${group[left].row_index}`,
          right: group[right].seed_variant || `row-${group[right].row_index}`,
          distance,
        });
        if (distance > config.maxIntraPromptDistance) {
          failures.push(
            `${slug} seed distance ${distance} > ${config.maxIntraPromptDistance} (${group[left].seed_variant || "left"} vs ${group[right].seed_variant || "right"})`,
          );
        }
      }
    }
  }

  const groupSignatures = [...groups.entries()].map(([slug, group]) => ({
    slug,
    number: expectedPrompts.get(slug).number,
    signature: meanSignature(group.map((row) => row.generated_signature)),
    mean_ink: Math.round(group.reduce((sum, row) => sum + row.generated_mean_ink, 0) / group.length),
  }));
  const distances = [];
  for (let left = 0; left < groupSignatures.length; left += 1) {
    for (let right = left + 1; right < groupSignatures.length; right += 1) {
      const leftGroup = groupSignatures[left];
      const rightGroup = groupSignatures[right];
      if (leftGroup.number === rightGroup.number) {
        continue;
      }
      const distance = signatureDistance(leftGroup.signature, rightGroup.signature);
      distances.push({ left: leftGroup.slug, right: rightGroup.slug, distance });
      if (distance < config.minInterClassDistance) {
        failures.push(
          `${leftGroup.slug} vs ${rightGroup.slug} generated signature distance ${distance} < ${config.minInterClassDistance}`,
        );
      }
    }
  }

  const evalPath = path.join(config.runDir, "eval-ledger.jsonl");
  const evalRow = fs.existsSync(evalPath) ? readLastJsonLine(evalPath) : null;
  const evalClassTop1 = evalRow ? metric(evalRow.prior_eval ?? evalRow.retrieval_eval, "all", "class_top1_per_mille") : 0;
  if (evalClassTop1 < config.minEvalClassTop1) {
    failures.push(`eval class_top1_per_mille ${evalClassTop1} < ${config.minEvalClassTop1}`);
  }

  const report = {
    schema: "nsrl.solomon_prior_smoke_check.v1",
    passed: failures.length === 0,
    run_dir: config.runDir,
    manifest: config.manifest,
    expected_target_source: config.expectedTargetSource,
    prompt_groups: groupSignatures.map((group) => ({
      slug: group.slug,
      expected_number: group.number,
      rows: (groups.get(group.slug) || []).length,
      mean_ink: group.mean_ink,
    })),
    min_inter_class_distance: distances.length
      ? Math.min(...distances.map((entry) => entry.distance))
      : 0,
    max_intra_prompt_distance: intraDistances.length
      ? Math.max(...intraDistances.map((entry) => entry.distance))
      : 0,
    max_target_distance: targetDistances.length
      ? Math.max(...targetDistances.map((entry) => entry.distance))
      : 0,
    intra_prompt_distances: intraDistances,
    inter_class_distances: distances,
    target_distances: targetDistances,
    eval_class_top1_per_mille: evalClassTop1,
    failures,
  };
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) {
    process.exit(1);
  }
}

main();
