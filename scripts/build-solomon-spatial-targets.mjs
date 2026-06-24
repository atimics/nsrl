#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  slicesManifest: "data/processed/key-solomon-goetia-bitmaps-pg72679/slices/manifest.json",
  outDir: "data/processed/key-solomon-goetia-spatial-targets-v1",
  imageSize: 128,
  targetGrids: [32, 64],
  kinds: ["seal-grid-cell"],
  inkThreshold: 64,
  distanceRadius: 8,
  strokeEpsilon: 1.2,
  previewColumns: 8,
};

const mapChannels = ["density", "stroke", "distance", "ring"];

function usage() {
  console.log(
    [
      "Usage: build-solomon-spatial-targets.mjs [options]",
      "",
      "Options:",
      "  --slices-manifest PATH   Solomon bitmap slice manifest",
      "  --out-dir PATH           Output artifact directory",
      "  --image-size N           Source ink tensor size, default 128",
      "  --target-grids LIST      Comma-separated condition grids, default 32,64",
      "  --kinds LIST             Comma-separated slice kinds, default seal-grid-cell",
      "  --ink-threshold N        Pixel threshold for stroke extraction, default 64",
      "  --distance-radius N      Source-pixel radius for distance channel, default 8",
      "  --stroke-epsilon N       RDP simplification epsilon in pixels, default 1.2",
      "  --preview-columns N      Contact-sheet columns, default 8",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, targetGrids: [...defaults.targetGrids], kinds: [...defaults.kinds] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--slices-manifest") {
      config.slicesManifest = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--image-size") {
      config.imageSize = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--target-grids") {
      config.targetGrids = parsePositiveIntegerList(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--kinds") {
      config.kinds = parseList(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--ink-threshold") {
      config.inkThreshold = parseByte(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--distance-radius") {
      config.distanceRadius = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--stroke-epsilon") {
      config.strokeEpsilon = parsePositiveNumber(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--preview-columns") {
      config.previewColumns = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
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

function parsePositiveInteger(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parsePositiveNumber(value, flag) {
  if (!/^(?:[1-9][0-9]*|0)(?:[.][0-9]+)?$/.test(value)) {
    throw new Error(`${flag} requires a non-negative number`);
  }
  return Number(value);
}

function parseByte(value, flag) {
  const parsed = parsePositiveInteger(value, flag);
  if (parsed > 255) {
    throw new Error(`${flag} must be <= 255`);
  }
  return parsed;
}

function parsePositiveIntegerList(value, flag) {
  const parsed = parseList(value, flag).map((item) => parsePositiveInteger(item, flag));
  if (new Set(parsed).size !== parsed.length) {
    throw new Error(`${flag} must not contain duplicate values`);
  }
  return parsed;
}

function parseList(value, flag) {
  const parsed = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (parsed.length === 0) {
    throw new Error(`${flag} requires at least one value`);
  }
  return parsed;
}

function spiritNumberForSlice(slice) {
  const match = slice.label.match(/^front-([345])-seal-grid-r(\d\d)-c(\d\d)$/);
  if (!match) {
    return null;
  }
  const front = Number(match[1]);
  const row = Number(match[2]);
  const col = Number(match[3]);
  return (front - 3) * 24 + (row - 1) * 4 + col;
}

function readSelectedRows(config) {
  const manifest = JSON.parse(fs.readFileSync(config.slicesManifest, "utf8"));
  const slicesRoot = path.dirname(path.dirname(config.slicesManifest));
  const expectedBytes = checkedImageBytes(config.imageSize);
  const rows = [];
  for (const slice of manifest.slices ?? []) {
    if (!config.kinds.includes(slice.kind)) {
      continue;
    }
    const inkRel = selectedInkPath(slice, config.imageSize);
    const inkPath = path.join(slicesRoot, stripSlicesPrefix(inkRel));
    const ink = fs.readFileSync(inkPath);
    if (ink.length !== expectedBytes) {
      throw new Error(`${inkPath} has ${ink.length} bytes, expected ${expectedBytes}`);
    }
    rows.push({
      number: spiritNumberForSlice(slice),
      sliceId: slice.id,
      label: slice.label,
      kind: slice.kind,
      sourceFile: slice.source_file,
      inkRel,
      ink: Uint8Array.from(ink),
    });
  }
  rows.sort((left, right) => {
    if (left.number != null && right.number != null) {
      return left.number - right.number;
    }
    if (left.number != null) {
      return -1;
    }
    if (right.number != null) {
      return 1;
    }
    return left.label.localeCompare(right.label, "en-US");
  });
  return rows;
}

function selectedInkPath(slice, imageSize) {
  if (imageSize === 128) {
    return slice.ink_128_u8;
  }
  if (imageSize === 256) {
    return slice.ink_256_u8;
  }
  throw new Error(`unsupported --image-size ${imageSize}; expected 128 or 256`);
}

function stripSlicesPrefix(value) {
  return value.startsWith("slices/") ? value.slice("slices/".length) : value;
}

function checkedImageBytes(imageSize) {
  const bytes = imageSize * imageSize;
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error("image byte count overflow");
  }
  return bytes;
}

function downsampleAverage(image, imageSize, grid) {
  const bins = grid * grid;
  const sums = new Uint32Array(bins);
  const counts = new Uint32Array(bins);
  for (let y = 0; y < imageSize; y += 1) {
    const binY = Math.floor((y * grid) / imageSize);
    for (let x = 0; x < imageSize; x += 1) {
      const binX = Math.floor((x * grid) / imageSize);
      const bin = binY * grid + binX;
      sums[bin] += image[y * imageSize + x];
      counts[bin] += 1;
    }
  }
  const out = Buffer.alloc(bins);
  for (let index = 0; index < bins; index += 1) {
    out[index] = Math.floor((sums[index] + Math.floor(counts[index] / 2)) / counts[index]);
  }
  return out;
}

function thresholdInk(ink, threshold) {
  const binary = new Uint8Array(ink.length);
  for (let index = 0; index < ink.length; index += 1) {
    binary[index] = ink[index] > threshold ? 1 : 0;
  }
  return binary;
}

function skeletonize(binary, imageSize) {
  const image = Uint8Array.from(binary);
  const at = (x, y) => image[y * imageSize + x];
  let changed = true;
  while (changed) {
    changed = false;
    for (const step of [0, 1]) {
      const remove = [];
      for (let y = 1; y < imageSize - 1; y += 1) {
        for (let x = 1; x < imageSize - 1; x += 1) {
          const index = y * imageSize + x;
          if (image[index] === 0) {
            continue;
          }
          const p2 = at(x, y - 1);
          const p3 = at(x + 1, y - 1);
          const p4 = at(x + 1, y);
          const p5 = at(x + 1, y + 1);
          const p6 = at(x, y + 1);
          const p7 = at(x - 1, y + 1);
          const p8 = at(x - 1, y);
          const p9 = at(x - 1, y - 1);
          const neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
          const count = neighbors.reduce((sum, value) => sum + value, 0);
          if (count < 2 || count > 6) {
            continue;
          }
          let transitions = 0;
          for (let n = 0; n < neighbors.length; n += 1) {
            if (neighbors[n] === 0 && neighbors[(n + 1) % neighbors.length] === 1) {
              transitions += 1;
            }
          }
          if (transitions !== 1) {
            continue;
          }
          const firstGate = step === 0 ? p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0 : p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0;
          if (firstGate) {
            remove.push(index);
          }
        }
      }
      if (remove.length > 0) {
        changed = true;
        for (const index of remove) {
          image[index] = 0;
        }
      }
    }
  }
  return image;
}

function binaryToInk(binary) {
  const out = Buffer.alloc(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    out[index] = binary[index] ? 255 : 0;
  }
  return out;
}

function distanceField(binary, imageSize, radius) {
  const out = Buffer.alloc(binary.length);
  const radiusSq = radius * radius;
  for (let y = 0; y < imageSize; y += 1) {
    for (let x = 0; x < imageSize; x += 1) {
      const index = y * imageSize + x;
      if (binary[index] !== 0) {
        out[index] = 255;
        continue;
      }
      let bestSq = radiusSq + 1;
      for (let dy = -radius; dy <= radius; dy += 1) {
        const yy = y + dy;
        if (yy < 0 || yy >= imageSize) {
          continue;
        }
        for (let dx = -radius; dx <= radius; dx += 1) {
          const xx = x + dx;
          if (xx < 0 || xx >= imageSize) {
            continue;
          }
          const distSq = dx * dx + dy * dy;
          if (distSq >= bestSq || distSq > radiusSq) {
            continue;
          }
          if (binary[yy * imageSize + xx] !== 0) {
            bestSq = distSq;
          }
        }
      }
      if (bestSq <= radiusSq) {
        out[index] = Math.max(0, Math.round(255 * (1 - Math.sqrt(bestSq) / radius)));
      }
    }
  }
  return out;
}

function ringPrior(imageSize) {
  const out = Buffer.alloc(checkedImageBytes(imageSize));
  const center = (imageSize - 1) / 2;
  const targetRadius = 0.78;
  const width = 0.12;
  for (let y = 0; y < imageSize; y += 1) {
    const ny = (y - center) / center;
    for (let x = 0; x < imageSize; x += 1) {
      const nx = (x - center) / center;
      const radius = Math.sqrt(nx * nx + ny * ny);
      const value = Math.max(0, 1 - Math.abs(radius - targetRadius) / width);
      out[y * imageSize + x] = Math.round(value * 255);
    }
  }
  return out;
}

function strokeGraph(skeleton, imageSize, epsilon) {
  const skeletonPixels = [];
  for (let index = 0; index < skeleton.length; index += 1) {
    if (skeleton[index] !== 0) {
      skeletonPixels.push(index);
    }
  }
  if (skeletonPixels.length === 0) {
    return { strokePixels: 0, nodeCount: 0, paths: [], simplifiedPointCount: 0 };
  }

  const point = (index) => [index % imageSize, Math.floor(index / imageSize)];
  const neighbors = (index) => {
    const x = index % imageSize;
    const y = Math.floor(index / imageSize);
    const out = [];
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || xx >= imageSize || yy < 0 || yy >= imageSize) {
          continue;
        }
        const neighbor = yy * imageSize + xx;
        if (skeleton[neighbor] !== 0) {
          out.push(neighbor);
        }
      }
    }
    return out;
  };

  const neighborCache = new Map();
  for (const index of skeletonPixels) {
    neighborCache.set(index, neighbors(index));
  }
  const nodeSet = new Set(
    skeletonPixels.filter((index) => (neighborCache.get(index)?.length ?? 0) !== 2),
  );
  const visitedEdges = new Set();
  const edgeKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const markEdge = (a, b) => visitedEdges.add(edgeKey(a, b));
  const hasEdge = (a, b) => visitedEdges.has(edgeKey(a, b));

  function walk(start, next) {
    const points = [point(start), point(next)];
    let previous = start;
    let current = next;
    markEdge(start, next);
    for (let guard = 0; guard < skeletonPixels.length + 1; guard += 1) {
      if (current !== start && nodeSet.has(current)) {
        break;
      }
      const choices = (neighborCache.get(current) ?? []).filter((candidate) => candidate !== previous);
      if (choices.length === 0) {
        break;
      }
      const unvisited = choices.find((candidate) => !hasEdge(current, candidate));
      if (unvisited == null) {
        break;
      }
      markEdge(current, unvisited);
      points.push(point(unvisited));
      previous = current;
      current = unvisited;
      if (current === start) {
        break;
      }
    }
    return points;
  }

  const paths = [];
  const starts = nodeSet.size > 0 ? [...nodeSet].sort((a, b) => a - b) : skeletonPixels;
  for (const start of starts) {
    const nextPixels = [...(neighborCache.get(start) ?? [])].sort((a, b) => a - b);
    for (const next of nextPixels) {
      if (hasEdge(start, next)) {
        continue;
      }
      const raw = walk(start, next);
      if (raw.length > 1) {
        paths.push(simplifyPath(raw, epsilon));
      }
    }
  }
  let simplifiedPointCount = 0;
  for (const pathPoints of paths) {
    simplifiedPointCount += pathPoints.length;
  }
  return {
    strokePixels: skeletonPixels.length,
    nodeCount: nodeSet.size,
    paths,
    simplifiedPointCount,
  };
}

function simplifyPath(points, epsilon) {
  if (points.length <= 2 || epsilon <= 0) {
    return points;
  }
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [start, end] = stack.pop();
    let bestIndex = -1;
    let bestDistance = 0;
    for (let index = start + 1; index < end; index += 1) {
      const distance = pointLineDistance(points[index], points[start], points[end]);
      if (distance > bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    if (bestIndex >= 0 && bestDistance > epsilon) {
      keep[bestIndex] = 1;
      stack.push([start, bestIndex], [bestIndex, end]);
    }
  }
  return points.filter((_, index) => keep[index] !== 0);
}

function pointLineDistance(point, start, end) {
  const [px, py] = point;
  const [sx, sy] = start;
  const [ex, ey] = end;
  const dx = ex - sx;
  const dy = ey - sy;
  if (dx === 0 && dy === 0) {
    return Math.hypot(px - sx, py - sy);
  }
  const t = Math.max(0, Math.min(1, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)));
  const xx = sx + t * dx;
  const yy = sy + t * dy;
  return Math.hypot(px - xx, py - yy);
}

function makeContactSheet(images, width, height, columns, gap) {
  const rows = Math.ceil(images.length / columns);
  const sheetWidth = columns * width + (columns - 1) * gap;
  const sheetHeight = rows * height + (rows - 1) * gap;
  const bytes = Buffer.alloc(sheetWidth * sheetHeight, 0);
  for (let imageIndex = 0; imageIndex < images.length; imageIndex += 1) {
    const image = images[imageIndex];
    const col = imageIndex % columns;
    const row = Math.floor(imageIndex / columns);
    const left = col * (width + gap);
    const top = row * (height + gap);
    for (let y = 0; y < height; y += 1) {
      const sourceOffset = y * width;
      const targetOffset = (top + y) * sheetWidth + left;
      for (let x = 0; x < width; x += 1) {
        bytes[targetOffset + x] = image[sourceOffset + x];
      }
    }
  }
  return { width: sheetWidth, height: sheetHeight, bytes };
}

function writePgm(filePath, width, height, bytes) {
  const header = Buffer.from(`P5\n${width} ${height}\n255\n`, "ascii");
  fs.writeFileSync(filePath, Buffer.concat([header, Buffer.from(bytes)]));
}

function escapeTsv(value) {
  return String(value ?? "")
    .replace(/\t/g, " ")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const rows = readSelectedRows(config);
  if (rows.length === 0) {
    throw new Error(`no slices matched --kinds ${config.kinds.join(",")}`);
  }

  fs.rmSync(config.outDir, { recursive: true, force: true });
  fs.mkdirSync(config.outDir, { recursive: true });
  const mapsDir = path.join(config.outDir, "maps");
  const previewDir = path.join(config.outDir, "preview");
  const strokesDir = path.join(config.outDir, "strokes");
  fs.mkdirSync(mapsDir, { recursive: true });
  fs.mkdirSync(previewDir, { recursive: true });
  fs.mkdirSync(strokesDir, { recursive: true });

  const ring = ringPrior(config.imageSize);
  const mapFiles = new Map();
  const previews = new Map();
  for (const grid of config.targetGrids) {
    const gridDir = path.join(mapsDir, `grid-${grid}`);
    fs.mkdirSync(gridDir, { recursive: true });
    for (const channel of mapChannels) {
      const filePath = path.join(gridDir, `${channel}.u8`);
      mapFiles.set(`${grid}:${channel}`, filePath);
      fs.writeFileSync(filePath, Buffer.alloc(0));
      previews.set(`${grid}:${channel}`, []);
    }
  }

  const strokeJsonlPath = path.join(strokesDir, "stroke-graphs.jsonl");
  const strokeJsonl = fs.createWriteStream(strokeJsonlPath, { encoding: "utf8" });
  const rowSummaries = [];

  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    const binary = thresholdInk(row.ink, config.inkThreshold);
    const skeleton = skeletonize(binary, config.imageSize);
    const stroke = binaryToInk(skeleton);
    const distance = distanceField(skeleton, config.imageSize, config.distanceRadius);
    const graph = strokeGraph(skeleton, config.imageSize, config.strokeEpsilon);

    for (const grid of config.targetGrids) {
      const channelImages = {
        density: downsampleAverage(row.ink, config.imageSize, grid),
        stroke: downsampleAverage(stroke, config.imageSize, grid),
        distance: downsampleAverage(distance, config.imageSize, grid),
        ring: downsampleAverage(ring, config.imageSize, grid),
      };
      for (const [channel, image] of Object.entries(channelImages)) {
        fs.appendFileSync(mapFiles.get(`${grid}:${channel}`), image);
        previews.get(`${grid}:${channel}`).push(image);
      }
    }

    strokeJsonl.write(
      `${JSON.stringify({
        schema: "nsrl.solomon_stroke_graph.v1",
        row_index: rowIndex,
        number: row.number,
        slice_id: row.sliceId,
        label: row.label,
        kind: row.kind,
        image_size: config.imageSize,
        ink_threshold: config.inkThreshold,
        stroke_pixels: graph.strokePixels,
        node_count: graph.nodeCount,
        path_count: graph.paths.length,
        simplified_point_count: graph.simplifiedPointCount,
        paths: graph.paths,
      })}\n`,
    );

    rowSummaries.push({
      rowIndex,
      number: row.number,
      sliceId: row.sliceId,
      label: row.label,
      kind: row.kind,
      sourceFile: row.sourceFile,
      inkRel: row.inkRel,
      meanInkQ8: Math.round(row.ink.reduce((sum, value) => sum + value, 0) / row.ink.length * 256),
      strokePixels: graph.strokePixels,
      nodeCount: graph.nodeCount,
      pathCount: graph.paths.length,
      simplifiedPointCount: graph.simplifiedPointCount,
    });
  }
  strokeJsonl.end();

  for (const grid of config.targetGrids) {
    for (const channel of mapChannels) {
      const images = previews.get(`${grid}:${channel}`);
      const sheet = makeContactSheet(images, grid, grid, config.previewColumns, 2);
      writePgm(path.join(previewDir, `grid-${grid}-${channel}.pgm`), sheet.width, sheet.height, sheet.bytes);
    }
  }

  const rowsTsvPath = path.join(config.outDir, "rows.tsv");
  const rowsTsv = [
    [
      "row_index",
      "number",
      "slice_id",
      "label",
      "kind",
      "source_file",
      "ink_u8",
      "mean_ink_q8",
      "stroke_pixels",
      "node_count",
      "path_count",
      "simplified_point_count",
    ].join("\t"),
    ...rowSummaries.map((row) =>
      [
        row.rowIndex,
        row.number ?? "",
        escapeTsv(row.sliceId),
        escapeTsv(row.label),
        escapeTsv(row.kind),
        escapeTsv(row.sourceFile),
        escapeTsv(row.inkRel),
        row.meanInkQ8,
        row.strokePixels,
        row.nodeCount,
        row.pathCount,
        row.simplifiedPointCount,
      ].join("\t"),
    ),
  ].join("\n");
  fs.writeFileSync(rowsTsvPath, `${rowsTsv}\n`, "utf8");

  const manifest = {
    schema: "nsrl.solomon_spatial_targets.v1",
    source_slices_manifest: config.slicesManifest,
    rows: rows.length,
    image_size: config.imageSize,
    target_grids: config.targetGrids,
    selected_kinds: config.kinds,
    channels: {
      density: "area-averaged raw u8 ink, no sharpening or thresholding",
      stroke: "Zhang-Suen thinned centerline map from thresholded source ink",
      distance: `linear falloff to nearest skeleton pixel within ${config.distanceRadius} source pixels`,
      ring: "normalized annulus prior centered in the source frame",
    },
    row_tsv: path.relative(config.outDir, rowsTsvPath),
    stroke_graph_jsonl: path.relative(config.outDir, strokeJsonlPath),
    maps: Object.fromEntries(
      config.targetGrids.map((grid) => [
        `grid_${grid}`,
        Object.fromEntries(
          mapChannels.map((channel) => [
            channel,
            {
              path: path.relative(config.outDir, mapFiles.get(`${grid}:${channel}`)),
              bytes_per_row: grid * grid,
              row_major: true,
            },
          ]),
        ),
      ]),
    ),
    preview_dir: path.relative(config.outDir, previewDir),
  };
  const manifestPath = path.join(config.outDir, "manifest.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(
    JSON.stringify({
      manifest: manifestPath,
      rows: rows.length,
      target_grids: config.targetGrids,
      stroke_graphs: strokeJsonlPath,
    }),
  );
}

main();
