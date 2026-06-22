#!/usr/bin/env node
// Fetch and clean public-domain source texts for the Crowley Bard lane.
//
// Crowley Bard is a Shakespeare x Blake x Crowley voice. The two small lanes
// (Blake, Crowley) are the bottleneck: too few unique bytes forces the corpus
// builder to loop them, which encourages memorization. This script expands the
// cleaned library from public-domain sources so byte budgets become honest
// subsampling instead of repetition.
//
// Sources and licensing (verified by header / API at fetch time):
//   - Blake (d. 1827): public domain worldwide. Project Gutenberg.
//   - Milton / Dante (adjacent visionary canon): public domain. Project
//     Gutenberg. Tagged "adjacent" because they shift the blend; kept in
//     separate files so the pure three-way blend stays a choice.
//   - Crowley (d. 1947): not on Project Gutenberg beyond Tannhauser / Household
//     Gods (already cleaned). His pre-1930 works are public domain (US: pub.
//     <= 1930; life+70 jurisdictions: all works since 2018) and are on English
//     Wikisource, fetched here as plain text via the MediaWiki extracts API.
//
// This script does no model training and posts nothing. Output is local only
// (data/ is gitignored).

import https from "node:https";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const outDir = process.env.OUT_DIR || path.join(repoRoot, "data/processed/crowley-bard-sources");
const rawDir = path.join(outDir, "raw");
const includeAdjacent = process.env.INCLUDE_ADJACENT !== "0"; // Milton/Dante
const userAgent =
  "nsrl-crowley-bard-corpus/1.0 (local research; contact: repo maintainer)";

fs.mkdirSync(rawDir, { recursive: true });

// Project Gutenberg works, verified by reading each file's "Title:"/"Author:"
// header before trusting the id.
const gutenbergSources = [
  { id: 1934, label: "blake-songs", title: "Songs of Innocence and of Experience", author: "William Blake", tier: "core" },
  { id: 574, label: "blake-poems-yeats", title: "Poems of William Blake", author: "William Blake", tier: "core" },
  { id: 26, label: "milton-paradise-lost", title: "Paradise Lost", author: "John Milton", tier: "adjacent" },
  { id: 8800, label: "dante-divine-comedy", title: "The Divine Comedy", author: "Dante Alighieri", tier: "adjacent" },
];

const wikisourceAuthor = "Author:Aleister_Crowley";
const wikisourceApi = "https://en.wikisource.org/w/api.php";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fetchOnce(url, json) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      { headers: { "User-Agent": userAgent, Accept: json ? "application/json" : "text/plain" } },
      (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          const next = new URL(res.headers.location, url).toString();
          resolve(fetchOnce(next, json));
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          const err = new Error(`HTTP ${res.statusCode} for ${url}`);
          err.statusCode = res.statusCode;
          err.retryAfter = Number(res.headers["retry-after"]) || 0;
          reject(err);
          return;
        }
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          resolve(json ? JSON.parse(body) : body);
        });
      },
    );
    req.on("error", reject);
    req.setTimeout(30000, () => req.destroy(new Error(`timeout for ${url}`)));
  });
}

async function fetchText(url, { json = false, attempts = 5 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await fetchOnce(url, json);
    } catch (err) {
      lastErr = err;
      const retryable = err.statusCode === 429 || (err.statusCode >= 500 && err.statusCode < 600) || /timeout/.test(err.message);
      if (!retryable || i === attempts - 1) throw err;
      const wait = Math.max((err.retryAfter || 0) * 1000, 800 * 2 ** i);
      console.warn(`  retry in ${wait}ms (${err.message})`);
      await sleep(wait);
    }
  }
  throw lastErr;
}

function stripGutenberg(text) {
  const startRe = /\*\*\* *START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*/i;
  const endRe = /\*\*\* *END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*/i;
  const startMatch = text.match(startRe);
  const endMatch = text.match(endRe);
  let body = text;
  if (startMatch) body = body.slice(startMatch.index + startMatch[0].length);
  if (endMatch) {
    const endIdx = body.search(endRe);
    if (endIdx !== -1) body = body.slice(0, endIdx);
  }
  return cleanLines(body);
}

function stripWikisource(text) {
  // explaintext extracts use "= Heading =" / "==== Chapter ====" markers and a
  // trailing reference/notes section. Drop heading lines and the tail apparatus.
  const lines = text.split("\n").filter((line) => {
    const t = line.trim();
    if (/^=+\s.*\s=+$/.test(t)) return false; // section headings
    return true;
  });
  let body = lines.join("\n");
  body = body.replace(/\n(References|Notes|Footnotes|External links)\b[\s\S]*$/i, "\n");
  return cleanLines(body);
}

function cleanLines(body) {
  const dropRe =
    /(project gutenberg|www\.gutenberg|gutenberg\.org|produced by|transcriber|this ebook is for the use)/i;
  // Editorial inserts found in some Gutenberg scans, e.g. "[Picture: ...]".
  const editorialRe = /\[(?:Picture|Illustration|Image|Plate|Footnote)[^\]]*\]/gi;
  const kept = body
    .replace(editorialRe, "")
    .split("\n")
    .filter((line) => !dropRe.test(line))
    .join("\n");
  return kept.replace(/\r/g, "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
}

function writeClean(label, title, author, tier, source, cleaned, rawUrl) {
  const cleanPath = path.join(outDir, `${label}.clean.txt`);
  fs.writeFileSync(cleanPath, cleaned);
  const bytes = Buffer.byteLength(cleaned, "utf8");
  console.log(`  ${label.padEnd(26)} ${String(bytes).padStart(8)}B  ${title}`);
  return { label, title, author, tier, source, raw_url: rawUrl, clean_path: path.relative(repoRoot, cleanPath), clean_bytes: bytes };
}

async function fetchGutenberg(records) {
  console.log("Project Gutenberg:");
  for (const s of gutenbergSources) {
    if (s.tier === "adjacent" && !includeAdjacent) continue;
    const url = `https://www.gutenberg.org/cache/epub/${s.id}/pg${s.id}.txt`;
    const raw = await fetchText(url);
    fs.writeFileSync(path.join(rawDir, `gutenberg-${s.id}.txt`), raw);
    // Verify the file is the work we think it is before keeping it.
    const titleLine = (raw.match(/^Title:\s*(.+)$/im) || [])[1] || "";
    if (s.title && titleLine && !titleLine.toLowerCase().includes(s.title.toLowerCase().slice(0, 12))) {
      console.warn(`  WARNING id=${s.id} header title "${titleLine}" != expected "${s.title}" — skipping`);
      continue;
    }
    records.push(writeClean(s.label, s.title, s.author, s.tier, `gutenberg:${s.id}`, stripGutenberg(raw), url));
  }
}

async function listSubpages(title) {
  const url = `${wikisourceApi}?action=query&list=allpages&apprefix=${encodeURIComponent(title + "/")}&apnamespace=0&aplimit=500&format=json`;
  const j = await fetchText(url, { json: true });
  return (j.query?.allpages || []).map((p) => p.title).sort();
}

async function extract(title) {
  const url = `${wikisourceApi}?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=${encodeURIComponent(title)}`;
  const j = await fetchText(url, { json: true });
  const pages = j.query?.pages || {};
  const page = Object.values(pages)[0];
  return page?.extract || "";
}

// The plaintext extracts API returns empty for works whose body is transcluded
// from the ProofreadPage (Page:) namespace, e.g. "Clouds without Water". For
// those we render the page (action=parse) and strip HTML + the running header.
async function parseText(title) {
  const url = `${wikisourceApi}?action=parse&prop=text&format=json&redirects=1&disabletoc=1&page=${encodeURIComponent(title)}`;
  let j;
  try {
    j = await fetchText(url, { json: true });
  } catch {
    return "";
  }
  if (j.error) return "";
  return j.parse?.text?.["*"] || "";
}

function decodeEntities(s) {
  return s
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&nbsp;/g, " ")
    .replace(/&mdash;/g, "—")
    .replace(/&ndash;/g, "–")
    .replace(/&quot;/g, '"')
    .replace(/&(?:#39|apos);/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function stripRenderedHtml(htmlText, workTitle) {
  let t = htmlText
    .replace(/<(style|script|sup|table)\b[\s\S]*?<\/\1>/gi, "")
    .replace(/<[^>]+>/g, " ");
  t = decodeEntities(t).replace(/​/g, "").replace(/ /g, " ");
  const yearRe = /\b(?:1[6-9]\d\d|20\d\d)\b/;
  const lines = t
    .split("\n")
    .map((l) => l.trim())
    .filter((s) => {
      if (!s) return true;
      if (s.includes("←") || s.includes("→")) return false; // nav arrows
      if (/^\d+$/.test(s)) return false; // bare page ids
      if (workTitle && s.includes(workTitle) && yearRe.test(s)) return false; // running header
      return true;
    })
    .map((s) => s.replace(/^\d{5,}\s+/, "").replace(/[ \t]+/g, " "));
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

// One page of a work: prefer the clean plaintext extract; fall back to rendered
// parse output for ProofreadPage-transcluded pages.
async function pageText(title, workTitle, throttleMs) {
  const ex = await extract(title);
  if (Buffer.byteLength(ex, "utf8") > 200) return ex;
  await sleep(throttleMs);
  const rendered = await parseText(title);
  return rendered ? stripRenderedHtml(rendered, workTitle) : ex;
}

async function fetchCrowley(records) {
  console.log("Wikisource (Aleister Crowley, pre-1930 public domain):");
  const authorJson = await fetchText(
    `${wikisourceApi}?action=parse&page=${wikisourceAuthor}&prop=links&format=json`,
    { json: true },
  );
  const works = (authorJson.parse?.links || [])
    .filter((l) => l.ns === 0)
    .map((l) => l["*"])
    .sort();

  const throttleMs = Number(process.env.WIKISOURCE_THROTTLE_MS || 1500);
  for (const title of works) {
    // Decide once per work whether its body is transcluded (extracts empty);
    // if so, fetch subpages with parse only and skip the doomed extract calls.
    const mainExtract = await extract(title);
    await sleep(throttleMs);
    const transcluded = Buffer.byteLength(mainExtract, "utf8") <= 200;
    let text = mainExtract;
    if (transcluded) {
      const rendered = await parseText(title);
      await sleep(throttleMs);
      if (rendered) text = stripRenderedHtml(rendered, title);
    }
    const subpages = await listSubpages(title);
    await sleep(throttleMs);
    if (subpages.length > 0) {
      const parts = [];
      for (const sub of subpages) {
        if (transcluded) {
          const rendered = await parseText(sub);
          parts.push(rendered ? stripRenderedHtml(rendered, title) : "");
        } else {
          parts.push(await pageText(sub, title, throttleMs));
        }
        await sleep(throttleMs);
      }
      const joined = parts.filter(Boolean).join("\n\n");
      // For transcluded works the subpages are the content; prefer the longer body.
      if (joined.length > text.length) text = joined;
    }
    const cleaned = stripWikisource(text);
    if (Buffer.byteLength(cleaned, "utf8") < 200) {
      console.warn(`  skip "${title}" (too short / not transcribed on Wikisource)`);
      continue;
    }
    const label = "crowley-" + title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    // Save a heading so RECLEAN_FROM_CACHE can recover the work title.
    const rawToSave = /^\s*=/.test(text) ? text : `= ${title} =\n\n${text}`;
    fs.writeFileSync(path.join(rawDir, `wikisource-${label}.txt`), rawToSave);
    records.push(
      writeClean(label, title, "Aleister Crowley", "core", `wikisource:${title}`, cleaned, `https://en.wikisource.org/wiki/${encodeURIComponent(title)}`),
    );
  }
}

function recleanFromCache(records) {
  console.log("Re-clean from cached raw/ (no network):");
  const files = fs.readdirSync(rawDir).filter((f) => f.endsWith(".txt")).sort();
  for (const file of files) {
    const raw = fs.readFileSync(path.join(rawDir, file), "utf8");
    const gut = file.match(/^gutenberg-(\d+)\.txt$/);
    if (gut) {
      const s = gutenbergSources.find((g) => String(g.id) === gut[1]);
      if (!s) continue;
      if (s.tier === "adjacent" && !includeAdjacent) continue;
      records.push(writeClean(s.label, s.title, s.author, s.tier, `gutenberg:${s.id}`, stripGutenberg(raw), `https://www.gutenberg.org/cache/epub/${s.id}/pg${s.id}.txt`));
      continue;
    }
    const ws = file.match(/^wikisource-(.+)\.txt$/);
    if (ws) {
      const label = ws[1];
      const title = ((raw.match(/^=+\s*(.+?)\s*=+\s*$/m) || [])[1] || label).trim();
      records.push(writeClean(label, title, "Aleister Crowley", "core", `wikisource:${title}`, stripWikisource(raw), `https://en.wikisource.org/wiki/${encodeURIComponent(title)}`));
    }
  }
}

async function main() {
  const records = [];
  if (process.env.RECLEAN_FROM_CACHE === "1") {
    recleanFromCache(records);
  } else {
    await fetchGutenberg(records);
    await fetchCrowley(records);
  }

  const byAuthor = {};
  for (const r of records) {
    byAuthor[r.author] = (byAuthor[r.author] || 0) + r.clean_bytes;
  }
  const manifest = {
    schema: "nsrl.crowley_bard_sources.v1",
    out_dir: path.relative(repoRoot, outDir),
    include_adjacent: includeAdjacent,
    generated_at: new Date().toISOString(),
    total_clean_bytes: records.reduce((a, r) => a + r.clean_bytes, 0),
    clean_bytes_by_author: byAuthor,
    license_note:
      "Blake/Milton/Dante: public domain (author d. pre-1900). Crowley: pre-1930 works, public domain in the US by publication and worldwide under life+70 since 2018.",
    sources: records,
  };
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log("\nClean bytes by author:");
  for (const [a, b] of Object.entries(byAuthor)) console.log(`  ${a.padEnd(20)} ${String(b).padStart(9)}B`);
  console.log(`\nmanifest: ${path.relative(repoRoot, path.join(outDir, "manifest.json"))}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
