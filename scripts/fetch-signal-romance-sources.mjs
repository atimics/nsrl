#!/usr/bin/env node
import childProcess from "node:child_process";
import fs from "node:fs";
import https from "node:https";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "data/processed/signal-romance-sources",
};

const gutenbergSources = [
  {
    label: "time-machine",
    url: "https://www.gutenberg.org/cache/epub/35/pg35.txt",
  },
  {
    label: "war-of-the-worlds",
    url: "https://www.gutenberg.org/cache/epub/36/pg36.txt",
  },
  {
    label: "princess-of-mars",
    url: "https://www.gutenberg.org/cache/epub/62/pg62.txt",
  },
  {
    label: "twenty-thousand-leagues",
    url: "https://www.gutenberg.org/cache/epub/164/pg164.txt",
  },
];

const publicSources = {
  apollo11AirToGroundHtml: "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11transcript_tec.html",
  faaRadioPhraseologyHtml: "https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap4_section_2.html",
  earhartRadioLogHtml: "https://www.archives.gov/college-park/highlights/earhart-log",
};

const nasaTranscriptPdfs = [
  {
    label: "apollo-07-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as07-tec.pdf",
  },
  {
    label: "apollo-08-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as08-tec.pdf",
  },
  {
    label: "apollo-10-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as10-tec.pdf",
  },
  {
    label: "apollo-11-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as11-tec.pdf",
  },
  {
    label: "apollo-12-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as12-tec.pdf",
  },
  {
    label: "apollo-13-technical",
    url: "https://www.nasa.gov/wp-content/uploads/2026/01/as13-tec.pdf",
  },
];

const deprecatedSourceFiles = [
  "trucker-cb-script.clean.txt",
  "wwii-radio-catalog.clean.txt",
  "fmcsa-cb-script.pdf",
];

function usage() {
  console.log(`Usage: node scripts/fetch-signal-romance-sources.mjs [options]

Options:
  --out-dir PATH    Output source directory [${defaults.outDir}]
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
    if (arg !== "--out-dir") {
      throw new Error(`unknown option: ${arg}`);
    }
    options.outDir = argv[++index];
    if (!options.outDir) {
      throw new Error("--out-dir requires a value");
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

function fetchText(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { "User-Agent": "nsrl-signal-romance-corpus/1.0" } }, (response) => {
      if ([301, 302, 303, 307, 308].includes(response.statusCode ?? 0)) {
        const location = response.headers.location;
        if (!location) {
          reject(new Error(`redirect without location: ${url}`));
          return;
        }
        resolve(fetchText(new URL(location, url).toString()));
        return;
      }
      if ((response.statusCode ?? 500) >= 400) {
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        return;
      }
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    }).on("error", reject);
  });
}

function fetchBuffer(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { "User-Agent": "nsrl-signal-romance-corpus/1.0" } }, (response) => {
      if ([301, 302, 303, 307, 308].includes(response.statusCode ?? 0)) {
        const location = response.headers.location;
        if (!location) {
          reject(new Error(`redirect without location: ${url}`));
          return;
        }
        resolve(fetchBuffer(new URL(location, url).toString()));
        return;
      }
      if ((response.statusCode ?? 500) >= 400) {
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        return;
      }
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve(Buffer.concat(chunks)));
    }).on("error", reject);
  });
}

function cleanAscii(text) {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[–—]/g, "-")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function stripGutenberg(text) {
  let stripped = text.replace(/\r/g, "");
  const start = stripped.match(/\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n/i);
  if (start) {
    stripped = stripped.slice((start.index ?? 0) + start[0].length);
  }
  const end = stripped.search(/\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK/i);
  if (end !== -1) {
    stripped = stripped.slice(0, end);
  }
  return cleanAscii(stripped);
}

function htmlToText(html) {
  return cleanAscii(html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(?:p|div|tr|h[1-6]|li|pre)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'"));
}

function radioProcedureSeed() {
  return cleanAscii(`
RADIO PROCEDURE SEED
Copy that.
Say again, last transmission.
Stand by.
Negative copy.
Affirmative, traffic in sight.
No joy.
Read back confirmed.
Hold this channel.
Convoy checks in by call sign.
Tower, request vector.
Flight, maintain heading and report visual.
Roger, holding pattern.
Mayday relay received.
Channel clear.
Over.
Out.
`);
}

function removeDeprecatedSources(outDir) {
  for (const file of deprecatedSourceFiles) {
    const filePath = path.join(outDir, file);
    if (fs.existsSync(filePath)) {
      fs.rmSync(filePath);
    }
  }
}

async function writeGutenberg(outDir, manifest) {
  const parts = [];
  for (const source of gutenbergSources) {
    const raw = await fetchText(source.url);
    const clean = stripGutenberg(raw);
    parts.push(`SOURCE ${source.label}\n${clean}\n`);
    manifest.sources.push({
      label: source.label,
      kind: "project_gutenberg_public_domain_sci_fi",
      url: source.url,
      bytes: Buffer.byteLength(clean),
    });
  }
  fs.writeFileSync(path.join(outDir, "old-sci-fi.clean.txt"), parts.join("\n\n"), "utf8");
}

async function writeWwiiMetadata(outDir, manifest) {
  const raw = await fetchText(publicSources.internetArchiveWwii1944Metadata);
  const metadata = JSON.parse(raw);
  const names = (metadata.files ?? [])
    .map((file) => file.name)
    .filter((name) => /\.(mp3|ogg|flac|wav)$/i.test(name))
    .slice(0, 160)
    .map((name) => name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "));
  const text = cleanAscii([
    "WWII RADIO PUBLIC DOMAIN CATALOG",
    metadata.metadata?.title ?? "WWII News and Related Sound files from 1944",
    metadata.metadata?.description ?? "",
    ...names,
  ].join("\n"));
  fs.writeFileSync(path.join(outDir, "wwii-radio-catalog.clean.txt"), text, "utf8");
  manifest.sources.push({
    label: "wwii-radio-catalog",
    kind: "internet_archive_public_domain_metadata",
    url: publicSources.internetArchiveWwii1944Metadata,
    bytes: Buffer.byteLength(text),
    note: "Catalog metadata and recording titles only; no audio transcription is inferred.",
  });
}

async function writeNasaApollo(outDir, manifest) {
  const html = await fetchText(publicSources.apollo11AirToGroundHtml);
  const text = htmlToText(html)
    .split(/\n/)
    .map((line) => line.trim())
    .filter((line) =>
      line &&
      !/^last revised/i.test(line) &&
      !/^journal text/i.test(line) &&
      !/^copyright/i.test(line)
    )
    .join("\n");
  fs.writeFileSync(path.join(outDir, "nasa-apollo-radio.clean.txt"), text, "utf8");
  manifest.sources.push({
    label: "nasa-apollo-radio",
    kind: "nasa_public_air_to_ground_transcript_html",
    url: publicSources.apollo11AirToGroundHtml,
    bytes: Buffer.byteLength(text),
  });
}

async function writeNasaTranscriptPdfs(outDir, manifest) {
  const parts = [];
  const pdfDir = path.join(outDir, "nasa-pdf-cache");
  fs.mkdirSync(pdfDir, { recursive: true });
  for (const source of nasaTranscriptPdfs) {
    const pdfPath = path.join(pdfDir, `${source.label}.pdf`);
    const pdf = await fetchBuffer(source.url);
    fs.writeFileSync(pdfPath, pdf);
    const pdftotext = childProcess.spawnSync("pdftotext", [pdfPath, "-"], {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
    if (pdftotext.status !== 0 || !pdftotext.stdout.trim()) {
      manifest.notes.push(`NASA PDF extraction skipped for ${source.label}`);
      continue;
    }
    const clean = cleanAscii(pdftotext.stdout);
    parts.push(`SOURCE ${source.label}\n${clean}\n`);
    manifest.sources.push({
      label: source.label,
      kind: "nasa_public_air_to_ground_transcript_pdf",
      url: source.url,
      bytes: Buffer.byteLength(clean),
    });
  }
  if (parts.length > 0) {
    fs.writeFileSync(path.join(outDir, "nasa-mission-transcripts.clean.txt"), parts.join("\n\n"), "utf8");
  }
}

async function writeFaaPhraseology(outDir, manifest) {
  const html = await fetchText(publicSources.faaRadioPhraseologyHtml);
  const text = htmlToText(html)
    .split(/\n/)
    .map((line) => line.trim())
    .filter((line) =>
      line &&
      !/^chapter /i.test(line) &&
      !/^section /i.test(line) &&
      !/^top$/i.test(line)
    )
    .join("\n");
  fs.writeFileSync(path.join(outDir, "faa-radio-phraseology.clean.txt"), text, "utf8");
  manifest.sources.push({
    label: "faa-radio-phraseology",
    kind: "faa_public_radio_phraseology_html",
    url: publicSources.faaRadioPhraseologyHtml,
    bytes: Buffer.byteLength(text),
  });
}

async function writeEarhartRadioLog(outDir, manifest) {
  const html = await fetchText(publicSources.earhartRadioLogHtml);
  const text = htmlToText(html);
  fs.writeFileSync(path.join(outDir, "earhart-radio-log.clean.txt"), text, "utf8");
  manifest.sources.push({
    label: "earhart-radio-log",
    kind: "national_archives_public_domain_radio_log_page",
    url: publicSources.earhartRadioLogHtml,
    bytes: Buffer.byteLength(text),
  });
}

async function writeCbScript(outDir, manifest) {
  const pdfPath = path.join(outDir, "fmcsa-cb-script.pdf");
  const txtPath = path.join(outDir, "trucker-cb-script.clean.txt");
  try {
    const pdf = await fetchBuffer(publicSources.fmcsaCbScriptPdf);
    fs.writeFileSync(pdfPath, pdf);

    const pdftotext = childProcess.spawnSync("pdftotext", [pdfPath, "-"], {
      encoding: "utf8",
      maxBuffer: 1024 * 1024,
    });
    if (pdftotext.status === 0 && pdftotext.stdout.trim()) {
      const clean = cleanAscii(pdftotext.stdout);
      fs.writeFileSync(txtPath, clean, "utf8");
      manifest.sources.push({
        label: "trucker-cb-script",
        kind: "us_government_pdf_extracted_text",
        url: publicSources.fmcsaCbScriptPdf,
        bytes: Buffer.byteLength(clean),
      });
      return;
    }
  } catch (error) {
    manifest.notes.push(`FMCSA CB PDF fetch skipped: ${error.message}`);
  }

  const fallback = cleanAscii(`
TRUCKER CB STYLE SEED
Driver, come back.
Where you headed, driver?
Load moving westbound.
Mile marker check.
Keep the lane clear.
Breaker, traffic ahead.
Copy that, big rig.
`);
  fs.writeFileSync(txtPath, fallback, "utf8");
  manifest.sources.push({
    label: "trucker-cb-script",
    kind: "fallback_cb_style_seed",
    url: publicSources.fmcsaCbScriptPdf,
    bytes: Buffer.byteLength(fallback),
    note: "pdftotext unavailable or extraction failed; wrote non-verbatim style seed.",
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });
  removeDeprecatedSources(outDir);
  const manifest = {
    schema: "nsrl.signal_romance_sources.v1",
    created_at: new Date().toISOString(),
    out_dir: outDir,
    sources: [],
    notes: [
      "ATCO2 is referenced as a future licensed ATC source, but not ingested by this script.",
      "Truck/CB and WWII catalog lanes were deliberately removed; NASA mission transcripts are the radio-procedure source of record.",
    ],
  };

  fs.writeFileSync(path.join(outDir, "radio-procedure.clean.txt"), radioProcedureSeed(), "utf8");
  manifest.sources.push({
    label: "radio-procedure",
    kind: "composed_nonverbatim_radio_phrase_seed",
    bytes: Buffer.byteLength(radioProcedureSeed()),
  });

  await writeGutenberg(outDir, manifest);
  await writeNasaApollo(outDir, manifest);
  await writeNasaTranscriptPdfs(outDir, manifest);
  await writeFaaPhraseology(outDir, manifest);
  await writeEarhartRadioLog(outDir, manifest);

  fs.writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`out_dir=${outDir}`);
  for (const source of manifest.sources) {
    console.log(`${source.label}\t${source.bytes}`);
  }
}

main().catch((error) => {
  console.error(`fetch-signal-romance-sources: ${error.message}`);
  process.exit(1);
});
