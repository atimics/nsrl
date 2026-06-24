import init, { SolomonSampler } from "./pkg/nsrl_web_wasm.js";

const ASSETS = {
  solomonModel: "./assets/solomon-model.nsrltch?v=2-16ch",
  solomonTextIndex: "./assets/solomon-spirit-text-signatures.tsv?v=2-16ch",
};

let samplerPromise = null;
let latestJob = 0;

self.addEventListener("message", (event) => {
  const message = event.data || {};
  if (message.type !== "sample") {
    return;
  }
  latestJob = Math.max(latestJob, Number(message.id || 0));
  sampleSigil(message).catch((error) => {
    self.postMessage({
      type: "sample",
      id: message.id,
      error: error instanceof Error ? error.message : String(error),
    });
  });
});

async function sampleSigil(message) {
  const job = Number(message.id || 0);
  const sampler = await getSampler();
  if (job !== latestJob) {
    return;
  }

  let sample = null;
  try {
    const text = String(message.text || "");
    const seed = sigilSeed(text, String(message.salt || ""));
    sample = sampler.sample_fast(
      text,
      seed,
      Number(message.candidates || 4),
      Number(message.passes || 3),
      Number(message.conditions || 24),
    );
    if (job !== latestJob) {
      return;
    }
    const rgba = sample.rgba();
    self.postMessage(
      {
        type: "sample",
        id: job,
        width: sample.width(),
        height: sample.height(),
        rgba: rgba.buffer,
        metadata: JSON.parse(sample.metadata_json()),
      },
      [rgba.buffer],
    );
  } finally {
    sample?.free?.();
  }
}

async function getSampler() {
  if (!samplerPromise) {
    samplerPromise = (async () => {
      await init();
      const [modelBytes, textIndex] = await Promise.all([
        fetchBytes(ASSETS.solomonModel),
        fetchText(ASSETS.solomonTextIndex),
      ]);
      return new SolomonSampler(modelBytes, textIndex);
    })();
  }
  return samplerPromise;
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url}`);
  }
  return response.text();
}

function sigilSeed(text, salt) {
  return `web-${salt}-${hashText(text).toString(16)}`;
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
