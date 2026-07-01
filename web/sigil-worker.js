import init, { SolomonSampler } from "./pkg/nsrl_web_wasm.js";
import { SolomonAttentionSampler } from "./attention-sampler.js?v=3";
import { SolomonMultimodalSampler } from "./multimodal-sampler.js";

const ASSETS = {
  solomonAttentionModel: "./assets/solomon-attention.nsrllmm?v=5",
  solomonMultimodalModel: "./assets/solomon-multimodal.nsrlmod?v=2",
  solomonModel: "./assets/solomon-model.nsrltch?v=4-seal-scaled",
  solomonTextIndex: "./assets/solomon-spirit-text-signatures.tsv?v=4-seal-scaled",
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
  const loaded = await getSampler();
  if (job !== latestJob) {
    return;
  }

  let sample = null;
  try {
    const text = String(message.text || "");
    const seed = sigilSeed(text, String(message.salt || ""));
    if (loaded.kind === "attention") {
      const result = loaded.sampler.sample(text, {
        seed: hashText(seed),
        topK: Math.max(1, Number(message.candidates || 4)),
      });
      if (job !== latestJob) {
        return;
      }
      self.postMessage(
        {
          type: "sample",
          id: job,
          width: result.width,
          height: result.height,
          rgba: result.rgba.buffer,
          metadata: result.metadata,
          generatedText: result.text,
        },
        [result.rgba.buffer],
      );
      return;
    }
    if (loaded.kind === "multimodal") {
      const result = loaded.sampler.sample(text, {
        seed: hashText(seed),
        topK: Math.max(1, Number(message.candidates || 4)),
      });
      if (job !== latestJob) {
        return;
      }
      self.postMessage(
        {
          type: "sample",
          id: job,
          width: result.width,
          height: result.height,
          rgba: result.rgba.buffer,
          metadata: result.metadata,
          generatedText: result.text,
        },
        [result.rgba.buffer],
      );
      return;
    }
    sample = loaded.sampler.sample_fast(
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
      const attentionBytes = await fetchOptionalBytes(ASSETS.solomonAttentionModel);
      if (attentionBytes) {
        try {
          return {
            kind: "attention",
            sampler: new SolomonAttentionSampler(attentionBytes),
          };
        } catch (error) {
          console.warn("Could not load NSRLLMM1 attention model", error);
        }
      }
      const multimodalBytes = await fetchOptionalBytes(ASSETS.solomonMultimodalModel);
      if (multimodalBytes) {
        return {
          kind: "multimodal",
          sampler: new SolomonMultimodalSampler(multimodalBytes),
        };
      }
      await init();
      const [modelBytes, textIndex] = await Promise.all([
        fetchBytes(ASSETS.solomonModel),
        fetchText(ASSETS.solomonTextIndex),
      ]);
      return {
        kind: "denoiser",
        sampler: new SolomonSampler(modelBytes, textIndex),
      };
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

async function fetchOptionalBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    return null;
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
