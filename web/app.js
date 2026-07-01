import init, { SolomonSampler } from "./pkg/nsrl_web_wasm.js";
import { SolomonAttentionSampler } from "./attention-sampler.js?v=3";
import { SolomonMultimodalSampler } from "./multimodal-sampler.js";

const ASSETS = {
  solomonAttentionModel: "./assets/solomon-attention.nsrllmm?v=5",
  solomonMultimodalModel: "./assets/solomon-multimodal.nsrlmod?v=1",
  solomonModel: "./assets/solomon-model.nsrltch?v=4-seal-scaled",
  solomonTextIndex: "./assets/solomon-spirit-text-signatures.tsv?v=4-seal-scaled",
};
const DEFAULT_PROMPT = "king solomon seal";

const state = {
  solomon: null,
  solomonLoad: null,
  attention: null,
  attentionLoad: null,
  multimodal: null,
  multimodalLoad: null,
  sigilWorker: null,
  sigilJob: 0,
  pendingSigil: null,
  lastSigilMetadata: null,
};

const nodes = {
  form: document.querySelector("#sampleForm"),
  input: document.querySelector("#promptInput"),
  sample: document.querySelector("#sampleButton"),
  reset: document.querySelector("#resetButton"),
  oracle: document.querySelector("#oracleText"),
  sigil: document.querySelector("#sigilCanvas"),
  status: document.querySelector("#statusPill"),
  target: document.querySelector("#targetPill"),
  score: document.querySelector("#scorePill"),
  candidates: document.querySelector("#candidateSlider"),
  passes: document.querySelector("#passSlider"),
  conditions: document.querySelector("#conditionSlider"),
};

boot();

nodes.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = nodes.input.value.trim();
  if (!prompt) {
    return;
  }
  nodes.oracle.textContent = prompt;
  scheduleSolomonSigil(prompt, `prompt-${Date.now()}`);
});

nodes.reset.addEventListener("click", () => {
  nodes.input.value = DEFAULT_PROMPT;
  nodes.oracle.textContent = DEFAULT_PROMPT;
  updateMetadata(null);
  scheduleSolomonSigil(DEFAULT_PROMPT, "reset");
  nodes.input.focus();
});

async function boot() {
  setBusy(true, "Loading");
  try {
    await init();
    state.sigilWorker = createSigilWorker();
    if (!state.sigilWorker) {
      await ensureMainThreadSampler();
    }
    nodes.input.value = DEFAULT_PROMPT;
    nodes.oracle.textContent = DEFAULT_PROMPT;
    scheduleSolomonSigil(DEFAULT_PROMPT, "boot");
    nodes.input.focus();
  } catch (error) {
    nodes.oracle.textContent = error instanceof Error ? error.message : String(error);
    setBusy(false, "Error");
  }
}

function setBusy(isBusy, label = null) {
  nodes.input.disabled = isBusy;
  nodes.sample.disabled = isBusy;
  if (label) {
    nodes.status.textContent = label;
  }
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

function scheduleSolomonSigil(text, salt = "") {
  const job = ++state.sigilJob;
  state.pendingSigil = { text, salt };
  setBusy(true, "Sampling");
  if (state.sigilWorker) {
    state.sigilWorker.postMessage({
      type: "sample",
      id: job,
      text,
      salt,
      candidates: Number(nodes.candidates.value),
      passes: Number(nodes.passes.value),
      conditions: Number(nodes.conditions.value),
    });
    return;
  }
  const run = () => {
    if (job === state.sigilJob) {
      renderSolomonSigilFallback(text, salt, job);
    }
  };
  if (typeof globalThis.requestIdleCallback === "function") {
    globalThis.requestIdleCallback(run, { timeout: 350 });
  } else {
    setTimeout(run, 0);
  }
}

function createSigilWorker() {
  if (typeof Worker !== "function") {
    return null;
  }
  try {
    const worker = new Worker(new URL("./sigil-worker.js?v=6", import.meta.url), {
      type: "module",
    });
    worker.addEventListener("message", handleSigilWorkerMessage);
    worker.addEventListener("error", (event) => {
      console.warn("Solomon model worker failed", event.message || event);
      disableSigilWorker();
      const pending = state.pendingSigil;
      if (pending) {
        scheduleSolomonSigil(pending.text, pending.salt);
      }
    });
    return worker;
  } catch (error) {
    console.warn("Could not start Solomon model worker", error);
    return null;
  }
}

function handleSigilWorkerMessage(event) {
  const message = event.data || {};
  if (message.type !== "sample" || message.id !== state.sigilJob) {
    return;
  }
  if (message.error) {
    console.warn("Could not render Solomon sigil", message.error);
    disableSigilWorker();
    const pending = state.pendingSigil;
    if (pending) {
      scheduleSolomonSigil(pending.text, pending.salt);
    }
    return;
  }
  drawSolomonSigil(new Uint8ClampedArray(message.rgba), message.width, message.height);
  state.lastSigilMetadata = message.metadata || null;
  if (message.generatedText) {
    nodes.oracle.textContent = message.generatedText;
  }
  updateMetadata(message.metadata || null);
  setBusy(false, "Ready");
}

function disableSigilWorker() {
  state.sigilWorker?.terminate?.();
  state.sigilWorker = null;
}

async function ensureMainThreadSampler() {
  await ensureAttentionFallback();
  if (state.attention) {
    return;
  }
  await ensureMultimodalFallback();
  if (state.multimodal) {
    return;
  }
  await ensureSolomonFallback();
}

async function ensureAttentionFallback() {
  if (state.attention) {
    return;
  }
  if (!state.attentionLoad) {
    state.attentionLoad = fetchOptionalBytes(ASSETS.solomonAttentionModel).then((modelBytes) => {
      if (!modelBytes) {
        return;
      }
      try {
        state.attention = new SolomonAttentionSampler(modelBytes);
      } catch (error) {
        console.warn("Could not load NSRLLMM1 attention model", error);
      }
    });
  }
  await state.attentionLoad;
}

async function ensureMultimodalFallback() {
  if (state.multimodal) {
    return;
  }
  if (!state.multimodalLoad) {
    state.multimodalLoad = fetchOptionalBytes(ASSETS.solomonMultimodalModel).then((modelBytes) => {
      if (modelBytes) {
        state.multimodal = new SolomonMultimodalSampler(modelBytes);
      }
    });
  }
  await state.multimodalLoad;
}

async function ensureSolomonFallback() {
  if (state.solomon) {
    return;
  }
  if (!state.solomonLoad) {
    state.solomonLoad = Promise.all([
      fetchBytes(ASSETS.solomonModel),
      fetchText(ASSETS.solomonTextIndex),
    ]).then(([modelBytes, textIndex]) => {
      state.solomon = new SolomonSampler(modelBytes, textIndex);
    });
  }
  await state.solomonLoad;
}

async function renderSolomonSigilFallback(text, salt, job) {
  try {
    await ensureMainThreadSampler();
    if (job === state.sigilJob) {
      renderSolomonSigil(text, salt);
    }
  } catch (error) {
    clearSolomonSigil(error);
  }
}

function renderSolomonSigil(text, salt = "") {
  const canvas = nodes.sigil;
  if (!canvas || (!state.solomon && !state.multimodal)) {
    return;
  }
  let sample = null;
  try {
    const seed = sigilSeed(text, salt);
    if (state.attention) {
      const result = state.attention.sample(text, {
        seed: hashText(seed),
        topK: Number(nodes.candidates.value),
      });
      drawSolomonSigil(result.rgba, result.width, result.height);
      state.lastSigilMetadata = result.metadata;
      nodes.oracle.textContent = result.text;
      updateMetadata(result.metadata);
      setBusy(false, "Ready");
      return;
    }
    if (state.multimodal) {
      const result = state.multimodal.sample(text, {
        seed: hashText(seed),
        topK: Number(nodes.candidates.value),
      });
      drawSolomonSigil(result.rgba, result.width, result.height);
      state.lastSigilMetadata = result.metadata;
      nodes.oracle.textContent = result.text;
      updateMetadata(result.metadata);
      setBusy(false, "Ready");
      return;
    }
    sample =
      typeof state.solomon.sample_fast === "function"
        ? state.solomon.sample_fast(
            text,
            seed,
            Number(nodes.candidates.value),
            Number(nodes.passes.value),
            Number(nodes.conditions.value),
          )
        : state.solomon.sample(
            text,
            seed,
            Number(nodes.candidates.value),
            Number(nodes.passes.value),
          );
    const width = sample.width();
    const height = sample.height();
    drawSolomonSigil(new Uint8ClampedArray(sample.rgba()), width, height);
    state.lastSigilMetadata = JSON.parse(sample.metadata_json());
    updateMetadata(state.lastSigilMetadata);
    setBusy(false, "Ready");
  } catch (error) {
    clearSolomonSigil(error);
  } finally {
    sample?.free?.();
  }
}

function drawSolomonSigil(rgba, width, height) {
  const canvas = nodes.sigil;
  const context = canvas?.getContext("2d");
  if (!canvas || !context) {
    return;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.putImageData(new ImageData(rgba, width, height), 0, 0);
}

function clearSolomonSigil(error) {
  const canvas = nodes.sigil;
  const context = canvas?.getContext("2d");
  console.warn("Could not render Solomon sigil", error);
  context?.clearRect(0, 0, canvas.width, canvas.height);
  state.lastSigilMetadata = null;
  updateMetadata(null);
  nodes.oracle.textContent = error instanceof Error ? error.message : String(error);
  setBusy(false, "Error");
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

function updateMetadata(metadata) {
  if (!metadata) {
    nodes.target.textContent = "Target -";
    nodes.score.textContent = "Score -";
    return;
  }
  const row = typeof metadata === "string" ? JSON.parse(metadata) : metadata;
  if (row.model_kind === "NSRLMOD1") {
    nodes.target.textContent = "Joint";
    nodes.score.textContent = row.model_hash || "NSRLMOD1";
    return;
  }
  if (row.model_kind === "NSRLLMM1") {
    const strictImageMemory = row.image_source === "embedded_image_memory_strict";
    nodes.target.textContent = "Attention";
    if (strictImageMemory && row.text_source === "embedded_text_lm_strict") {
      nodes.score.textContent = "Embedded LM text+image";
    } else if (strictImageMemory && row.text_source === "embedded_text_memory_guard") {
      nodes.score.textContent = "Guarded memory text+image";
    } else if (strictImageMemory && row.text_source === "embedded_text_memory_strict") {
      nodes.score.textContent = "Memory text+image";
    } else {
      nodes.score.textContent = row.text_source || row.model_hash || "NSRLLMM1";
    }
    return;
  }
  nodes.target.textContent = `Target ${row.target_number}`;
  nodes.score.textContent = `Score ${row.score}`;
}
