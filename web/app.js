import init, { NsrlChat, SolomonSampler } from "./pkg/nsrl_web_wasm.js";

const ASSETS = {
  model: "./assets/model.nsrllm",
  vocab: "./assets/v4096.vocab.tsv",
  tokens: "./assets/v4096.tokens.u16",
  solomonModel: "./assets/solomon-model.nsrltch?v=4-seal-scaled",
  solomonTextIndex: "./assets/solomon-spirit-text-signatures.tsv?v=4-seal-scaled",
};
const SIGIL_CANDIDATES = 4;
const SIGIL_PASSES = 3;
const SIGIL_CONDITIONS = 24;
const ADAPT_MAX_WINDOWS = 48;

const DB_NAME = "nsrl-crowley-bard-aphorism-v2-decode-v2";
const DB_VERSION = 1;
const STORE_NAME = "session";

const state = {
  engine: null,
  transcript: "",
  turn: 0,
  lastMessage: "The aphorism model is stable in the browser.",
  lastAdaptWindows: 0,
  modelBytes: null,
  vocabText: "",
  tokenBytes: null,
  solomon: null,
  solomonLoad: null,
  sigilWorker: null,
  sigilJob: 0,
  pendingSigil: null,
  lastSigilMetadata: null,
  savedModelBytes: null,
};

const nodes = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#promptInput"),
  send: document.querySelector("#sendButton"),
  reset: document.querySelector("#resetButton"),
  oracle: document.querySelector("#oracleText"),
  sigil: document.querySelector("#sigilCanvas"),
  status: document.querySelector("#statusPill"),
  turn: document.querySelector("#turnPill"),
  adapt: document.querySelector("#adaptPill"),
  tokens: document.querySelector("#tokenSlider"),
  topK: document.querySelector("#topKSlider"),
};

boot();

nodes.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = nodes.input.value.trim();
  if (!message || !state.engine) {
    return;
  }

  state.turn += 1;
  state.transcript += `human: ${message}\n`;
  nodes.input.value = "";
  setBusy(true, "Thinking");
  nodes.oracle.textContent = message;

  await yieldFrame();

  try {
    const result = JSON.parse(
      state.engine.adapt_and_reply(
        state.transcript,
        message,
        Number(nodes.tokens.value),
        nextSeed(message),
        Number(nodes.topK.value),
        ADAPT_MAX_WINDOWS,
      ),
    );
    state.transcript += `model: ${result.text}\n`;
    state.lastMessage = result.text;
    state.lastAdaptWindows = result.fine_tune_windows;
    nodes.oracle.textContent = result.text;
    scheduleSolomonSigil(result.text, `model-${state.turn}`);
    nodes.turn.textContent = `Turn ${state.turn}`;
    nodes.adapt.textContent = `Adapt ${result.fine_tune_windows}`;
    nodes.status.textContent = "Saving";
    await saveLocalState({ persistModel: Boolean(result.adapted) });
    nodes.status.textContent = "Ready";
  } catch (error) {
    nodes.oracle.textContent = error instanceof Error ? error.message : String(error);
    nodes.status.textContent = "Error";
  } finally {
    setBusy(false);
  }
});

nodes.reset.addEventListener("click", async () => {
  if (!state.modelBytes || !state.tokenBytes) {
    return;
  }
  setBusy(true, "Resetting");
  await clearLocalState();
  state.engine = new NsrlChat(state.modelBytes, state.vocabText, state.tokenBytes);
  state.transcript = "";
  state.turn = 0;
  state.lastMessage = "The model is awake again.";
  state.lastAdaptWindows = 0;
  state.savedModelBytes = null;
  nodes.oracle.textContent = state.lastMessage;
  scheduleSolomonSigil(state.lastMessage, "reset");
  nodes.turn.textContent = "Turn 0";
  nodes.adapt.textContent = "Adapt 0";
  nodes.input.value = "";
  setBusy(false, "Ready");
  nodes.input.focus();
});

async function boot() {
  setBusy(true, "Loading");
  try {
    await init();
    state.sigilWorker = createSigilWorker();
    const [[modelBytes, vocabText, tokenBytes], fallbackSolomon] = await Promise.all([
      Promise.all([
        fetchBytes(ASSETS.model),
        fetchText(ASSETS.vocab),
        fetchBytes(ASSETS.tokens),
      ]),
      state.sigilWorker
        ? Promise.resolve(null)
        : Promise.all([fetchBytes(ASSETS.solomonModel), fetchText(ASSETS.solomonTextIndex)]),
    ]);
    state.modelBytes = modelBytes;
    state.vocabText = vocabText;
    state.tokenBytes = tokenBytes;
    if (fallbackSolomon) {
      state.solomon = new SolomonSampler(fallbackSolomon[0], fallbackSolomon[1]);
    }
    let saved = await loadLocalState();
    let savedModel = saved?.modelBytes instanceof Uint8Array ? saved.modelBytes : null;
    try {
      state.engine = new NsrlChat(savedModel || modelBytes, vocabText, tokenBytes);
    } catch (error) {
      await clearLocalState();
      saved = null;
      savedModel = null;
      state.engine = new NsrlChat(modelBytes, vocabText, tokenBytes);
    }
    JSON.parse(state.engine.model_card());
    if (saved) {
      state.transcript = saved.transcript || "";
      state.turn = Number(saved.turn || 0);
      state.lastMessage = saved.lastMessage || "The model remembers this browser.";
      state.lastAdaptWindows = Number(saved.lastAdaptWindows || 0);
    }
    state.savedModelBytes = savedModel;
    nodes.oracle.textContent = state.lastMessage;
    nodes.turn.textContent = `Turn ${state.turn}`;
    nodes.adapt.textContent = `Adapt ${state.lastAdaptWindows}`;
    nodes.status.textContent = saved ? "Restored" : "Ready";
    setBusy(false);
    scheduleSolomonSigil(state.lastMessage, saved ? "restored" : "boot");
    nodes.input.focus();
  } catch (error) {
    nodes.oracle.textContent = error instanceof Error ? error.message : String(error);
    nodes.status.textContent = "Error";
  }
}

function setBusy(isBusy, label = null) {
  nodes.input.disabled = isBusy;
  nodes.send.disabled = isBusy;
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

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url}`);
  }
  return response.text();
}

async function saveLocalState({ persistModel = false } = {}) {
  if (!state.engine) {
    return;
  }
  let modelBytes = state.savedModelBytes;
  if (persistModel) {
    modelBytes = state.engine.export_model();
    state.savedModelBytes = modelBytes;
  }
  try {
    await putSessionRecord({
      modelBytes,
      transcript: state.transcript,
      turn: state.turn,
      lastMessage: state.lastMessage,
      lastAdaptWindows: state.lastAdaptWindows,
      savedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.warn("Could not save local model state", error);
  }
}

async function loadLocalState() {
  try {
    return await getSessionRecord();
  } catch (error) {
    console.warn("Could not load local model state", error);
    return null;
  }
}

async function clearLocalState() {
  try {
    const db = await openDb();
    await requestToPromise(
      db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).delete("current"),
    );
    db.close();
  } catch (error) {
    console.warn("Could not clear local model state", error);
  }
}

async function putSessionRecord(record) {
  const db = await openDb();
  await requestToPromise(
    db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).put(record, "current"),
  );
  db.close();
}

async function getSessionRecord() {
  const db = await openDb();
  const value = await requestToPromise(
    db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get("current"),
  );
  db.close();
  return value || null;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open local model storage"));
  });
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Local storage request failed"));
  });
}

function nextSeed(message) {
  let hash = 2166136261;
  for (let index = 0; index < message.length; index += 1) {
    hash ^= message.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash ^ Date.now() ^ (state.turn * 2654435761)) >>> 0;
}

function scheduleSolomonSigil(text, salt = "") {
  const job = ++state.sigilJob;
  state.pendingSigil = { text, salt };
  if (state.sigilWorker) {
    state.sigilWorker.postMessage({
      type: "sample",
      id: job,
      text,
      salt,
      candidates: SIGIL_CANDIDATES,
      passes: SIGIL_PASSES,
      conditions: SIGIL_CONDITIONS,
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
    const worker = new Worker(new URL("./sigil-worker.js", import.meta.url), { type: "module" });
    worker.addEventListener("message", handleSigilWorkerMessage);
    worker.addEventListener("error", (event) => {
      console.warn("Solomon sigil worker failed", event.message || event);
      disableSigilWorker();
      const pending = state.pendingSigil;
      if (pending) {
        scheduleSolomonSigil(pending.text, pending.salt);
      }
    });
    return worker;
  } catch (error) {
    console.warn("Could not start Solomon sigil worker", error);
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
}

function disableSigilWorker() {
  state.sigilWorker?.terminate?.();
  state.sigilWorker = null;
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
    await ensureSolomonFallback();
    if (job === state.sigilJob) {
      renderSolomonSigil(text, salt);
    }
  } catch (error) {
    clearSolomonSigil(error);
  }
}

function renderSolomonSigil(text, salt = "") {
  const canvas = nodes.sigil;
  if (!canvas || !state.solomon) {
    return;
  }
  let sample = null;
  try {
    const seed = sigilSeed(text, salt);
    sample =
      typeof state.solomon.sample_fast === "function"
        ? state.solomon.sample_fast(
            text,
            seed,
            SIGIL_CANDIDATES,
            SIGIL_PASSES,
            SIGIL_CONDITIONS,
          )
        : state.solomon.sample(text, seed, SIGIL_CANDIDATES, SIGIL_PASSES);
    const width = sample.width();
    const height = sample.height();
    drawSolomonSigil(new Uint8ClampedArray(sample.rgba()), width, height);
    state.lastSigilMetadata = JSON.parse(sample.metadata_json());
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

function yieldFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
