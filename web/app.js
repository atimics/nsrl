import init, { NsrlChat } from "./pkg/nsrl_web_wasm.js";

const ASSETS = {
  model: "./assets/model.nsrllm",
  vocab: "./assets/v4096.vocab.tsv",
  tokens: "./assets/v4096.tokens.u16",
};

const DB_NAME = "nsrl-crowley-bard";
const DB_VERSION = 1;
const STORE_NAME = "session";

const state = {
  engine: null,
  transcript: "",
  turn: 0,
  lastMessage: "The model is stable in the browser.",
  lastAdaptWindows: 0,
  modelBytes: null,
  vocabText: "",
  tokenBytes: null,
};

const nodes = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#promptInput"),
  send: document.querySelector("#sendButton"),
  reset: document.querySelector("#resetButton"),
  oracle: document.querySelector("#oracleText"),
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
  setBusy(true, "Fine-tuning");
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
        0,
      ),
    );
    state.transcript += `model: ${result.text}\n`;
    state.lastMessage = result.text;
    state.lastAdaptWindows = result.fine_tune_windows;
    nodes.oracle.textContent = result.text;
    nodes.turn.textContent = `Turn ${state.turn}`;
    nodes.adapt.textContent = `Adapt ${result.fine_tune_windows}`;
    nodes.status.textContent = "Saving";
    await saveLocalState();
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
  nodes.oracle.textContent = state.lastMessage;
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
    const [modelBytes, vocabText, tokenBytes] = await Promise.all([
      fetchBytes(ASSETS.model),
      fetchText(ASSETS.vocab),
      fetchBytes(ASSETS.tokens),
    ]);
    state.modelBytes = modelBytes;
    state.vocabText = vocabText;
    state.tokenBytes = tokenBytes;
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
    if (savedModel) {
      state.transcript = saved.transcript || "";
      state.turn = Number(saved.turn || 0);
      state.lastMessage = saved.lastMessage || "The model remembers this browser.";
      state.lastAdaptWindows = Number(saved.lastAdaptWindows || 0);
    }
    nodes.oracle.textContent = state.lastMessage;
    nodes.turn.textContent = `Turn ${state.turn}`;
    nodes.adapt.textContent = `Adapt ${state.lastAdaptWindows}`;
    nodes.status.textContent = savedModel ? "Restored" : "Ready";
    setBusy(false);
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

async function saveLocalState() {
  if (!state.engine) {
    return;
  }
  const modelBytes = state.engine.export_model();
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

function yieldFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
