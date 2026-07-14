const nodes = {
  networkNotice: document.querySelector("#networkNotice"),
  publishedModels: document.querySelector("#publishedModels"),
  proofBlocks: document.querySelector("#proofBlocks"),
  mintedUnits: document.querySelector("#mintedUnits"),
  rewardSymbol: document.querySelector("#rewardSymbol"),
  achievedMetric: document.querySelector("#achievedMetric"),
  metricImprovement: document.querySelector("#metricImprovement"),
  baselineMetric: document.querySelector("#baselineMetric"),
  modelHash: document.querySelector("#modelHash"),
  datasetHash: document.querySelector("#datasetHash"),
  heldoutTargets: document.querySelector("#heldoutTargets"),
  settledBaseline: document.querySelector("#settledBaseline"),
  settledTarget: document.querySelector("#settledTarget"),
  settledAchieved: document.querySelector("#settledAchieved"),
  settledEscrow: document.querySelector("#settledEscrow"),
  guardrailList: document.querySelector("#guardrailList"),
  improvementRange: document.querySelector("#improvementRange"),
  improvementOutput: document.querySelector("#improvementOutput"),
  rewardRange: document.querySelector("#rewardRange"),
  rewardOutput: document.querySelector("#rewardOutput"),
  bonusRange: document.querySelector("#bonusRange"),
  bonusOutput: document.querySelector("#bonusOutput"),
  draftTarget: document.querySelector("#draftTarget"),
  draftBaseline: document.querySelector("#draftBaseline"),
  halfwayPayout: document.querySelector("#halfwayPayout"),
  targetPayout: document.querySelector("#targetPayout"),
  promotedPayout: document.querySelector("#promotedPayout"),
  downloadRecipe: document.querySelector("#downloadRecipe"),
  blockHeight: document.querySelector("#blockHeight"),
  blockHash: document.querySelector("#blockHash"),
  previousHash: document.querySelector("#previousHash"),
  evidenceHash: document.querySelector("#evidenceHash"),
  recipeHash: document.querySelector("#recipeHash"),
  blockReward: document.querySelector("#blockReward"),
  blockRewardSymbol: document.querySelector("#blockRewardSymbol"),
  allocationList: document.querySelector("#allocationList"),
  recipePreview: document.querySelector("#recipePreview"),
  copyRecipe: document.querySelector("#copyRecipe"),
  gapGrid: document.querySelector("#gapGrid"),
  localnetHeight: document.querySelector("#localnetHeight"),
  localnetAccounts: document.querySelector("#localnetAccounts"),
  localnetStageQuorum: document.querySelector("#localnetStageQuorum"),
  localnetCandidateQuorum: document.querySelector("#localnetCandidateQuorum"),
  localnetHead: document.querySelector("#localnetHead"),
  localnetLaunch: document.querySelector("#localnetLaunch"),
  localnetStages: document.querySelector("#localnetStages"),
  localnetChallenges: document.querySelector("#localnetChallenges"),
  localnetBounty: document.querySelector("#localnetBounty"),
  localnetBalances: document.querySelector("#localnetBalances"),
  transcriptFlow: document.querySelector("#transcriptFlow"),
  signedMark: document.querySelector(".signed-mark"),
  toast: document.querySelector("#toast"),
  targetBar: document.querySelector(".scale-target"),
  resultBar: document.querySelector(".scale-result"),
};

const allocationColors = ["#c9ff57", "#8c96ff", "#6de3c1", "#ff8b5d", "#f2f2e9"];
const guardrailOperators = {
  lt: "<",
  lte: "≤",
  eq: "=",
  gte: "≥",
  gt: ">",
};

let networkData = null;
let toastTimer = null;

boot();

async function boot() {
  const [networkResult, localnetResult] = await Promise.allSettled([
    fetchJson("./network.json"),
    fetchJson("./localnet.json"),
  ]);
  if (networkResult.status === "fulfilled") {
    networkData = networkResult.value;
    hydrate(networkData);
  } else {
    console.warn("Could not load Forge network data", networkResult.reason);
    showToast("The live specimen could not be loaded. Interactive draft controls still work locally.");
  }
  if (localnetResult.status === "fulfilled") {
    hydrateLocalnet(localnetResult.value);
  } else {
    console.warn("Could not load signed localnet transcript", localnetResult.reason);
  }

  nodes.improvementRange?.addEventListener("input", updateComposer);
  nodes.rewardRange?.addEventListener("input", updateComposer);
  nodes.bonusRange?.addEventListener("input", updateComposer);
  nodes.downloadRecipe?.addEventListener("click", downloadDraftRecipe);
  nodes.copyRecipe?.addEventListener("click", copySpecimenRecipe);
  document.querySelector("#composer")?.addEventListener("submit", (event) => event.preventDefault());
  updateComposer();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function hydrate(data) {
  const { network, launch, bounty, publication, gaps, recipe } = data;
  setText(nodes.networkNotice, network.notice);
  setText(nodes.publishedModels, formatInteger(network.published_models));
  setText(nodes.proofBlocks, formatInteger(network.proof_blocks));
  setText(nodes.mintedUnits, formatInteger(network.minted_units));
  setText(nodes.rewardSymbol, network.reward_symbol);

  setText(nodes.achievedMetric, formatInteger(bounty.achieved));
  setText(nodes.metricImprovement, `${formatBps(bounty.improvement_bps)}%`);
  setText(nodes.baselineMetric, formatInteger(bounty.baseline));
  setText(nodes.modelHash, shortHash(launch.model_hash));
  nodes.modelHash.title = launch.model_hash;
  setText(nodes.datasetHash, shortHash(launch.dataset_hash));
  nodes.datasetHash.title = launch.dataset_hash;
  setText(nodes.heldoutTargets, formatInteger(launch.targets));

  setText(nodes.settledBaseline, formatInteger(bounty.baseline));
  setText(nodes.settledTarget, formatInteger(bounty.target));
  setText(nodes.settledAchieved, formatInteger(bounty.achieved));
  setText(nodes.settledEscrow, formatInteger(bounty.settled_units));
  renderGuardrails(bounty.guardrails);

  const baseline = Number(bounty.baseline);
  const target = Number(bounty.target);
  const achieved = Number(bounty.achieved);
  if (baseline > 0 && nodes.targetBar && nodes.resultBar) {
    nodes.targetBar.style.width = `${Math.max(4, Math.min(100, (target / baseline) * 100))}%`;
    nodes.resultBar.style.width = `${Math.max(4, Math.min(100, (achieved / baseline) * 100))}%`;
  }

  const block = publication.block;
  setText(nodes.blockHeight, String(block.height).padStart(4, "0"));
  setText(nodes.blockHash, shortHash(block.block_sha256));
  nodes.blockHash.title = block.block_sha256;
  setText(nodes.previousHash, shortHash(block.previous_block_sha256));
  nodes.previousHash.title = block.previous_block_sha256;
  setText(nodes.evidenceHash, shortHash(block.evidence_sha256));
  nodes.evidenceHash.title = block.evidence_sha256;
  setText(nodes.recipeHash, shortHash(block.recipe_sha256));
  nodes.recipeHash.title = block.recipe_sha256;
  setText(nodes.blockReward, formatInteger(block.reward.minted_units));
  setText(nodes.blockRewardSymbol, block.reward.asset_symbol);
  renderAllocations(block.reward.allocations);
  renderGaps(gaps);
  nodes.recipePreview.textContent = JSON.stringify(recipe, null, 2);

  nodes.improvementRange.value = String(Number(data.next_bounty.improvement_bps) / 100);
  nodes.rewardRange.value = data.next_bounty.escrow_units;
  nodes.bonusRange.value = String(data.next_bounty.promotion_bonus_bps / 100);
  updateComposer();
}

function renderGuardrails(guardrails) {
  if (!nodes.guardrailList) {
    return;
  }
  nodes.guardrailList.replaceChildren(
    ...guardrails.map((guardrail) => {
      const item = document.createElement("li");
      item.textContent = `${humanize(guardrail.metric)} ${guardrailOperators[guardrail.operator] ?? guardrail.operator} ${formatInteger(guardrail.value)}`;
      return item;
    }),
  );
}

function renderAllocations(allocations) {
  if (!nodes.allocationList) {
    return;
  }
  nodes.allocationList.replaceChildren(
    ...allocations.map((allocation, index) => {
      const row = document.createElement("div");
      row.className = "allocation-row";
      const label = document.createElement("span");
      label.textContent = allocation.role;
      const track = document.createElement("div");
      track.className = "allocation-track";
      const fill = document.createElement("i");
      fill.style.width = `${allocation.basis_points / 100}%`;
      fill.style.setProperty("--allocation-color", allocationColors[index % allocationColors.length]);
      track.append(fill);
      const amount = document.createElement("strong");
      amount.textContent = formatInteger(allocation.units);
      amount.title = allocation.account;
      row.append(label, track, amount);
      return row;
    }),
  );
}

function renderGaps(gaps) {
  if (!nodes.gapGrid) {
    return;
  }
  nodes.gapGrid.replaceChildren(
    ...gaps.map((gap) => {
      const card = document.createElement("article");
      card.className = "gap-card";
      card.dataset.status = gap.status;
      const header = document.createElement("header");
      const title = document.createElement("h3");
      title.textContent = gap.capability;
      const status = document.createElement("span");
      status.className = "gap-status";
      status.textContent = gap.label;
      header.append(title, status);
      const detail = document.createElement("p");
      detail.textContent = gap.detail;
      card.append(header, detail);
      return card;
    }),
  );
}

function hydrateLocalnet(data) {
  const { summary, events } = data;
  const launch = summary.launches[0];
  const bountyRows = Object.values(summary.bounty_settlements[launch.id] ?? {});
  const funded = bountyRows.reduce((sum, bounty) => sum + BigInt(bounty.escrow_units), 0n);
  const settled = bountyRows.reduce((sum, bounty) => sum + BigInt(bounty.settled_units), 0n);
  const challengeEntries = Object.entries(summary.challenges);

  setText(nodes.localnetHeight, formatInteger(summary.height));
  setText(nodes.localnetAccounts, formatInteger(summary.accounts));
  setText(nodes.localnetStageQuorum, summary.network.stage_quorum);
  setText(nodes.localnetCandidateQuorum, summary.network.candidate_quorum);
  setText(nodes.localnetHead, shortHash(summary.head_sha256));
  nodes.localnetHead.title = summary.head_sha256;
  setText(nodes.localnetLaunch, launch.status);
  setText(nodes.localnetStages, `${launch.accepted_stages} accepted`);
  setText(
    nodes.localnetChallenges,
    challengeEntries.length
      ? challengeEntries.map(([status, count]) => `${count} ${status}`).join(" · ")
      : "none",
  );
  setText(nodes.localnetBounty, `${formatInteger(settled)} / ${formatInteger(funded)} settled`);
  setText(nodes.signedMark, `${events.length} / ${events.length} linked`);
  renderLocalnetBalances(summary.model_balances);
  renderTranscriptFlow(events, launch);
}

function renderLocalnetBalances(balances) {
  if (!nodes.localnetBalances) {
    return;
  }
  const [symbol, accounts] = Object.entries(balances)[0] ?? ["credits", {}];
  nodes.localnetBalances.previousElementSibling.textContent = `model-local balances · ${symbol}`;
  nodes.localnetBalances.replaceChildren(
    ...Object.entries(accounts).map(([account, units]) => {
      const row = document.createElement("div");
      const label = document.createElement("span");
      label.textContent = account.split(":").at(-1).replaceAll("-", " ");
      label.title = account;
      const amount = document.createElement("strong");
      amount.textContent = formatInteger(units);
      row.append(label, amount);
      return row;
    }),
  );
}

function renderTranscriptFlow(events, launch) {
  if (!nodes.transcriptFlow) {
    return;
  }
  const count = (eventType) =>
    events.filter((event) => event.signed_intent.event_type === eventType).length;
  const bounty = Object.values(
    networkData?.bounty ? { primary: networkData.bounty } : {},
  )[0];
  const rows = [
    ["Launch published", "signed"],
    ["Bounty funded", bounty ? formatInteger(bounty.escrow_units) : `${count("bounty_funded")} escrow`],
    ["Compute stages", `${launch.accepted_stages} / ${launch.required_stages}`],
    ["Validator attestations", formatInteger(count("validation_attested"))],
    ["Challenge resolved", count("challenge_resolved") ? "rejected" : "none"],
    ["Model published", launch.status],
  ];
  nodes.transcriptFlow.replaceChildren(
    ...rows.map(([labelText, valueText]) => {
      const item = document.createElement("li");
      const marker = document.createElement("i");
      const label = document.createElement("span");
      label.textContent = labelText;
      const value = document.createElement("strong");
      value.textContent = valueText;
      item.append(marker, label, value);
      return item;
    }),
  );
}

function updateComposer() {
  const improvementPercent = Number(nodes.improvementRange?.value ?? 10);
  const bountyUnits = BigInt(nodes.rewardRange?.value ?? 120000);
  const bonusPercent = Number(nodes.bonusRange?.value ?? 25);
  const baseline = BigInt(networkData?.next_bounty?.baseline ?? 260536589);
  const target = (baseline * BigInt(100 - improvementPercent)) / 100n;
  const progressPool = (bountyUnits * BigInt(100 - bonusPercent)) / 100n;
  const halfway = progressPool / 2n;

  setText(nodes.improvementOutput, `${improvementPercent}%`);
  setText(nodes.rewardOutput, `${formatInteger(bountyUnits)} credits`);
  setText(nodes.bonusOutput, `${bonusPercent}%`);
  setText(nodes.draftTarget, formatInteger(target));
  setText(nodes.draftBaseline, formatInteger(baseline));
  setText(nodes.halfwayPayout, formatInteger(halfway));
  setText(nodes.targetPayout, formatInteger(progressPool));
  setText(nodes.promotedPayout, formatInteger(bountyUnits));
}

function draftRecipe() {
  if (!networkData?.recipe) {
    return null;
  }
  const recipe = structuredClone(networkData.recipe);
  const improvementPercent = Number(nodes.improvementRange.value);
  const baseline = BigInt(networkData.next_bounty.baseline);
  recipe.launch.id = `integer-transformer-frontier-${improvementPercent}-draft`;
  recipe.launch.title = `Integer Transformer ${improvementPercent}% Frontier`;
  recipe.launch.summary = `Reduce probability_error_q15 by ${improvementPercent}% from the promoted NSRLMT5 baseline without regressing frozen health gates.`;
  recipe.launch.status = "draft";
  recipe.launch.published_at = new Date().toISOString();
  recipe.model.parent_model_hash = networkData.launch.model_hash;
  recipe.bounties[0].id = `reduce-error-${improvementPercent}-percent`;
  recipe.bounties[0].baseline = baseline.toString();
  recipe.bounties[0].target = ((baseline * BigInt(100 - improvementPercent)) / 100n).toString();
  recipe.bounties[0].escrow_units = nodes.rewardRange.value;
  recipe.bounties[0].payout.promotion_bonus_bps = Number(nodes.bonusRange.value) * 100;
  recipe.publication = null;
  return recipe;
}

function downloadDraftRecipe() {
  const recipe = draftRecipe();
  if (!recipe) {
    showToast("Network specimen is still loading.");
    return;
  }
  const blob = new Blob([`${JSON.stringify(recipe, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${recipe.launch.id}.launch.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  showToast("Draft model recipe created.");
}

async function copySpecimenRecipe() {
  if (!networkData?.recipe) {
    showToast("Network specimen is still loading.");
    return;
  }
  try {
    await navigator.clipboard.writeText(`${JSON.stringify(networkData.recipe, null, 2)}\n`);
    showToast("Specimen recipe copied.");
  } catch (error) {
    console.warn("Could not copy recipe", error);
    showToast("Copy is unavailable in this browser context.");
  }
}

function showToast(message) {
  if (!nodes.toast) {
    return;
  }
  nodes.toast.textContent = message;
  nodes.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => nodes.toast.classList.remove("visible"), 3200);
}

function setText(node, value) {
  if (node) {
    node.textContent = String(value);
  }
}

function formatInteger(value) {
  return BigInt(value).toLocaleString("en-US");
}

function formatBps(value) {
  const basisPoints = Number(value);
  return (basisPoints / 100).toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function shortHash(value) {
  const text = String(value);
  if (text.length <= 18) {
    return text;
  }
  return `${text.slice(0, 10)}…${text.slice(-7)}`;
}

function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
