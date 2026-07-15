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
  marketBidCount: document.querySelector("#marketBidCount"),
  marketPaidStages: document.querySelector("#marketPaidStages"),
  marketPayments: document.querySelector("#marketPayments"),
  marketRefund: document.querySelector("#marketRefund"),
  marketAuctionList: document.querySelector("#marketAuctionList"),
  marketConservation: document.querySelector("#marketConservation"),
  marketFunded: document.querySelector("#marketFunded"),
  marketPaid: document.querySelector("#marketPaid"),
  marketRefundDetail: document.querySelector("#marketRefundDetail"),
  marketBountyPaid: document.querySelector("#marketBountyPaid"),
  marketPaidBar: document.querySelector("#marketPaidBar"),
  marketRefundBar: document.querySelector("#marketRefundBar"),
  auctionLab: document.querySelector("#auctionLab"),
  auctionStage: document.querySelector("#auctionStage"),
  bidControlList: document.querySelector("#bidControlList"),
  auctionResult: document.querySelector("#auctionResult"),
  providerRewardList: document.querySelector("#providerRewardList"),
  automationTrigger: document.querySelector("#automationTrigger"),
  automationImprovement: document.querySelector("#automationImprovement"),
  automationCycle: document.querySelector("#automationCycle"),
  automationCommitted: document.querySelector("#automationCommitted"),
  automationSource: document.querySelector("#automationSource"),
  automationLaunch: document.querySelector("#automationLaunch"),
  automationTarget: document.querySelector("#automationTarget"),
  keeperEvents: document.querySelector("#keeperEvents"),
  automationPolicyHash: document.querySelector("#automationPolicyHash"),
  automationMaxSpend: document.querySelector("#automationMaxSpend"),
  automationRemaining: document.querySelector("#automationRemaining"),
  automationActiveLimit: document.querySelector("#automationActiveLimit"),
  automationCooldown: document.querySelector("#automationCooldown"),
  automationApproval: document.querySelector("#automationApproval"),
  automationState: document.querySelector("#automationState"),
  automationLab: document.querySelector("#automationLab"),
  automationImprovementRange: document.querySelector("#automationImprovementRange"),
  automationImprovementOutput: document.querySelector("#automationImprovementOutput"),
  automationBountyRange: document.querySelector("#automationBountyRange"),
  automationBountyOutput: document.querySelector("#automationBountyOutput"),
  automationComputeRange: document.querySelector("#automationComputeRange"),
  automationComputeOutput: document.querySelector("#automationComputeOutput"),
  automationLabResult: document.querySelector("#automationLabResult"),
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
let marketData = null;
let automationData = null;
let toastTimer = null;

boot();

async function boot() {
  const [networkResult, localnetResult, marketResult, automationResult] = await Promise.allSettled([
    fetchJson("./network.json"),
    fetchJson("./localnet.json"),
    fetchJson("./market.json"),
    fetchJson("./automation.json"),
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
  if (marketResult.status === "fulfilled") {
    marketData = marketResult.value;
    hydrateMarket(marketData);
  } else {
    console.warn("Could not load compute market transcript", marketResult.reason);
  }
  if (automationResult.status === "fulfilled") {
    automationData = automationResult.value;
    hydrateAutomation(automationData);
  } else {
    console.warn("Could not load bounty automation transcript", automationResult.reason);
  }

  nodes.improvementRange?.addEventListener("input", updateComposer);
  nodes.rewardRange?.addEventListener("input", updateComposer);
  nodes.bonusRange?.addEventListener("input", updateComposer);
  nodes.downloadRecipe?.addEventListener("click", downloadDraftRecipe);
  nodes.copyRecipe?.addEventListener("click", copySpecimenRecipe);
  nodes.auctionStage?.addEventListener("change", renderBidControls);
  nodes.auctionLab?.addEventListener("submit", runAuctionCounterfactual);
  nodes.automationImprovementRange?.addEventListener("input", updateAutomationLab);
  nodes.automationBountyRange?.addEventListener("input", updateAutomationLab);
  nodes.automationComputeRange?.addEventListener("input", updateAutomationLab);
  nodes.automationLab?.addEventListener("submit", (event) => event.preventDefault());
  document.querySelector("#composer")?.addEventListener("submit", (event) => event.preventDefault());
  updateComposer();
  updateAutomationLab();
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

function hydrateMarket(data) {
  const market = data.summary.market;
  const [escrow] = Object.values(market.compute_escrows);
  const payments = Object.values(market.stage_payments);
  const paidUnits = payments.reduce((sum, payment) => sum + BigInt(payment.payment_units), 0n);
  const fundedUnits = BigInt(escrow.funded_units);
  const refundedUnits = BigInt(escrow.refunded_units);
  const bountyPaid = Object.values(data.summary.bounty_settlements)
    .flatMap((launch) => Object.values(launch))
    .reduce((sum, bounty) => sum + BigInt(bounty.settled_units), 0n);

  setText(nodes.marketBidCount, market.bid_reveals);
  setText(nodes.marketPaidStages, `${payments.length} / ${market.auctions.length}`);
  setText(nodes.marketPayments, formatInteger(paidUnits));
  setText(nodes.marketRefund, formatInteger(refundedUnits));
  setText(nodes.marketConservation, `${formatInteger(market.accounted_supply_units)} = ${formatInteger(market.issued_supply_units)}`);
  setText(nodes.marketFunded, formatInteger(fundedUnits));
  setText(nodes.marketPaid, formatInteger(paidUnits));
  setText(nodes.marketRefundDetail, formatInteger(refundedUnits));
  setText(nodes.marketBountyPaid, formatInteger(bountyPaid));
  if (fundedUnits > 0n) {
    nodes.marketPaidBar.style.width = `${Number((paidUnits * 10000n) / fundedUnits) / 100}%`;
    nodes.marketRefundBar.style.width = `${Number((refundedUnits * 10000n) / fundedUnits) / 100}%`;
  }

  renderMarketAuctions(market.auctions);
  renderProviderRewards(market.compute_reward_distributions);
  nodes.auctionStage.replaceChildren(
    ...market.auctions.map((auction) => {
      const option = document.createElement("option");
      option.value = auction.stage_id;
      option.textContent = humanize(auction.stage_id);
      return option;
    }),
  );
  renderBidControls();
}

function hydrateAutomation(data) {
  const record = data.summary.automation.policies[0];
  const policy = record.policy;
  const cycle = record.cycles[0];
  const result = data.keeper_result;
  const improvementPercent = policy.objective.relative_improvement_bps / 100;

  setText(nodes.automationTrigger, "Promoted");
  setText(nodes.automationImprovement, `${improvementPercent}%`);
  setText(nodes.automationCycle, `${cycle.cycle_index} / ${policy.limits.max_cycles}`);
  setText(nodes.automationCommitted, formatInteger(cycle.committed_units));
  setText(nodes.automationSource, cycle.source_launch_id);
  setText(nodes.automationLaunch, cycle.launch_id);
  setText(nodes.automationTarget, formatInteger(result.target));
  setText(nodes.automationPolicyHash, shortHash(record.policy_sha256));
  nodes.automationPolicyHash.title = record.policy_sha256;
  setText(nodes.automationMaxSpend, formatInteger(policy.budgets.max_total_spend_units));
  setText(nodes.automationRemaining, formatInteger(record.remaining_units));
  setText(nodes.automationActiveLimit, `${policy.limits.max_active_bounties} max`);
  setText(nodes.automationCooldown, `${policy.limits.cooldown_slots} slots`);
  setText(
    nodes.automationApproval,
    formatInteger(policy.budgets.manual_approval_above_units),
  );
  setText(nodes.automationState, cycle.status);
  renderKeeperEvents(result.events);

  nodes.automationImprovementRange.value = String(improvementPercent);
  nodes.automationBountyRange.value = policy.budgets.bounty_units;
  nodes.automationComputeRange.value = policy.budgets.compute_budget_units;
  updateAutomationLab();
}

function renderKeeperEvents(events) {
  if (!nodes.keeperEvents) {
    return;
  }
  nodes.keeperEvents.replaceChildren(
    ...events.map((event) => {
      const item = document.createElement("li");
      const height = document.createElement("span");
      height.textContent = `#${String(event.height).padStart(2, "0")}`;
      const type = document.createElement("strong");
      type.textContent = humanize(event.event_type);
      type.title = event.event_id;
      item.append(height, type);
      return item;
    }),
  );
}

function renderMarketAuctions(auctions) {
  if (!nodes.marketAuctionList) {
    return;
  }
  nodes.marketAuctionList.replaceChildren(
    ...auctions.map((auction) => {
      const row = document.createElement("div");
      row.className = "auction-row";
      const stage = document.createElement("span");
      stage.textContent = humanize(auction.stage_id);
      const provider = document.createElement("span");
      provider.className = "provider-name";
      provider.textContent = providerLabel(auction.provider);
      provider.title = auction.provider;
      const bids = document.createElement("small");
      bids.textContent = `${auction.revealed_bids} reveals`;
      const price = document.createElement("small");
      price.textContent = `${auction.unit_price_units} / unit`;
      const payment = document.createElement("strong");
      payment.textContent = formatInteger(auction.payment_units);
      payment.title = "settled test credits";
      row.append(stage, provider, bids, price, payment);
      return row;
    }),
  );
}

function renderProviderRewards(distributions) {
  if (!nodes.providerRewardList) {
    return;
  }
  const [distribution] = Object.values(distributions);
  const total = BigInt(distribution.total_units);
  nodes.providerRewardList.replaceChildren(
    ...distribution.allocations.map((allocation) => {
      const row = document.createElement("div");
      row.className = "provider-row";
      const provider = document.createElement("span");
      provider.textContent = providerLabel(allocation.provider);
      provider.title = allocation.provider;
      const track = document.createElement("i");
      track.style.width = `${Number((BigInt(allocation.units) * 10000n) / total) / 100}%`;
      const units = document.createElement("strong");
      units.textContent = `${formatInteger(allocation.units)} ITP1`;
      row.append(provider, track, units);
      return row;
    }),
  );
}

function revealedBidsForStage(stageId) {
  return marketData.events
    .filter(
      (event) =>
        event.signed_intent.event_type === "provider_bid_revealed" &&
        event.signed_intent.payload.stage_id === stageId,
    )
    .map((event) => ({
      event_id: event.event_id,
      ...event.signed_intent.payload,
    }));
}

function renderBidControls() {
  if (!marketData || !nodes.bidControlList) {
    return;
  }
  const bids = revealedBidsForStage(nodes.auctionStage.value);
  nodes.bidControlList.replaceChildren(
    ...bids.map((bid) => {
      const row = document.createElement("div");
      row.className = "bid-control";
      const label = document.createElement("label");
      const id = `bid-${bid.stage_id}-${bid.provider.split(":").at(-1)}`;
      label.htmlFor = id;
      label.textContent = providerLabel(bid.provider);
      label.title = bid.provider;
      const input = document.createElement("input");
      input.id = id;
      input.name = bid.provider;
      input.type = "number";
      input.min = "1";
      input.max = "99";
      input.step = "1";
      input.value = bid.unit_price_units;
      input.setAttribute("aria-label", `${providerLabel(bid.provider)} unit price`);
      row.append(label, input);
      return row;
    }),
  );
  runAuctionCounterfactual();
}

function runAuctionCounterfactual(event) {
  event?.preventDefault();
  if (!marketData || !nodes.auctionResult) {
    return;
  }
  const stageId = nodes.auctionStage.value;
  const actual = marketData.summary.market.auctions.find((row) => row.stage_id === stageId);
  const bids = revealedBidsForStage(stageId)
    .map((bid) => {
      const input = [...nodes.bidControlList.querySelectorAll("input")].find(
        (node) => node.name === bid.provider,
      );
      return { ...bid, price: BigInt(input?.value || bid.unit_price_units) };
    })
    .sort((left, right) => {
      if (left.price !== right.price) {
        return left.price < right.price ? -1 : 1;
      }
      return left.event_id.localeCompare(right.event_id);
    });
  const winner = bids[0];
  const computeUnits = BigInt(actual.payment_units) / BigInt(actual.unit_price_units);
  const cost = computeUnits * winner.price;
  const delta = BigInt(actual.payment_units) - cost;
  const comparison =
    delta === 0n
      ? "same cost as signed clearing"
      : delta > 0n
        ? `${formatInteger(delta)} below signed clearing`
        : `${formatInteger(-delta)} above signed clearing`;
  nodes.auctionResult.textContent = `${providerLabel(winner.provider)} wins at ${winner.price} / unit · ${formatInteger(cost)} total · ${comparison}.`;
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

function updateAutomationLab() {
  const improvementPercent = Number(nodes.automationImprovementRange?.value ?? 10);
  const bountyUnits = BigInt(nodes.automationBountyRange?.value ?? 120000);
  const computeUnits = BigInt(nodes.automationComputeRange?.value ?? 12000);
  const baseline = BigInt(automationData?.keeper_result?.baseline ?? 260536589);
  const approvalThreshold = BigInt(
    automationData?.summary?.automation?.policies?.[0]?.policy?.budgets
      ?.manual_approval_above_units ?? 250000,
  );
  const target = (baseline * BigInt(100 - improvementPercent)) / 100n;
  const cycleSpend = bountyUnits + computeUnits;

  setText(nodes.automationImprovementOutput, `${improvementPercent}%`);
  setText(nodes.automationBountyOutput, formatInteger(bountyUnits));
  setText(nodes.automationComputeOutput, formatInteger(computeUnits));
  if (!nodes.automationLabResult) {
    return;
  }
  const headline = document.createElement("strong");
  headline.textContent = `${formatInteger(target)} target · ${formatInteger(cycleSpend)} credits`;
  const detail = document.createElement("span");
  detail.textContent =
    cycleSpend > approvalThreshold
      ? `Separate sponsor approval required above ${formatInteger(approvalThreshold)}.`
      : `Within the signed automatic threshold of ${formatInteger(approvalThreshold)}.`;
  nodes.automationLabResult.replaceChildren(headline, detail);
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

function providerLabel(account) {
  return String(account).split(":").at(-1).replaceAll("-", " ");
}
