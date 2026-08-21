// src/sarjy/interfaces/web/static/ocean.js — UI only; no business rules.
//
// Listens for `sarjy:turn-done` (dispatched by voice.js with the SSE `done`
// payload as `detail`). `detail.workflow` is the same small status blob every
// assessment reply carries mid-turn (`status`, `item`, `total`, `run_id`):
//   - "active"   -> show the item scale so the next turn can be a tap, not typing.
//   - "complete" -> fetch the full results from `GET /workflow/latest` and render them.
//   - anything else (no workflow, proposed/paused/scoring/abandoned) -> hide both.
import { ensureSession, sendText } from "./voice.js";

const { apiBase } = window.SARJY;
const panel = document.getElementById("ocean-panel");
const itemBox = document.getElementById("ocean-item");
const itemText = document.getElementById("ocean-item-text");
const results = document.getElementById("ocean-results");
const bars = document.getElementById("ocean-bars");
const narrative = document.getElementById("ocean-narrative");
const disclaimerEl = document.getElementById("ocean-disclaimer");

const TRAIT_NAMES = {
  O: "Openness",
  C: "Conscientiousness",
  E: "Extraversion",
  A: "Agreeableness",
  N: "Neuroticism",
};
const TRAIT_ORDER = ["O", "C", "E", "A", "N"];

function hideAll() {
  itemBox.hidden = true;
  results.hidden = true;
  panel.hidden = true;
}

function showItem(wf) {
  itemText.textContent = `Question ${wf.item} of ${wf.total}`;
  results.hidden = true;
  itemBox.hidden = false;
  panel.hidden = false;
}

function barRow(code, score, band) {
  const row = document.createElement("div");
  row.className = "bar-row";

  const label = document.createElement("span");
  label.className = "bar-label";
  label.textContent = TRAIT_NAMES[code] || code;

  const track = document.createElement("div");
  track.className = "bar";
  const fill = document.createElement("div");
  fill.className = "bar-fill";
  const pct = typeof score === "number" ? Math.max(0, Math.min(100, ((score - 1) / 4) * 100)) : 0;
  fill.style.width = `${pct}%`;
  track.append(fill);

  const val = document.createElement("span");
  val.className = "bar-val";
  val.textContent = typeof score === "number" ? `${score.toFixed(1)} · ${band || "n/a"}` : "n/a";

  row.append(label, track, val);
  return row;
}

async function renderResults() {
  let body;
  try {
    const { access_token } = await ensureSession();
    const res = await fetch(`${apiBase}/workflow/latest`, {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    if (!res.ok) { hideAll(); return; }
    body = await res.json();
  } catch {
    hideAll();
    return;
  }
  if (body.status !== "complete" || !body.results) { hideAll(); return; }

  bars.replaceChildren();
  for (const code of TRAIT_ORDER) {
    const bands = body.bands || {};
    bars.append(barRow(code, body.results[code], bands[code]));
  }
  narrative.textContent = body.narrative || "";
  disclaimerEl.textContent = body.disclaimer || "";

  itemBox.hidden = true;
  results.hidden = false;
  panel.hidden = false;
}

document.addEventListener("sarjy:turn-done", (e) => {
  const wf = e.detail && e.detail.workflow;
  if (!wf) { hideAll(); return; }
  if (wf.status === "active") { showItem(wf); return; }
  if (wf.status === "complete") { renderResults(); return; }
  hideAll();
});

itemBox.querySelectorAll("button[data-v]").forEach((b) => {
  b.addEventListener("click", () => sendText(b.dataset.v));
});
