// src/sarjy/interfaces/web/static/memory.js — UI only; no business rules.
import { ensureSession, sb } from "./voice.js";

const { apiBase } = window.SARJY;
const panel = document.getElementById("memory-panel");
const list = document.getElementById("memory-list");
const deleteBtn = document.getElementById("delete-account");
const deleteErr = document.getElementById("delete-account-err");
const transcript = document.getElementById("transcript");

async function authHeaders() {
  const { access_token } = await ensureSession();
  return { Authorization: `Bearer ${access_token}`, "Content-Type": "application/json" };
}

function render(items) {
  list.replaceChildren();
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "Nothing stored yet.";
    list.append(li);
    return;
  }
  for (const m of items) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${(m.key || "note").replaceAll("_", " ")}: ${m.value}`;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "mem-del";
    del.textContent = "✕";
    del.setAttribute("aria-label", `Forget ${m.key || "note"}`);
    const err = document.createElement("span");
    err.className = "mem-err";
    err.hidden = true;
    del.addEventListener("click", async () => {
      del.disabled = true;
      err.hidden = true;
      let ok = false;
      try {
        const res = await fetch(`${apiBase}/memory/${m.id}`, {
          method: "DELETE",
          headers: await authHeaders(),
        });
        ok = res.ok;
      } catch {
        ok = false;
      }
      if (!ok) {
        err.textContent = "Couldn't delete — try again.";
        err.hidden = false;
        del.disabled = false;
        return;
      }
      await refresh();
    });
    li.append(label, del, err);
    list.append(li);
  }
}

async function refresh() {
  try {
    const res = await fetch(`${apiBase}/memory`, { headers: await authHeaders() });
    if (res.ok) render(await res.json());
  } catch {
    /* panel is best-effort */
  }
}

panel.addEventListener("toggle", () => {
  if (panel.open) refresh();
});
document.addEventListener("sarjy:turn-done", () => {
  if (panel.open) refresh();
});

// ---------- account deletion (PRD §11 user rights) ----------
deleteBtn?.addEventListener("click", async () => {
  if (!confirm("Delete all your data? Sarjy will forget everything it knows about you.")) return;
  if (!confirm("This cannot be undone. Delete everything?")) return;
  deleteBtn.disabled = true;
  deleteErr.hidden = true;
  let ok = false;
  try {
    const res = await fetch(`${apiBase}/account`, { method: "DELETE", headers: await authHeaders() });
    ok = res.ok;
  } catch {
    ok = false;
  }
  if (!ok) {
    deleteErr.textContent = "Couldn't delete your data — try again.";
    deleteErr.hidden = false;
    deleteBtn.disabled = false;
    return;
  }
  const msg = "All your data is gone. Starting fresh.";
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  bubble.textContent = msg;
  transcript.append(bubble);
  await speakAndWait(msg);
  await sb.auth.signOut();
  location.reload();
});

// Speaks `msg` and resolves once it has finished — or after a ~4s fallback if
// speechSynthesis is unavailable, never fires `onend` (some browsers drop it
// on tab-hidden/navigation), or errors. Only after this may callers sign out
// and reload, so the deletion confirmation is actually heard first.
function speakAndWait(msg) {
  return new Promise((resolve) => {
    if (!("speechSynthesis" in window)) {
      resolve();
      return;
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(finish, 4000);
    const utterance = new SpeechSynthesisUtterance(msg);
    utterance.onend = finish;
    utterance.onerror = finish;
    speechSynthesis.speak(utterance);
  });
}
