// src/sarjy/interfaces/web/static/fillers.js
// fillers.js — local, never from the server; spoken only if no sentence arrives within 700 ms of tool start
export const FILLERS = { get_weather: ["Let me check.", "One sec, checking the weather.", "Checking now."],
                         default: ["One moment.", "Let me see.", "Just a sec."] };
export function pickFiller(tool) { const l = FILLERS[tool] || FILLERS.default; return l[Math.floor(Math.random() * l.length)]; }
