"use strict";
// WLED DJ studio: one WebSocket for status (2/s) and preview frames (60/s); controls send small commands.

const $ = (id) => document.getElementById(id);
const state = { status: null, frame: null, ws: null, retry: 500, lastSent: null, lastSentAt: null, rate: null,
                dragging: { master: false, trim: false }, quitArmed: null };

// ---------- connection ----------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(TOKEN)}`);
  state.ws = ws;
  ws.onopen = () => { state.retry = 500; document.body.classList.remove("disconnected"); };
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "status") renderStatus(message);
    else if (message.type === "frame") state.frame = message;
    else if (message.type === "error") showNotice(message.message);
  };
  ws.onclose = () => {
    document.body.classList.add("disconnected");
    $("summary").textContent = "Engine not running";
    $("finding").textContent = "The studio lost the engine. Start WLED DJ again from its menu-bar icon.";
    setTimeout(reconnect, state.retry);
    state.retry = Math.min(state.retry * 2, 5000);
  };
}

// Each engine run has its own token: once an engine answers again, reload to pick up the new one.
async function reconnect() {
  try {
    const health = await fetch("/api/health", { cache: "no-store" });
    if (health.ok) { location.reload(); return; }
  } catch (error) { /* engine still down */ }
  setTimeout(reconnect, state.retry);
  state.retry = Math.min(state.retry * 2, 5000);
}

function send(message) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(message));
}

let noticeTimer = null;
function showNotice(text) {
  const notice = $("notice");
  notice.textContent = text;
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 4000);
}

// ---------- status ----------
const ms = (value) => `${Math.round(value)} ms`;

function finding(s) {
  const capture = s.capture, controller = s.controller, out = capture.output;
  if (capture.state === "no-audio") return capture.message;
  if (capture.state === "restarting") return `Audio capture is restarting: ${capture.message}`;
  if (controller.state === "offline") return `Can't reach your lights at ${controller.host}. Trying again every 2 seconds.`
    + (controller.message ? ` ${controller.message}` : "");
  if (controller.state === "busy") return `Your lights are showing another live stream. WLED DJ takes over when it stops.`;
  if (controller.state === "choose") return "More than one WLED is on your network. Pick yours under Lights.";
  if (controller.state === "searching") return "Looking for WLED on your network. You can also connect by address under Lights.";
  if (s.output.handback) return "The music stopped, so WLED is showing its own effect. The show returns with the music.";
  if (!out) return "Waiting for the Mac's output device.";
  if (s.sync.delay_ms >= 1) return `Hearing ${out.name}. Lights wait ${ms(s.sync.delay_ms)} so they land with the sound.`;
  if (s.sync.lead_ms >= 1) return `Hearing ${out.name}. Lights run ${ms(s.sync.lead_ms)} behind, so beats fire early to make up for it.`;
  return `Hearing ${out.name}. Lights and sound are in step.`;
}

function renderStatus(s) {
  state.status = s;
  const controller = s.controller, out = s.capture.output;
  $("title").textContent = controller.state === "online" ? `${controller.name} · ${controller.pixels} lights` : "Your lights";
  $("finding").textContent = finding(s);
  const head = s.capture.state !== "running" ? "Capture " + s.capture.state.replace("-", " ")
             : s.output.handback ? "Idle" : "Running";
  $("summary").textContent = [head, out && out.name, controller.state === "online" ? `${controller.name} online` : `lights ${controller.state}`]
    .filter(Boolean).join(" · ");

  const music = s.music;
  $("k-tempo").textContent = music.bpm > 0 ? `${Math.round(music.bpm)} BPM` : "–";
  $("k-tempo-c").textContent = music.silent ? "no music playing" : music.confidence >= .6 ? "locked on the beat"
    : music.bpm > 0 ? "finding the beat" : "no steady beat yet";
  $("k-out").textContent = out ? ms(out.latency_ms) : "–";
  $("k-out-c").textContent = out ? out.name : "waiting for the output device";
  $("k-delay").textContent = ms(s.sync.delay_ms);
  $("k-delay-c").textContent = s.sync.lead_ms >= 1 ? `beats fire ${ms(s.sync.lead_ms)} early instead` : "to land with the sound";
  const now = performance.now();
  if (state.lastSent !== null && now > state.lastSentAt) {
    const rate = (s.output.sent - state.lastSent) / ((now - state.lastSentAt) / 1000);
    state.rate = state.rate === null ? rate : state.rate * .6 + rate * .4;
  }
  state.lastSent = s.output.sent; state.lastSentAt = now;
  const live = controller.state === "online" && !s.output.handback;
  $("k-fps").textContent = live && state.rate !== null ? `${Math.round(state.rate)} /s` : "0 /s";
  const audioDrops = (s.capture.drops || 0) + (s.output.audio_dropped || 0);
  $("k-fps-c").textContent = [controller.fps ? `of ${controller.fps} possible` : null,
    `${s.output.errors.toLocaleString()} failed`, `${audioDrops.toLocaleString()} audio drops`].filter(Boolean).join(" · ");

  const blackout = $("blackout");
  blackout.setAttribute("aria-pressed", String(s.blackout));
  blackout.textContent = s.blackout ? "Lights back on" : "Blackout";
  if (!state.dragging.master) { $("master").value = Math.round(s.master * 100); $("master-v").textContent = `${Math.round(s.master * 100)}%`; }
  if (!state.dragging.trim) { $("trim").value = Math.round(s.sync.trim_ms); $("trim-v").textContent = signedMs(s.sync.trim_ms); }
  $("trim").disabled = !out;
  $("trim-help").textContent = out
    ? `Pick Sync check, then slide until the flashes land on the kicks. Saved for ${out.name}.`
    : "Trim becomes available once WLED DJ knows where the sound is playing.";
  renderLayers(s.layers);
  renderStructure(s.structure);
  $("safety").hidden = s.flash_limit === undefined;     // an engine from before the flash limit sends none: no control for it
  for (const button of $("flash-limit").children) {
    button.setAttribute("aria-checked", String(Number(button.dataset.value) === s.flash_limit));
  }
  renderLooks(s);
  renderController(s);
  renderStrips(controller);
}

const signedMs = (value) => `${value > 0 ? "+" : ""}${Math.round(value)} ms`;

function renderLayers(layers) {
  if (!layers) return;
  $("layers-note").textContent = layers.message
    || (layers.source === "stems" ? "Layers are read by the separation model."
      : layers.state === "starting" ? "Layer separation is starting. Until then, layers are estimated from the mix."
      : layers.state === "on" ? "Layer separation is warming up. Until then, layers are estimated from the mix."
      : "Layers are estimated from the mix.");
  for (const button of $("stems").children) {
    button.setAttribute("aria-checked", String((button.dataset.on === "true") === layers.enabled));
  }
}

function renderStructure(structure) {
  if (!structure) return;
  const percent = (value) => `${Math.round(value * 100)}%`;
  $("structure-note").hidden = false;     // an engine from before this line existed sends no structure: the line stays away
  $("structure-note").textContent =
    `Section: ${structure.section}. Energy ${percent(structure.energy)}, tension ${percent(structure.tension)}.`;
}

function renderLooks(s) {
  const box = $("looks");
  if (box.childElementCount !== s.looks.length) {
    box.replaceChildren(...s.looks.map((look) => {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("role", "radio");
      button.dataset.look = look.id;
      button.textContent = look.name;
      button.addEventListener("click", () => send({ type: "look", id: look.id }));
      return button;
    }));
  }
  for (const button of box.children) button.setAttribute("aria-checked", String(button.dataset.look === s.look));
}

function renderController(s) {
  const c = s.controller;
  const names = { online: "Online", offline: "Offline", busy: "Busy", searching: "Searching", choose: "Pick one" };
  $("c-name").textContent = c.host ? (c.name || c.host) : "Looking for WLED…";
  $("c-meta").textContent = c.state === "online" ? `${c.host} · ${c.pixels} pixels · ${c.fps} fps` : (c.host || "on this network");
  const pill = $("c-state");
  pill.textContent = names[c.state] || c.state;
  pill.className = `pill ${c.state}`;
  const box = $("candidates");
  const key = JSON.stringify(c.candidates);
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.replaceChildren(...c.candidates.map((candidate) => {
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = `<b></b><span class="num"></span>`;
      button.querySelector("b").textContent = candidate.name;
      button.querySelector("span").textContent = candidate.host;
      button.addEventListener("click", () => send({ type: "controller", host: candidate.host }));
      return button;
    }));
  }
}

// ---------- preview ----------
let stripKey = "";
function renderStrips(controller) {
  const buses = controller.buses && controller.buses.length ? controller.buses
    : [{ start: 0, length: controller.pixels || 78, reversed: false }];
  const key = JSON.stringify(buses);
  if (key === stripKey) return;
  stripKey = key;
  $("strips").replaceChildren(...buses.map((bus, index) => {
    const row = document.createElement("div");
    row.className = "strip";
    const label = document.createElement("span");
    label.textContent = `Strip ${index + 1}`;
    const canvas = document.createElement("canvas");
    canvas.dataset.start = bus.start; canvas.dataset.length = bus.length; canvas.dataset.reversed = bus.reversed;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Strip ${index + 1}, ${bus.length} pixels`);
    row.append(label, canvas);
    return row;
  }));
}

function decode(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function draw() {
  const frame = state.frame;
  if (frame) {
    const bytes = decode(frame.pixels);
    const scale = window.devicePixelRatio || 1;
    for (const canvas of document.querySelectorAll(".strip canvas")) {
      const width = Math.round(canvas.clientWidth * scale), height = Math.round(canvas.clientHeight * scale);
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const ctx = canvas.getContext("2d");
      const start = +canvas.dataset.start, length = +canvas.dataset.length, reversed = canvas.dataset.reversed === "true";
      ctx.clearRect(0, 0, width, height);
      const cell = width / length;
      // A diffused strip: a soft spill around a solid core, like the strip under its cover.
      for (let pass = 0; pass < 2; pass++) {
        ctx.filter = pass === 0 ? `blur(${Math.max(3, cell * 1.2)}px)` : "none";
        ctx.globalAlpha = pass === 0 ? .55 : 1;
        for (let i = 0; i < length; i++) {
          const p = (start + (reversed ? length - 1 - i : i)) * 3;
          if (p + 2 >= bytes.length) break;
          ctx.fillStyle = `rgb(${bytes[p]},${bytes[p + 1]},${bytes[p + 2]})`;
          if (pass === 0) ctx.fillRect(i * cell, 0, cell + 1, height);
          else ctx.fillRect(i * cell, height * .3, cell + .6, height * .4);
        }
      }
      ctx.filter = "none";
      ctx.globalAlpha = 1;
    }
    for (const fill of document.querySelectorAll("[data-meter]")) {
      fill.style.width = `${Math.round((frame.meters[fill.dataset.meter] || 0) * 100)}%`;
    }
    for (const fill of document.querySelectorAll("[data-layer]")) {
      const layer = (frame.layers || {})[fill.dataset.layer] || { level: 0, present: false };
      fill.style.width = `${Math.round(layer.level * 100)}%`;
      fill.classList.toggle("absent", !layer.present);
    }
    const s = state.status;
    const onBeat = s && s.music.bpm > 0 && s.music.confidence >= .3 && frame.beat < .12;
    $("beat-dot").classList.toggle("on", Boolean(onBeat));
  }
  requestAnimationFrame(draw);
}

// ---------- controls ----------
function throttled(fn) {
  let pending = null;
  return (value) => {
    if (pending === null) requestAnimationFrame(() => { fn(pending); pending = null; });
    pending = value;
  };
}

const sendMaster = throttled((value) => send({ type: "master", value: value / 100 }));
const sendTrim = throttled((value) => send({ type: "trim", ms: value }));

function wireRange(id, key, onValue, format) {
  const input = $(id);
  input.addEventListener("pointerdown", () => { state.dragging[key] = true; });
  input.addEventListener("pointerup", () => { state.dragging[key] = false; });
  input.addEventListener("change", () => { state.dragging[key] = false; });
  input.addEventListener("input", () => { onValue(+input.value); $(`${id}-v`).textContent = format(+input.value); });
}

function toggleBlackout() {
  send({ type: "blackout", on: !(state.status && state.status.blackout) });
}

$("blackout").addEventListener("click", toggleBlackout);
// The link to the looks page shows once the engine answers for it: an engine from before the page has none.
fetch("/looks", { method: "HEAD", cache: "no-store" }).then((r) => { $("looks-link").hidden = !r.ok; }).catch(() => {});
for (const button of $("stems").children) {
  button.addEventListener("click", () => send({ type: "stems", on: button.dataset.on === "true" }));
}
for (const button of $("flash-limit").children) {
  button.addEventListener("click", () => send({ type: "flash_limit", value: Number(button.dataset.value) }));
}
wireRange("master", "master", sendMaster, (v) => `${v}%`);
wireRange("trim", "trim", sendTrim, signedMs);
// Space is the blackout key everywhere except text fields (Enter still presses a focused button).
document.addEventListener("keydown", (event) => {
  const typing = event.target instanceof HTMLInputElement && event.target.type !== "range";
  if (event.code === "Space" && !typing && !event.repeat) {
    event.preventDefault();
    toggleBlackout();
  }
});
document.addEventListener("keyup", (event) => {
  if (event.code === "Space" && !(event.target instanceof HTMLInputElement && event.target.type !== "range")) event.preventDefault();
});
$("add").addEventListener("submit", (event) => {
  event.preventDefault();
  const host = $("host").value.trim();
  if (!host) { showNotice("Enter your controller's address, for example 192.168.1.50."); return; }
  send({ type: "controller", host });
  $("host").value = "";
});
$("quit").addEventListener("click", () => {
  const button = $("quit");
  if (state.quitArmed) {
    clearTimeout(state.quitArmed);
    send({ type: "quit" });
    button.textContent = "Quitting…";
    return;
  }
  button.textContent = "Click again to quit";
  button.classList.add("confirm");
  state.quitArmed = setTimeout(() => {
    state.quitArmed = null;
    button.textContent = "Quit WLED DJ";
    button.classList.remove("confirm");
  }, 3000);
});

connect();
requestAnimationFrame(draw);
