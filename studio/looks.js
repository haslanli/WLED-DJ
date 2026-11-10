"use strict";
// The looks page: every clip under one look, beside another, with the numbers read from its rendered frames.
// Reads bench/out/looks/looks.json (written by bench/render_look.py). Writes nothing.

const $ = (id) => document.getElementById(id);
const media = (file) => `/looks/media/${encodeURIComponent(file)}`;
const LAYERS = ["drums", "bass", "vocals", "other"];
let index = null, look = null, beside = null;

const percent = (value) => `${Math.round(value * 100)}%`;

function numbers(n, kind) {
  if (!n) return [];
  const rows = [["Full-strip flashes in the busiest second", String(n.large_flashes)]];
  if (kind === "new" && n.unlimited_flashes !== undefined) {
    rows.push(["Before the flash limit", String(n.unlimited_flashes)]);
    rows.push(["Frames the flash limit dimmed", String(n.held)]);
  }
  rows.push(["Fastest blinking of one spot", `${n.blink_hz} a second`]);
  rows.push(["Average lightness", percent(n.lightness)]);
  rows.push(["Strip lit", percent(n.lit)]);
  return rows;
}

function video(file, muted) {
  const element = document.createElement("video");
  element.controls = true;
  element.preload = "metadata";
  element.playsInline = true;
  element.muted = muted;
  element.src = media(file);
  return element;
}

function side(key, clip, muted) {
  const entry = index.looks[key];
  const box = document.createElement("div");
  box.className = "look-side";
  const name = document.createElement("div");
  name.className = "look-name";
  name.textContent = entry.name || key;
  box.append(name);
  const file = entry.videos && entry.videos[clip.id];
  if (!file) {
    const none = document.createElement("p");
    none.className = "help";
    none.textContent = `There is no render of this clip for ${entry.name || key}.`;
    box.append(none);
    return box;
  }
  box.append(video(file, muted));
  const table = document.createElement("dl");
  table.className = "look-numbers";
  for (const [label, value] of numbers(entry.numbers && entry.numbers[clip.id], entry.kind)) {
    const term = document.createElement("dt"), detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    detail.className = "num";
    table.append(term, detail);
  }
  box.append(table);
  return box;
}

function card(clip) {
  const section = document.createElement("section");
  section.className = "card review";
  const head = document.createElement("div");
  head.className = "review-head";
  const title = document.createElement("h2");
  title.textContent = clip.title;
  const play = document.createElement("button");
  play.type = "button";
  play.className = "outline";
  play.textContent = "Play both";
  head.append(title, play);
  section.append(head, side(look, clip, false));
  if (beside && beside !== look) section.append(side(beside, clip, true));
  play.addEventListener("click", () => {
    const both = [...section.querySelectorAll(":scope > .look-side > video")];
    const playing = both.some((v) => !v.paused);
    for (const other of document.querySelectorAll("video")) other.pause();
    if (playing) return;
    for (const v of both) { v.currentTime = 0; v.play(); }
  });
  const layers = index.looks[look].layers && index.looks[look].layers[clip.id];
  if (layers) {
    const label = document.createElement("div");
    label.className = "look-name";
    label.textContent = "One layer at a time";
    const grid = document.createElement("div");
    grid.className = "look-layers";
    for (const layer of LAYERS) {
      if (!layers[layer]) continue;
      const cell = document.createElement("div");
      const name = document.createElement("div");
      name.className = "meta";
      name.textContent = layer === "other" ? "Other alone" : `${layer[0].toUpperCase()}${layer.slice(1)} alone`;
      cell.append(name, video(layers[layer], false));
      grid.append(cell);
    }
    section.append(label, grid);
  }
  return section;
}

function picker(box, keys, current, choose) {
  box.replaceChildren(...keys.map((key) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(key === current));
    button.textContent = key === null ? "Nothing" : index.looks[key].name || key;
    button.addEventListener("click", () => choose(key));
    return button;
  }));
}

function render() {
  const keys = Object.keys(index.looks);
  const fresh = keys.filter((key) => index.looks[key].kind === "new");
  picker($("look"), fresh, look, (key) => { look = key; render(); });
  picker($("beside"), [...keys.filter((key) => key !== look), null], beside === look ? null : beside,
         (key) => { beside = key; render(); });
  $("counts").textContent = `${index.clips.length} clips · ${keys.length} looks`;
  $("clips").replaceChildren(...index.clips.map(card));
}

fetch(media("looks.json"), { cache: "no-store" }).then((response) => {
  if (!response.ok) throw new Error("No renders yet. Run bench/render_look.py first.");
  return response.json();
}).then((data) => {
  index = data;
  const keys = Object.keys(index.looks);
  look = keys.find((key) => index.looks[key].kind === "new");
  beside = keys.find((key) => key !== look) || null;
  if (!look) throw new Error("No renders yet. Run bench/render_look.py first.");
  render();
}).catch((error) => {
  $("notice").textContent = error.message;
  $("notice").hidden = false;
  $("counts").textContent = "No renders";
});
