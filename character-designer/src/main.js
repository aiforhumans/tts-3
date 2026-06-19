import { templates } from "./templates/index.js";
import { createCharacterForm } from "./components/CharacterForm/index.js";
import { createPreview } from "./components/Preview/index.js";
import { createExportPanel } from "./components/ExportPanel/index.js";
import { createImportPanel } from "./components/ImportPanel/index.js";

let state = { ...templates.blank };
const listeners = [];

function getState() {
  return structuredClone(state);
}

function setState(next) {
  state = { ...templates.blank, ...next, traits: Array.isArray(next.traits) ? next.traits : [] };
  listeners.forEach((fn) => fn(getState()));
}

function notify(msg) {
  console.log(msg);
  alert(msg);
}

function mount() {
  const app = document.querySelector("#app");
  const left = document.createElement("section");
  const right = document.createElement("section");

  const form = createCharacterForm({ getState, setState, notify });
  const preview = createPreview();
  const importer = createImportPanel({ setState, notify });
  const exporter = createExportPanel({ getState, toPrompt: preview.toPrompt, notify });

  left.append(form.el, importer.el);
  right.append(preview.el, exporter.el);
  app.append(left, right);

  [form, preview, importer, exporter].forEach((component) => listeners.push(component.update));
  setState(state);

  const toggle = document.querySelector("#theme-toggle");
  toggle?.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme") || "dark";
    document.body.setAttribute("data-theme", current === "dark" ? "light" : "dark");
  });
}

mount();
