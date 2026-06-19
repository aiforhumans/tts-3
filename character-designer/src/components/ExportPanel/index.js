import { toSillyTavernV1, toSillyTavernV2 } from "../../exporters/sillyTavern.js";
import { toSpicyChat } from "../../exporters/spicyChat.js";
import { exportPngCard } from "../../exporters/genericPng.js";

function downloadBlob(filename, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

function downloadJson(filename, obj) {
  downloadBlob(filename, new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" }));
}

function downloadText(filename, text) {
  downloadBlob(filename, new Blob([text], { type: "text/plain" }));
}

export function createExportPanel({ getState, toPrompt, notify }) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Export</h2>
    <div class="row" style="flex-wrap:wrap;">
      <button type="button" id="st-v1">SillyTavern V1 JSON</button>
      <button type="button" id="st-v2">SillyTavern V2 JSON</button>
      <button type="button" id="spicy">SpicyChat JSON</button>
      <button type="button" id="generic-json">Generic JSON</button>
      <button type="button" id="prompt">System Prompt TXT</button>
      <button type="button" id="png">PNG Character Card</button>
    </div>
  `;

  panel.querySelector("#st-v1").onclick = () => downloadJson("character-st-v1.json", toSillyTavernV1(getState()));
  panel.querySelector("#st-v2").onclick = () => downloadJson("character-st-v2.json", toSillyTavernV2(getState()));
  panel.querySelector("#spicy").onclick = () => downloadJson("character-spicychat.json", toSpicyChat(getState()));
  panel.querySelector("#generic-json").onclick = () => downloadJson("character-generic.json", getState());
  panel.querySelector("#prompt").onclick = () => downloadText("character-system-prompt.txt", toPrompt(getState()));

  panel.querySelector("#png").onclick = async () => {
    try {
      const state = getState();
      const card = toSillyTavernV2(state);
      const blob = await exportPngCard(state, card, "chara");
      downloadBlob("character-card.png", blob);
    } catch (err) {
      notify(`PNG export failed: ${err.message || err}`);
    }
  };

  return { el: panel, update() {} };
}
