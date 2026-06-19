import { fromSillyTavern } from "../../importers/sillyTavern.js";
import { fromSpicyChat } from "../../importers/spicyChat.js";
import { readPngCard } from "../../importers/pngCard.js";

export function createImportPanel({ setState, notify }) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Import</h2>
    <p>Supports SillyTavern (.json, .png card) and SpicyChat (.json).</p>
    <input id="import-file" type="file" accept=".json,.png" />
  `;

  panel.querySelector("#import-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const lower = file.name.toLowerCase();
      if (lower.endsWith(".png")) {
        const parsed = await readPngCard(file);
        if (!parsed) throw new Error("No character metadata found in PNG");
        const normalized = parsed?.spec === "chara_card_v2" || parsed?.spec === "chara_card_v1"
          ? fromSillyTavern(parsed)
          : fromSpicyChat(parsed);
        setState(normalized);
        notify("Imported PNG character card");
      } else if (lower.endsWith(".json")) {
        const text = await file.text();
        const json = JSON.parse(text);
        const normalized = json?.platform === "spicychat" || json?.first_message
          ? fromSpicyChat(json)
          : fromSillyTavern(json);
        setState(normalized);
        notify("Imported JSON character");
      } else {
        throw new Error("Unsupported file type");
      }
    } catch (err) {
      notify(`Import failed: ${err.message || err}`);
    } finally {
      e.target.value = "";
    }
  });

  return { el: panel, update() {} };
}
