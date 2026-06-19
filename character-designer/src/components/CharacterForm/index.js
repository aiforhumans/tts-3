import { templates, traitSuggestions } from "../../templates/index.js";

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result || ""));
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

export function createCharacterForm({ getState, setState, notify }) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Character Profile Editor</h2>
    <div class="grid">
      <div class="field full">
        <label>Template</label>
        <select id="template-select">
          <option value="blank">Blank</option>
          <option value="warrior">Warrior</option>
          <option value="wizard">Wizard</option>
          <option value="romance">Romance</option>
          <option value="villain">Villain</option>
        </select>
      </div>
      <div class="field"><label>Name</label><input id="name" /></div>
      <div class="field"><label>Nickname</label><input id="nickname" /></div>
      <div class="field"><label>Age</label><input id="age" /></div>
      <div class="field"><label>Gender</label><input id="gender" /></div>
      <div class="field"><label>Species / Race</label><input id="species" /></div>

      <div class="field full">
        <label>Personality Traits</label>
        <div class="row">
          <input id="trait-input" placeholder="Add trait" />
          <button type="button" id="add-trait">Add</button>
        </div>
        <div id="trait-list" class="chips"></div>
        <small>Suggestions</small>
        <div id="trait-suggestions" class="chips"></div>
      </div>

      <div class="field full"><label>Description / Appearance</label><textarea id="description"></textarea></div>
      <div class="field full"><label>Background / Lore</label><textarea id="background"></textarea></div>
      <div class="field full"><label>Scenario</label><textarea id="scenario"></textarea></div>
      <div class="field full"><label>Greeting</label><textarea id="greeting"></textarea></div>
      <div class="field full"><label>Example Dialogue</label><textarea id="exampleDialogue"></textarea></div>

      <div class="field full">
        <label>Avatar URL</label>
        <input id="avatarUrl" placeholder="https://..." />
      </div>
      <div class="field full">
        <label>Avatar Upload</label>
        <input id="avatarFile" type="file" accept="image/*" />
      </div>
      <div class="field full">
        <img id="avatarPreview" class="avatar-preview" alt="avatar preview" />
      </div>
    </div>
  `;

  const ids = [
    "name",
    "nickname",
    "age",
    "gender",
    "species",
    "description",
    "background",
    "scenario",
    "greeting",
    "exampleDialogue",
    "avatarUrl",
  ];

  const traitList = panel.querySelector("#trait-list");
  const suggestions = panel.querySelector("#trait-suggestions");
  const traitInput = panel.querySelector("#trait-input");
  const avatarPreview = panel.querySelector("#avatarPreview");

  ids.forEach((id) => {
    panel.querySelector(`#${id}`).addEventListener("input", (e) => {
      const next = { ...getState(), [id]: e.target.value };
      if (id === "avatarUrl" && e.target.value.trim()) {
        next.avatarData = "";
      }
      setState(next);
    });
  });

  panel.querySelector("#template-select").addEventListener("change", (e) => {
    const selected = templates[e.target.value] || templates.blank;
    setState({ ...templates.blank, ...selected });
  });

  panel.querySelector("#add-trait").addEventListener("click", () => {
    const value = traitInput.value.trim();
    if (!value) return;
    const current = getState();
    if (!current.traits.includes(value)) {
      setState({ ...current, traits: [...current.traits, value] });
    }
    traitInput.value = "";
  });

  panel.querySelector("#avatarFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const data = await fileToDataUrl(file);
      setState({ ...getState(), avatarData: data, avatarUrl: "" });
    } catch {
      notify("Could not read image file");
    }
  });

  function renderTraits(state) {
    traitList.innerHTML = "";
    state.traits.forEach((t) => {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.type = "button";
      chip.textContent = `${t} ×`;
      chip.onclick = () => {
        setState({ ...state, traits: state.traits.filter((x) => x !== t) });
      };
      traitList.appendChild(chip);
    });

    suggestions.innerHTML = "";
    traitSuggestions.forEach((t) => {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.type = "button";
      chip.textContent = t;
      chip.disabled = state.traits.includes(t);
      chip.onclick = () => setState({ ...state, traits: [...state.traits, t] });
      suggestions.appendChild(chip);
    });
  }

  return {
    el: panel,
    update(state) {
      ids.forEach((id) => {
        const input = panel.querySelector(`#${id}`);
        if (input && input.value !== (state[id] || "")) input.value = state[id] || "";
      });
      const avatar = state.avatarData || state.avatarUrl;
      avatarPreview.src = avatar || "";
      avatarPreview.style.display = avatar ? "block" : "none";
      renderTraits(state);
    },
  };
}
