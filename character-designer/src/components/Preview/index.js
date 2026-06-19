function promptFromCharacter(c) {
  return [
    `You are ${c.name || "[Name]"}${c.nickname ? ` ("${c.nickname}")` : ""}.`,
    c.age ? `Age: ${c.age}.` : "",
    c.gender ? `Gender: ${c.gender}.` : "",
    c.species ? `Species: ${c.species}.` : "",
    c.traits?.length ? `Personality traits: ${c.traits.join(", ")}.` : "",
    c.description ? `Appearance: ${c.description}` : "",
    c.background ? `Background: ${c.background}` : "",
    c.scenario ? `Scenario: ${c.scenario}` : "",
    c.greeting ? `Initial greeting: ${c.greeting}` : "",
    c.exampleDialogue ? `Dialogue style example: ${c.exampleDialogue}` : "",
    "Stay in character and write immersive roleplay responses.",
  ]
    .filter(Boolean)
    .join("\n");
}

export function createPreview() {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Live System Prompt Preview</h2>
    <div class="preview" id="preview-content"></div>
  `;
  const target = panel.querySelector("#preview-content");

  return {
    el: panel,
    update(state) {
      target.textContent = promptFromCharacter(state);
    },
    toPrompt: promptFromCharacter,
  };
}
