function splitTraits(value) {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (!value || typeof value !== "string") return [];
  return value.split(",").map((t) => t.trim()).filter(Boolean);
}

export function fromSillyTavern(json) {
  const src = json?.spec === "chara_card_v2" ? json.data || {} : json || {};

  return {
    name: src.name || "",
    nickname: src.nickname || src.display_name || "",
    age: src.age || "",
    gender: src.gender || "",
    species: src.species || "",
    traits: splitTraits(src.tags || src.personality),
    description: src.description || "",
    background: src.creator_notes || src.character_book?.description || "",
    exampleDialogue: src.mes_example || "",
    greeting: src.first_mes || "",
    scenario: src.scenario || "",
    avatarData: src.avatar || "",
    avatarUrl: src.avatar?.startsWith?.("http") ? src.avatar : "",
  };
}
