function splitTraits(value) {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (!value || typeof value !== "string") return [];
  return value.split(",").map((t) => t.trim()).filter(Boolean);
}

export function fromSpicyChat(json) {
  return {
    name: json?.name || "",
    nickname: json?.display_name || "",
    age: json?.age || "",
    gender: json?.gender || "",
    species: json?.species || "",
    traits: splitTraits(json?.tags?.length ? json.tags : json?.personality),
    description: json?.description || "",
    background: json?.backstory || "",
    exampleDialogue: json?.example_dialogue || "",
    greeting: json?.first_message || "",
    scenario: json?.scenario || "",
    avatarData: json?.avatar?.startsWith?.("data:") ? json.avatar : "",
    avatarUrl: json?.avatar?.startsWith?.("http") ? json.avatar : "",
  };
}
