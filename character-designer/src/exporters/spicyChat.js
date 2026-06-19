export function toSpicyChat(character) {
  return {
    version: 1,
    platform: "spicychat",
    name: character.name || "Unnamed",
    display_name: character.nickname || character.name || "Unnamed",
    age: character.age || "",
    gender: character.gender || "",
    species: character.species || "",
    personality: (character.traits || []).join(", "),
    description: character.description || "",
    backstory: character.background || "",
    first_message: character.greeting || "",
    scenario: character.scenario || "",
    example_dialogue: character.exampleDialogue || "",
    avatar: character.avatarData || character.avatarUrl || "",
    tags: character.traits || [],
  };
}
