const PNG_SIG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c ^= bytes[i];
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
  }
  return (c ^ 0xffffffff) >>> 0;
}

function u32be(n) {
  return new Uint8Array([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]);
}

function concat(arrays) {
  const len = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(len);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

function makeTextChunk(keyword, text) {
  const encoder = new TextEncoder();
  const type = encoder.encode("tEXt");
  const data = concat([encoder.encode(keyword), new Uint8Array([0]), encoder.encode(text)]);
  const crc = crc32(concat([type, data]));
  return concat([u32be(data.length), type, data, u32be(crc)]);
}

function dataUrlToBuffer(dataUrl) {
  const base64 = dataUrl.split(",")[1] || "";
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function createBlankPngDataUrl(size = 512) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#6b7280";
  ctx.font = "bold 42px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Character Card", size / 2, size / 2);
  return c.toDataURL("image/png");
}

export async function exportPngCard(character, cardData, keyword = "chara") {
  const source = character.avatarData?.startsWith("data:image/png")
    ? character.avatarData
    : createBlankPngDataUrl();
  const original = new Uint8Array(dataUrlToBuffer(source));

  const sigOk = PNG_SIG.every((v, i) => original[i] === v);
  if (!sigOk) throw new Error("Avatar must be PNG for embedded metadata export");

  const textPayload = btoa(unescape(encodeURIComponent(JSON.stringify(cardData))));
  const chunk = makeTextChunk(keyword, textPayload);

  let ptr = 8;
  while (ptr + 8 < original.length) {
    const len = (original[ptr] << 24) | (original[ptr + 1] << 16) | (original[ptr + 2] << 8) | original[ptr + 3];
    const type = String.fromCharCode(
      original[ptr + 4],
      original[ptr + 5],
      original[ptr + 6],
      original[ptr + 7],
    );
    if (type === "IEND") {
      const before = original.slice(0, ptr);
      const after = original.slice(ptr);
      return new Blob([before, chunk, after], { type: "image/png" });
    }
    ptr += 12 + len;
  }

  throw new Error("Invalid PNG file");
}
