function readU32(view, offset) {
  return view.getUint32(offset);
}

function decodeText(bytes) {
  return new TextDecoder().decode(bytes);
}

function maybeDecodeCard(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    try {
      return JSON.parse(decodeURIComponent(escape(atob(raw))));
    } catch {
      return null;
    }
  }
}

export async function readPngCard(file, keyword = "chara") {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  let ptr = 8;
  while (ptr + 8 < bytes.length) {
    const len = readU32(view, ptr);
    const type = decodeText(bytes.slice(ptr + 4, ptr + 8));
    const dataStart = ptr + 8;
    const dataEnd = dataStart + len;

    if (type === "tEXt") {
      const chunk = bytes.slice(dataStart, dataEnd);
      const zero = chunk.indexOf(0);
      if (zero > 0) {
        const key = decodeText(chunk.slice(0, zero));
        const value = decodeText(chunk.slice(zero + 1));
        if (key === keyword || key === "ccv3" || key === "chara") {
          const parsed = maybeDecodeCard(value);
          if (parsed) return parsed;
        }
      }
    }

    ptr += 12 + len;
  }

  return null;
}
