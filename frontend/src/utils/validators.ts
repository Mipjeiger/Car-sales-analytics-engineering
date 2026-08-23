const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"];

export function isValidImage(file: File): boolean {
  if (IMAGE_TYPES.includes(file.type)) return true;
  return /\.(jpe?g|png|webp|gif|bmp|tiff)$/i.test(file.name);
}

export function isNonEmpty(value: string): boolean {
  return value.trim().length > 0;
}

export function toNumberOrNull(value: string): number | null {
  if (value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
