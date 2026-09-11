const DEFAULT_MAX_LEN = 25;

export function abbreviateName(name: string, maxLen: number = DEFAULT_MAX_LEN): string {
  if (!name || name.length <= maxLen) return name;
  return name.slice(0, maxLen - 3) + "...";
}