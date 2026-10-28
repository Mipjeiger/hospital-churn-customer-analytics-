/** Generic formatting helpers shared across pages. */

export const fmtNum = (x: number | null | undefined): string =>
  x != null && isFinite(x) ? x.toLocaleString("en-US", { maximumFractionDigits: 0 }) : "—";

export const fmtFloat = (x: number | null | undefined, d = 1): string =>
  x != null && isFinite(x) ? x.toFixed(d) : "—";

export const fmtPct = (x: number | null | undefined, d = 1): string =>
  x != null && isFinite(x) ? `${(x * 100).toFixed(d)}%` : "—";

export const fmtPctRaw = (x: number | null | undefined, d = 1): string =>
  x != null && isFinite(x) ? `${x.toFixed(d)}%` : "—";

/** Convert snake_case to Title Case */
export const toTitle = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** Clamp a value between min and max */
export const clamp = (v: number, min: number, max: number) =>
  Math.min(max, Math.max(min, v));

/** Return a colour from a 0-1 ratio (green → amber → red) */
export function trafficLight(ratio: number): string {
  if (ratio >= 0.8) return "#0E7C7B"; // green
  if (ratio >= 0.5) return "#E6A817"; // amber
  return "#D64545";                   // red
}

