/**
 * Single source of truth for every color the app renders.
 *
 * The palette is exactly four base colors. Everything below is a tint,
 * shade, or mix of these -- never a new hue.
 *
 *   cream #F5F0E8   page ground
 *   ink   #0B1F3F   primary text / strongest mark
 *   gold  #C4956A   the one accent
 *   mute  #5A5A6E   secondary text
 *
 * WHY this module exists: color that encodes DATA (scale identity,
 * variance magnitude, delta sign) is applied via inline `style={{}}`
 * in the tab components, so Tailwind never sees it. Restyling CSS
 * alone would leave those hexes untouched. Every data->color function
 * in the app must import from here.
 *
 * Mirrored in src/styles/globals.css (:root) and consumed by
 * SCALE_COLORS in src/api/client.ts. Keep all three in sync.
 */

export const PALETTE = {
  cream: '#F5F0E8',
  ink: '#0B1F3F',
  gold: '#C4956A',
  mute: '#5A5A6E',
  // Shades of the four above.
  goldDeep: '#A87D55',
  goldBadge: '#FEF3E2',
  hair: '#E8E4DE',
  white: '#FFFFFF',
} as const

/* ------------------------------------------------------------------ */
/* Low-level mixing                                                     */
/* ------------------------------------------------------------------ */

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ]
}

function toHex(n: number): string {
  return Math.max(0, Math.min(255, Math.round(n)))
    .toString(16)
    .padStart(2, '0')
}

const clamp01 = (t: number): number => (t < 0 ? 0 : t > 1 ? 1 : t)

/** Linear blend of two hex colors. t=0 -> a, t=1 -> b. */
export function mix(a: string, b: string, t: number): string {
  const k = clamp01(t)
  const [r1, g1, b1] = hexToRgb(a)
  const [r2, g2, b2] = hexToRgb(b)
  return `#${toHex(r1 + (r2 - r1) * k)}${toHex(g1 + (g2 - g1) * k)}${toHex(
    b1 + (b2 - b1) * k
  )}`
}

/* ------------------------------------------------------------------ */
/* Sequential ramp: magnitude / intensity                              */
/* ------------------------------------------------------------------ */

/**
 * Sequential ramp for magnitude, t in [0,1]: cream -> gold -> ink.
 *
 * Monotonic in luminance, unlike the red->yellow->green ramp it
 * replaces (which ordered by hue at near-constant brightness and so
 * vanished in greyscale and for red-green color blindness).
 * Callers pass the same normalized t they always did; only the two
 * endpoints changed, so no clamping or bucketing math moves.
 */
export function magnitudeColor(t: number): string {
  const k = clamp01(t)
  return k <= 0.5
    ? mix(PALETTE.cream, PALETTE.gold, k * 2)
    : mix(PALETTE.gold, PALETTE.ink, (k - 0.5) * 2)
}

/** Readable text color for a label sitting on magnitudeColor(t). */
export function onMagnitude(t: number): string {
  return clamp01(t) > 0.55 ? PALETTE.white : PALETTE.ink
}

/* ------------------------------------------------------------------ */
/* Diverging ramp: signed deltas                                       */
/* ------------------------------------------------------------------ */

/**
 * Diverging ramp for signed values, t in [-1,1]:
 *   negative -> navy, zero -> cream, positive -> gold.
 *
 * A single-hue ramp cannot express sign, so the two arms use the two
 * strongest palette colors. Still four colors; sign survives.
 */
export function divergingColor(t: number): string {
  const k = t < -1 ? -1 : t > 1 ? 1 : t
  return k >= 0
    ? mix(PALETTE.cream, PALETTE.gold, k)
    : mix(PALETTE.cream, PALETTE.ink, -k)
}

/** Readable text color for a label sitting on divergingColor(t). */
export function onDiverging(t: number): string {
  return Math.abs(t) > 0.55 ? PALETTE.white : PALETTE.ink
}

/* ------------------------------------------------------------------ */
/* Scale identity: the four grading scales                             */
/* ------------------------------------------------------------------ */

/**
 * Single-hue gold ramp, light -> dark, mirroring binary(2 levels) ->
 * continuous(unbounded) granularity. Distinguished by LUMINANCE, not
 * hue. Duplicated as SCALE_COLORS in src/api/client.ts, which is what
 * the 13 consumer components import; keep the two identical.
 */
export const SCALE_RAMP = {
  binary: '#E0C39B',
  ternary: '#C4956A',
  quaternary: '#9A6F45',
  continuous: '#6B4A2B',
} as const

/**
 * Second, orthogonal encoding channel for scale identity.
 *
 * Needed because PCAPlot3D separates the four scales by color ALONE:
 * every point is an identical sphere. Four tints of one hue, lit with
 * emissive material and freely rotatable, are not separable. Shape is
 * rotation-invariant and survives overlap.
 */
export const SCALE_MARKER: Record<string, 'sphere' | 'box' | 'octahedron' | 'tetrahedron'> = {
  binary: 'sphere',
  ternary: 'box',
  quaternary: 'octahedron',
  continuous: 'tetrahedron',
}

/** Dash pattern per series index, so overlapping lines stay separable. */
export const DASH_PATTERNS = ['0', '6 3', '2 3', '10 4', '4 2 1 2'] as const

export function dashFor(index: number): string {
  return DASH_PATTERNS[index % DASH_PATTERNS.length]
}

/* ------------------------------------------------------------------ */
/* Categorical series (up to 10 texts on the degradation charts)       */
/* ------------------------------------------------------------------ */

/**
 * Ten categorical series in one hue. Luminance alone cannot carry ten
 * legible categories, so callers MUST pair this with dashFor(index)
 * on line charts. Steps run gold -> ink, skipping the palest end where
 * strokes would disappear against cream.
 */
export const TEXT_PALETTE: string[] = Array.from({ length: 10 }, (_, i) =>
  mix(PALETTE.gold, PALETTE.ink, i / 9)
)

/** Categorical color for series `index`, cycling if needed. */
export function seriesColor(index: number): string {
  return TEXT_PALETTE[index % TEXT_PALETTE.length]
}

/* ------------------------------------------------------------------ */
/* Structural chart colors (grid, axes, ticks)                         */
/* ------------------------------------------------------------------ */

export const CHART = {
  grid: PALETTE.hair,
  axis: PALETTE.mute,
  tick: PALETTE.mute,
  label: PALETTE.ink,
  tooltipBg: PALETTE.white,
  tooltipBorder: PALETTE.hair,
  reference: 'rgba(11, 31, 63, 0.25)',
} as const
