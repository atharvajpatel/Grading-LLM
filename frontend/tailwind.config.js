/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // No safelist. The previous 11-hue safelist existed to protect dynamic
  // `bg-${color}-100` / `text-${color}-700` strings on the Documentation
  // tab. Those dynamic accents are gone; the palette is four colors and
  // every class is now statically analyzable.
  theme: {
    extend: {
      colors: {
        cream: '#F5F0E8',
        ink: {
          DEFAULT: '#0B1F3F',
          700: '#2A333E',
          300: '#B6BCC5',
        },
        gold: {
          DEFAULT: '#C4956A',
          deep: '#A87D55',
          badge: '#FEF3E2',
        },
        mute: '#5A5A6E',
        hair: '#E8E4DE',
        // The four grading scales, by luminance. Mirrors SCALE_COLORS
        // in src/api/client.ts and SCALE_RAMP in src/theme/palette.ts.
        scale: {
          binary: '#E0C39B',
          ternary: '#C4956A',
          quaternary: '#9A6F45',
          continuous: '#6B4A2B',
        },
      },
      fontFamily: {
        ui: ['Inter', 'system-ui', 'sans-serif'],
        body: ['"Open Sans"', 'system-ui', 'sans-serif'],
        stat: ['"Roboto Condensed"', 'Inter', 'sans-serif'],
      },
      borderRadius: {
        card: '16px',
        ctl: '8px',
      },
      boxShadow: {
        raised:
          '0 1px 0 rgba(255,255,255,0.55) inset, 0 6px 18px -8px rgba(11,31,63,0.18), 0 2px 6px -2px rgba(11,31,63,0.08)',
      },
    },
  },
  plugins: [],
}
