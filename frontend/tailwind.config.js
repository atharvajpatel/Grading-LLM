/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Statement-archetype tag chips on the Documentation tab use dynamic
  // `bg-${color}-100` / `text-${color}-700` strings. Keep the palette safelisted.
  safelist: [
    { pattern: /^bg-(blue|green|red|amber|pink|purple|orange|indigo|rose|cyan|emerald)-100$/ },
    { pattern: /^text-(blue|green|red|amber|pink|purple|orange|indigo|rose|cyan|emerald)-700$/ },
  ],
  theme: {
    extend: {
      colors: {
        'scale-binary': '#2196F3',
        'scale-ternary': '#4CAF50',
        'scale-quaternary': '#FF9800',
        'scale-continuous': '#E91E63',
      },
    },
  },
  plugins: [],
}
