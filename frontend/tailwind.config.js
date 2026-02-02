/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
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
