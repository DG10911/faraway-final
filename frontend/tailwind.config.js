/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "-apple-system", "sans-serif"],
      },
      colors: {
        rail: {
          50: "#fdf2f4",
          100: "#fce7eb",
          200: "#f9d6dc",
          300: "#e88ba0",
          400: "#d4607a",
          500: "#c84a6a",
          600: "#b33d5c",
          700: "#9a2f4a",
          800: "#7a1f38",
          900: "#4a1525",
        },
        danger: {
          50: "#fff1f2",
          100: "#ffe4e6",
          200: "#fecdd3",
          300: "#fda4af",
          400: "#fb7185",
          500: "#f43f5e",
          600: "#e11d48",
          700: "#be123c",
          800: "#9f1239",
          900: "#881337",
        },
        emerald: {
          400: "#34d399",
          500: "#009767",
          600: "#007a52",
        },
        gold: {
          400: "#e8b84b",
          500: "#d4a43a",
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
}
