/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "-apple-system", "sans-serif"],
      },
      colors: {
        // Brand ramp (electric indigo → violet). Kept under the `rail` key so all
        // existing rail-* class usages re-skin automatically.
        rail: {
          50: "#eef1ff",
          100: "#e0e5ff",
          200: "#c6ccff",
          300: "#a3acff",
          400: "#7c87ff",
          500: "#5b63f5",
          600: "#4a45e6",
          700: "#3a34c2",
          800: "#2a2790",
          900: "#191a4d",
        },
        accent: {
          300: "#5eead4",
          400: "#2dd4bf",
          500: "#14b8a6",
          600: "#0d9488",
        },
        cyan: {
          300: "#67e8f9",
          400: "#22d3ee",
          500: "#06b6d4",
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
        emerald: { 400: "#34d399", 500: "#10b981", 600: "#059669" },
        gold: { 400: "#e8b84b", 500: "#d4a43a" },
        ink: "#070b16",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(124,135,255,0.20), 0 8px 40px -8px rgba(74,69,230,0.45)",
        "glow-accent": "0 0 0 1px rgba(45,212,191,0.20), 0 8px 40px -8px rgba(20,184,166,0.40)",
        card: "0 1px 2px rgba(0,0,0,0.4), 0 12px 40px -16px rgba(0,0,0,0.6)",
        "btn-primary": "0 8px 24px -6px rgba(74,69,230,0.6)",
      },
      backgroundImage: {
        "brand-gradient": "linear-gradient(135deg,#7c87ff 0%,#5b63f5 45%,#2dd4bf 120%)",
        "brand-soft": "linear-gradient(135deg,rgba(124,135,255,0.16),rgba(45,212,191,0.10))",
        "mesh":
          "radial-gradient(60% 50% at 15% 0%,rgba(74,69,230,0.22),transparent 60%),radial-gradient(50% 40% at 100% 10%,rgba(20,184,166,0.16),transparent 60%),radial-gradient(60% 60% at 80% 100%,rgba(91,99,245,0.14),transparent 60%)",
      },
      keyframes: {
        floaty: { "0%,100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-6px)" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "gradient-x": {
          "0%,100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        glowpulse: { "0%,100%": { opacity: "0.5" }, "50%": { opacity: "1" } },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        floaty: "floaty 6s ease-in-out infinite",
        "gradient-x": "gradient-x 8s ease infinite",
        glowpulse: "glowpulse 2.4s ease-in-out infinite",
        "fade-up": "fade-up 0.5s ease both",
      },
    },
  },
  plugins: [],
}
