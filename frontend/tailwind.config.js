/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f7f8f9",
          100: "#eceef1",
          200: "#d5d9e0",
          300: "#b0b8c6",
          400: "#8593a8",
          500: "#667590",
          600: "#525d77",
          700: "#434c61",
          800: "#394051",
          900: "#1d2433",
          950: "#121722",
        },
        // Risk is the only place amber is allowed.
        risk: {
          50: "#fff8eb",
          100: "#ffefd4",
          200: "#ffdfaa",
          300: "#ffc871",
          400: "#fba63f",
          500: "#f5821f",
          600: "#dd6012",
          700: "#b74310",
          800: "#933514",
          900: "#772c15",
          950: "#411206",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
