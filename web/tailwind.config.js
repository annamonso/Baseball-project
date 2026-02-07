/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0e7ea3',
          light: '#19c3e6',
          dark: '#0a5f7a',
        },
        accent: '#19c3e6',
        surface: '#1f2937',
        background: '#181f25',
        predicted: '#2A9DF4',
        actual: '#E64C65',
        'field-grass': '#6B9B3C',
        'field-dirt': '#8B6914',
        success: '#22c55e',
        warning: '#f97316',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Space Grotesk', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '8px',
        lg: '12px',
      },
    },
  },
  plugins: [],
}
