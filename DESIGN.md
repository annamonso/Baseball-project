# Design System: Baseball Analytics (Stitch)

**Theme:** Dark mode with teal accents - inspired by stadium night games and modern data visualization.

## 1. Visual Theme & Atmosphere

**Mood:** Professional, futuristic, analytical
**Aesthetic:** Dark dashboard with blueprint-inspired backgrounds and vibrant data accents
**Density:** Comfortable spacing with clear visual hierarchy

The design evokes the atmosphere of a night game under stadium lights, combining the precision of data science with modern sports analytics visualization.

## 2. Color Palette & Roles

### Primary Colors
| Token | Value | Usage |
|-------|-------|-------|
| Primary | `#0e7ea3` | Primary brand color, navigation active states |
| Accent | `#19c3e6` | Interactive elements, links, focus states |
| Background | `#181f25` | Page background |
| Surface | `#1f2937` | Card backgrounds, elevated surfaces |

### Data Visualization Colors
| Token | Value | Usage |
|-------|-------|-------|
| Predicted | `#2A9DF4` | Predicted values, electric blue |
| Actual | `#E64C65` | Actual values, crimson |
| Field Grass | `#6B9B3C` | Baseball field grass |
| Field Dirt | `#8B6914` | Baseball field dirt/infield |

### Outcome Colors
| Token | Value | Usage |
|-------|-------|-------|
| Out | `#6b7280` | Out outcome |
| Single | `#19c3e6` | Single hit |
| Double | `#2A9DF4` | Double hit |
| Triple | `#a855f7` | Triple hit |
| Home Run | `#f97316` | Home run |

### Text Colors
| Token | Value | Usage |
|-------|-------|-------|
| Text Primary | `#f8fafc` | Primary text on dark backgrounds |
| Text Secondary | `#94a3b8` | Secondary text, labels |
| Text Muted | `#64748b` | Muted text, captions |

### Semantic Colors
| Token | Value | Usage |
|-------|-------|-------|
| Success | `#22c55e` | Success states, positive indicators |
| Warning | `#f97316` | Warnings, important metrics |
| Danger | `#ef4444` | Errors, negative indicators |

## 3. Typography Rules

### Font Family
- **Primary:** 'Space Grotesk', system-ui, sans-serif
- **Monospace:** 'JetBrains Mono', monospace (for statistics)

### Type Scale
| Level | Size | Weight | Usage |
|-------|------|--------|-------|
| Display | 3rem (48px) | 700 | Hero headers |
| H1 | 2.25rem (36px) | 700 | Page titles |
| H2 | 1.5rem (24px) | 600 | Section headers |
| H3 | 1.25rem (20px) | 600 | Card titles |
| Body | 1rem (16px) | 400 | Body text |
| Small | 0.875rem (14px) | 400 | Captions, labels |
| XSmall | 0.75rem (12px) | 500 | Badges, tooltips |

### Character
- Headers: Tight tracking (-0.025em)
- Body: Normal tracking (0)
- Statistics: Tabular numbers, monospace

## 4. Component Stylings

### Buttons
- **Primary Button:**
  - Background: Primary (`#0e7ea3`)
  - Text: White
  - Shape: Rounded corners (8px radius)
  - Padding: 12px 24px
  - Hover: Brighten to accent (`#19c3e6`)

- **Secondary Button:**
  - Background: Transparent
  - Border: 1px Surface lighter (`#374151`)
  - Text: Text Primary (`#f8fafc`)
  - Hover: Surface background

### Cards/Containers
- Background: Surface (`#1f2937`)
- Border: 1px solid `#374151` (subtle)
- Corner roundness: 12px
- Padding: 24px

### Inputs/Forms
- Background: `#111827`
- Border: 1px `#374151`
- Focus border: Accent (`#19c3e6`)
- Border radius: 8px
- Padding: 10px 14px
- Text: Text Primary (`#f8fafc`)
- Placeholder: Text Muted (`#64748b`)

### Navigation (Bottom)
- Background: Surface (`#1f2937`)
- Border-top: 1px `#374151`
- Icon size: 24px
- Active: Accent color (`#19c3e6`)
- Inactive: Text Secondary (`#94a3b8`)

## 5. Layout Principles

### Spacing Scale
| Token | Value | Usage |
|-------|-------|-------|
| xs | 4px | Tight inline spacing |
| sm | 8px | Related elements |
| md | 16px | Default spacing |
| lg | 24px | Section padding |
| xl | 32px | Major sections |
| 2xl | 48px | Page margins |

### Grid
- Max content width: 1280px
- Gutter: 24px
- Card grid: 2 columns on desktop, 1 on mobile

### Blueprint Background Pattern
Used on hero sections for visual interest:
```css
.blueprint-bg {
  background-image:
    linear-gradient(rgba(14, 126, 163, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(14, 126, 163, 0.03) 1px, transparent 1px);
  background-size: 20px 20px;
}
```

## 6. Baseball-Specific Elements

### Strike Zone Visualization
- Background: Dark (`#111827`)
- Zone border: 2px Accent (`#19c3e6`)
- Grid lines: `#374151`, dashed
- Ball location: Predicted color (`#2A9DF4`) with glow effect
- Label: "Catcher's Perspective" in small text

### Baseball Field Visualization
- Grass: Field Grass (`#6B9B3C`)
- Dirt/Infield: Field Dirt (`#8B6914`)
- Lines: White (`#ffffff`)
- Ball trajectory: Accent with gradient

### Probability Displays
- Progress bars with rounded ends
- Fill: Primary (`#0e7ea3`) or gradient
- Background: `#111827`
- Include percentage label

### Stat Display
- Use monospace font for numbers
- Large, bold values
- Muted labels above values
- Accent color for highlighted stats

## 7. Icons

Using Material Symbols Rounded:
- Navigation: `home`, `sports_baseball`, `query_stats`
- Weight: 400
- Size: 24px for nav, 20px inline

## 8. Tailwind Configuration

```javascript
// tailwind.config.js
export default {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0e7ea3',
          light: '#19c3e6',
        },
        accent: '#19c3e6',
        surface: '#1f2937',
        background: '#181f25',
        predicted: '#2A9DF4',
        actual: '#E64C65',
        'field-grass': '#6B9B3C',
        'field-dirt': '#8B6914',
      },
      fontFamily: {
        sans: ['Space Grotesk', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
}
```

## 9. CSS Variables

```css
:root {
  --color-primary: #0e7ea3;
  --color-accent: #19c3e6;
  --color-background: #181f25;
  --color-surface: #1f2937;
  --color-text: #f8fafc;
  --color-text-secondary: #94a3b8;
  --color-predicted: #2A9DF4;
  --color-actual: #E64C65;
}
```
