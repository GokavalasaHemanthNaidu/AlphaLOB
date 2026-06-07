# UI/UX Design Brief: AlphaLOB

*Note: This document is written retrospectively based on a full audit of the HTML, CSS, and Tailwind configuration embedded in `src/api/main.py`.*

## 1. Overall Aesthetic
- **Style:** Technical, high-performance, and terminal-inspired.
- **Vibe:** Geared towards quants and developers. It uses raw data aesthetics, micro-animations (like a pulsing cursor `> AlphaLOB_`), and technical borders to mimic a modern trading terminal.

## 2. Theme Mode
- **Primary Mode:** **Strict Dark Mode.** 
- The `<html>` tag is hardcoded with `class="dark"`. There is no light mode toggle. The interface is specifically designed for low-light trading/monitoring environments.

## 3. Color Palette
The exact theme tokens are extracted directly from the injected Tailwind config and inline classes:

| Role | Color Name / HEX | Usage / Context |
|---|---|---|
| **Background** | `background` (#0d1117) | The deepest layer of the page. |
| **Surface** | `gray-800/30` / `#182028` | Elevated cards, API sandbox blocks, pipeline cards. |
| **Text (Primary)** | `on-background` (#dae3ee) | Main body text and standard headings. |
| **Text (Muted)** | `gray-400` / `gray-500` | Subtitles, labels, disabled states. |
| **Accent (Success)**| `emerald-400/500` | Positive metrics, "LIVE" badges, API online status. |
| **Accent (Action)** | `blue-400/500` | Primary buttons, active glow effects, ONNX Runtime branding. |
| **Accent (Warning)**| `amber-400/500` | Regime detection actions, "Waking up..." status. |
| **Accent (Danger)** | `red-400/500` | Error states and failed API requests. |

## 4. Typography
- **Primary Font:** `JetBrains Mono` (Google Fonts).
- **Usage:** Used universally. By overriding *all* text classes (`title-sm`, `headline-md`, `body-sm`, etc.) to use `JetBrains Mono`, the app achieves a rigid, data-heavy, terminal-like consistency.
- **Terminal output:** Monospaced terminal output is heavily utilized in the `<pre>` tags for the API Sandbox.

## 5. Border Radius
- **Style:** Sharp to slightly-rounded.
- **Tailwind Config Rules:**
  - `DEFAULT`: `0.125rem` (2px)
  - `lg`: `0.25rem` (4px)
  - `xl`: `0.5rem` (8px)
- **Usage:** Metrics cards and API boxes use subtle rounding to soften the edges, but maintain a highly structured, rectangular feel.

## 6. Shadows & Elevation
- **Style:** Flat by default, with subtle neon "glows" on interaction.
- **Rule:** Cards do not cast traditional drop shadows. Instead, hovering over a metric card (`hover:-translate-y-1 hover:shadow-blue-500/10`) creates a subtle blue neon glow under the card, emphasizing the "technical" aesthetic.

## 7. Visual References (Inferred)
1. **Modern Developer Tooling:** Vercel / Railway / Stripe Docs (Clean API sandboxes, dark themes, precise spacing).
2. **Trading Terminals:** Bloomberg Terminal / TradingView (Dense information, monospaced numbers, stark contrast).

## 8. Key UI Components
- **Metrics Grid Cards:** 4-column layout displaying key KPIs with bottom "sparkline" progress bars and interactive tooltip labels.
- **Pipeline Timeline:** An interactive timeline where user clicks toggle the visibility of milestone cards.
- **API Sandbox:** Interactive terminal blocks (`<pre>`) that execute `fetch` commands and stream JSON responses.
- **Status Badges:** Small pill-shaped indicators (e.g., `[● LIVE]`) used for system health.

## 9. Accessibility (a11y)
- **Contrast:** The dark mode `#0d1117` background against `#dae3ee` text mathematically yields a **14.60:1 contrast ratio**, easily exceeding strict WCAG AAA compliance standards (7:1).
- **Semantic HTML:** Relies heavily on `<header>`, `<main>`, `<section>`, and `<footer>` tags for screen-reader navigation.
- **Tooltips:** Hover-based CSS tooltips provide critical context without cluttering the screen.

## 10. Mobile Responsiveness
- **Status:** Fully Responsive.
- **Mechanism:** Standard Tailwind breakpoints (`md:grid-cols-2`, `lg:grid-cols-4`) collapse the grids.
- **Custom Rule:** A specific CSS `@media (max-width: 768px)` query converts the horizontal pipeline connectors (`.flow-line`) into a vertical stack design for smaller screens.

---

## Do's and Don'ts

### ✅ DO
- **DO** use `JetBrains Mono` for all new text.
- **DO** use subtle hover glows (`hover:shadow-[color]/10`) to indicate interactivity.
- **DO** present data using monospace styling so decimals and numbers align vertically.
- **DO** keep API responses wrapped in raw `<pre>` blocks to maintain the developer-first feel.

### ❌ DON'T
- **DON'T** introduce light mode styling. It breaks the high-frequency trading aesthetic.
- **DON'T** use heavy, generic drop shadows.
- **DON'T** use highly rounded, "bubbly" border radii (e.g., `rounded-full` on cards). Keep edges sharp (`0.125rem`).
- **DON'T** rely on modals that trap the user's focus. Use lightweight tooltips and inline expanding divs.
