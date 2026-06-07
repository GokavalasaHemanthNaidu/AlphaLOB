# App Flow Document: AlphaLOB

*Note: This document reflects the actual retrospective state of the AlphaLOB repository.*

## 1. Pages and Screens
AlphaLOB is designed as a streamlined, single-page application (SPA) embedded directly into the backend, supplemented by auto-generated documentation endpoints.
- `/` — Main Dashboard & Interactive Landing Page.
- `/docs` — Auto-generated OpenAPI (Swagger) Documentation.
- `/redoc` — Auto-generated ReDoc Documentation.

## 2. First-Time Visitor Experience
A brand new visitor navigating to the root URL (`/`) immediately sees:
1. A top navigation bar showing the system status ("Checking...").
2. A hero section introducing **> AlphaLOB_** and **Low-Latency Limit Order Book Alpha Signals**.
3. A 4-column metrics grid summarizing the system's technical merits (Predictive Edge, Inference Latency, Core Engine, Market State).

## 3. Authentication Flow
**Not Applicable.** AlphaLOB is a public-facing portfolio application and technical demonstration. There is no user signup, login, or onboarding flow. API abuse is mitigated invisibly via IP-based rate limiting (30 requests/minute) rather than user authentication.

## 4. Primary User Journey: Interactive API Sandbox
The core purpose of the dashboard is to prove the API works in real-time.
1. **Discover:** The user scrolls to the "API Documentation & Sandbox" section.
2. **Review:** The user reads the description and the provided `curl` command for an endpoint (e.g., `/predict`).
3. **Action:** The user clicks the "▶ Send Request" button.
4. **Execution:** The frontend executes an asynchronous `fetch()` call to the backend.
5. **Feedback:** A hidden terminal-style `<pre>` block expands, displaying the JSON response (e.g., directional probabilities, regime state) and the latency of the request.

## 5. Secondary User Journeys
**Journey A: Pipeline Comprehension**
1. User scrolls to the "End-to-End Pipeline" section.
2. User sees a 6-step horizontal timeline.
3. User clicks on individual numbered nodes (e.g., "03 LOBTransformer").
4. A card expands below the node detailing specific completed milestones for that step (e.g., "51.25% Val Acc").

**Journey B: Formal API Exploration**
1. User navigates to `/docs`.
2. User interacts with the Swagger UI to view the detailed Pydantic schemas for `LOBSnapshot`, `PredictResponse`, etc.
3. User executes custom JSON payloads against the backend using Swagger's "Try it out" feature.

## 6. Navigation System
- **Top Navbar:** Contains anchor links (`#api`) to smooth-scroll down the page, external links (GitHub), and a dynamic status indicator. The navbar is statically positioned at the top of the page.
- **Scroll-based:** The page relies on vertical scrolling for discovering content, typical of landing pages.

## 7. Empty States & Error States
- **Empty States:** The API response terminal blocks are entirely hidden from the DOM until the user triggers a request.
- **Error States:** 
  - If the rate limit is exceeded (HTTP 429), the API returns a standard JSON error detail.
  - If the `fetch()` call fails in the browser, the catch block updates the terminal UI to visually display `Error: [message]` in red/green text to the user.

## 8. Redirect Rules
**Not Applicable.** As a pure single-page application without authentication walls or protected routes, there are no internal redirects.

## 9. Modals, Drawers, and Overlays
There are no heavy modal windows or side-drawers. The app utilizes lightweight CSS overlays:
- **Tooltips:** Hovering over specific metric labels (e.g., "Honest Ceiling" under Predictive Edge) triggers a CSS-based absolute positioned tooltip providing deep technical context without interrupting the user flow.
