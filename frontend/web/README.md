# Forged Data

A standalone light-themed React frontend for **Machine Learning on Artificially Forged Data**. No backend or root project changes are required.

```sh
cd web
npm install
npm run dev
```

Open the URL Vite prints. `VITE_USE_MOCKS=true` is the default. Set `VITE_USE_MOCKS=false` and `VITE_API_URL` to connect to the API; the default base is `/api`. For a different origin, enable CORS on that API, including on the job event stream.

```sh
VITE_USE_MOCKS=false VITE_API_URL=http://localhost:8000/api npm run dev
```

## What mock mode contains

`PROJECT_BRIEF.md`, the root `README.md`, `AUDIT.md`, and `results/RESULTS.md` were not present in this workspace. Mock mode therefore shows the supplied experiment design and its reported conclusion: **0 of 10 combinations passed all frozen rules**. It shows missing measurements as unavailable, not zero. The CSV lab can profile a local file in the browser; it does not train a generator, send the file, or fabricate experiment results.

Connected mode uses the typed routes in `src/api/`. It requests correlations, projections, and distributions for the selected random seed, and follows lab jobs through server-sent events with polling as a fallback. The contract does not specify the contents of `job.result`: for charts the UI recognizes optional `class_balance: {real, synthetic}` count maps, `fidelity` and `privacy` numeric maps, and `per_column` keyed by column with comparison distribution objects. Other result shapes remain downloadable without fabricated charts.

## Checks and screenshots

```sh
npm run test
npm run build
npx playwright install chromium
npm run test:smoke
```

The smoke test opens the built site directly and writes one PNG per section at 375px and 1440px to `screenshots/`. Set `PLAYWRIGHT_BASE_URL` to test against an existing server, or `CHROMIUM_PATH` to use an installed Chromium executable.

To verify the connected-mode UI without a backend, build it against the intercepted test origin and run the same suite with `TEST_CONNECTED=1`:

```sh
VITE_USE_MOCKS=false VITE_API_URL=https://research.test/api npx vite build --outDir dist/connected
TEST_CONNECTED=1 npm run test:smoke
```

The connected test responses are artificial test inputs and are not bundled into the public app or used for the screenshots.
