# EEG_DEMO — Minimal End-to-End EEG Web Demo

A minimal, runnable end-to-end EEG demo:

- **Mock EEG (16 channels)** → **band power** (delta/theta/alpha/beta/gamma)
- Backend estimates **Valence / Arousal + label**
- Frontend shows:
  - Left: emotion + VA 2D plot
  - Right: waveform + 16-channel grid
  - **Resizable panels** (drag splitters), persisted in `localStorage`

## Ports

This OpenHands runtime exposes:

- Frontend HTTP: **51730**
- Backend WebSocket: **55141**

The UI also supports overriding the WS port:

- `http://localhost:51730/frontend/index.html?ws_port=55141`

## Quick start

From repo root:

```bash
./run_demo.sh
```

Then open:

- `http://localhost:51730/frontend/index.html`

(If you run with different ports, use `?ws_port=...` accordingly.)

## Configuration

### Environment variables

`run_demo.sh` and the Python servers read:

- `EEG_DEMO_HTTP_PORT` (default **51730**)
- `EEG_DEMO_WS_PORT` (default **55141**)

Example:

```bash
EEG_DEMO_HTTP_PORT=51730 EEG_DEMO_WS_PORT=55141 ./run_demo.sh
```

### Frontend query params

- `ws_port`: WebSocket port to connect to

Example:

```
http://localhost:51730/frontend/index.html?ws_port=55141
```

## How it works (repo structure)

```
EEG_DEMO/
  backend/
    server.py            # WS server: mock EEG + band power + emotion estimate
  frontend/
    index.html           # Single-page UI (no build step)
    serve_frontend.py    # Tiny static HTTP server
  outputs/logs/          # runtime logs + pid files
  run_demo.sh            # starts both servers
```

## UI usage

- Drag the **vertical splitter** between left/right panels to resize.
- Drag the **horizontal splitter** in the right panel to resize waveform vs 16ch grid.
- Layout is persisted:
  - `eeg_demo_left_w`: left panel width (px)
  - `eeg_demo_right_top_h`: right-top height (px)

## Troubleshooting

### Page loads but nothing updates

1. Open DevTools Console and check for errors.
2. Confirm WebSocket is reachable:
   - UI should show **Connected**
   - Backend should be listening on `ws://localhost:55141`

If you changed ports, ensure the URL contains `?ws_port=<your_ws_port>`.

### "Cannot access 'canvas' before initialization"

This indicates the script crashed early (canvas resize ran before canvas variables existed).
Update to the latest `frontend/index.html` in this repo; the resizable layout code is executed **after** canvas initialization.

### Check logs

Logs are written to:

- `outputs/logs/frontend_http.log`
- `outputs/logs/backend_mock.log`

PID files:

- `outputs/logs/frontend_http.pid`
- `outputs/logs/backend_mock.pid`

## Notes

- Backend emits messages with:
  - `eeg.waveform` (latest sample row)
  - `band_power.values` (per-band per-channel energy)
  - `emotion` (label, valence, arousal, confidence)
- Mock generator includes multiple rhythms + drift/noise + simple artifacts, and VA is driven by a 2D latent state so motion does not collapse to a single line.
