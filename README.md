# Fire Spread Grid Visualizer

A single-page, fully client-side fire spread simulator.

## Hosting on GitHub Pages (no backend needed)

This app now runs 100% in the browser. No Flask server is required for deployment.

### Deploy steps

1. Commit and push this repo to GitHub.
2. In GitHub, open **Settings -> Pages**.
3. Under **Build and deployment**, select:
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or your branch)
   - **Folder**: `/docs`
4. Save. GitHub Pages will publish `docs/index.html`.

Your site URL will be:

`https://<username>.github.io/<repo>/`

## Local preview

You can open `static/index.html` directly in a browser, or run a simple static server:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/docs/`.

## Notes

- The UI includes:
  - object placement and fire source selection
  - branch-and-bound-style placement solver (in JS)
  - simulation from source
  - uniform-source simulation
  - playback controls, scrubber, cumulative loss, and per-object status table
