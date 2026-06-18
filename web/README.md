# Silica landing page

Single static page. No build step, no framework, no dependencies.

- `index.html` — everything (HTML + inline CSS + inline JS)
- `fonts/martian-mono-latin.woff2` — self-hosted variable font (latin subset, ~23 KB, one request)
- `favicon.svg`, `vercel.json`

## Deploy to Vercel

```bash
cd web
vercel --prod      # or: drag this folder into the Vercel dashboard
```

Vercel auto-detects it as a static site; `vercel.json` only sets long cache headers on the font/icon.

## Local preview

```bash
python3 -m http.server -d web 8000   # then open http://localhost:8000
```
