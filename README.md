# The Starry Night — Animated

An interactive, real-time WebGL tribute to Vincent van Gogh's *The Starry Night* (1889).
The painting comes alive in your browser: the sky flows, eleven stars twinkle at their
true positions, the crescent moon breathes, the great cypress sways in the wind and the
village window lights flicker like candlelight — **every color is sampled from the
painting itself, so Van Gogh's palette is never altered**.

Rebuilt and upgraded from
[CathyKernel/Animated-version-of-Van-Gogh-Starry-Night](https://github.com/CathyKernel/Animated-version-of-Van-Gogh-Starry-Night)
(original: a Python/OpenCV script that rendered an 8-second MP4). This version adds a
zero-dependency WebGL web app plus a much-improved Python video generator.

![Preview](assets/screenshot.jpg)

## What moves

| Element | Animation |
|---|---|
| **The sky** | A twin-whirlpool flow field (centered on the painting's real swirls) + curl-noise advection make the brushwork stream like a current |
| **Eleven stars + 2 swirl cores** | Detected at their real positions (including Venus, the brightest) — each twinkles with its own phase, halo breathing included |
| **The moon** | The crescent brightens and dims on a slow ~14 s cycle, its golden halo pulsing in step |
| **The cypress** | The dark flame sways in the wind: wide slow sweeps at the crown, quick tremors among the leaves, steady roots (amplitude driven by a generated mask) |
| **The village** | Thirteen window lights flicker in a two-frequency candlelight rhythm |
| **Shooting stars** | A meteor crosses the sky once in a while — or click the painting to summon one at any time |

> All colors are extracted from the original painting's pixels at load time
> (glow colors included), so the animation only changes brightness and position —
> never the hues.

## Project structure

```
animated-starry-night/
├── index.html               # entry point — works offline via file:// too
├── netlify.toml             # Netlify deploy config (static site, no build step)
├── css/
│   └── style.css            # late-night museum theme
├── js/
│   ├── scene-data.js        # element data + painting/mask embedded as base64 (generated)
│   ├── starry-engine.js     # WebGL engine: shaders + render loop
│   └── main.js              # UI wiring (controls, keyboard, click-to-meteor)
├── assets/
│   ├── starry-night.jpg     # the painting (Wikimedia Commons, 1280px)
│   ├── masks.png            # cypress sway-amplitude mask (generated)
│   ├── elements.json        # detected element coordinates (generated)
│   └── screenshot.png       # preview image for this README
└── python/
    ├── starry_night_enhanced.py   # upgraded video generator (MP4)
    └── analyze_elements.py        # element detection (generates masks + data)
```

## Run locally

No build step, no dependencies. Either:

- **Double-click `index.html`** — the painting and masks are embedded as base64,
  so it works fully offline, or
- serve it (recommended, closest to production):

```bash
cd animated-starry-night
python3 -m http.server 8000
# open http://localhost:8000
```

### Controls

| Action | How |
|---|---|
| Play / pause | Button or `Space` |
| Speed | Slider, 0.2× – 3× |
| Intensity | Global strength of every effect |
| Effect toggles | Flowing sky / stars / moon / cypress / village lights / meteors |
| Summon a shooting star | Click anywhere on the sky |
| Fullscreen | Button or `F` |
| Reset | Restore all defaults |

The opening reveal (~2.8 s) is a nod to the original project's
`smooth_unfold_effect`.

## Deploy to Netlify

The site is 100% static — deployment takes under a minute.

### Option A — drag & drop (fastest)

1. Download / unzip this folder.
2. Go to <https://app.netlify.com/drop>.
3. Drag the whole `animated-starry-night` folder onto the page.
4. Netlify publishes it immediately at a random `*.netlify.app` URL
   (you can rename it under *Site settings → Change site name*).

### Option B — GitHub + Netlify auto-deploy (recommended)

1. **Push this folder to GitHub** (see the next section).
2. In Netlify: **Add new site → Import an existing project → GitHub**,
   then pick your repository.
3. Build settings (Netlify usually fills these in automatically from
   `netlify.toml`; if not, set them manually):
   - **Branch to deploy:** `main`
   - **Build command:** *(leave empty)*
   - **Publish directory:** `.`
4. Click **Deploy**. Every future `git push` now redeploys the site automatically.

## Push to GitHub

```bash
cd animated-starry-night
git init
git add .
git commit -m "The Starry Night — animated WebGL tribute"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Or upload the folder via GitHub's web UI: *New repository → uploading an
existing file* — make sure `index.html` sits at the repository root.)

## Python video generator (optional)

The `python/` folder contains the upgraded offline renderer, for when you want
an MP4 instead of the live web app:

```bash
cd python
pip install opencv-python numpy requests pillow

python3 starry_night_enhanced.py                      # 12 s / 30 fps / 1024px
python3 starry_night_enhanced.py --duration 20 --fps 30 --size 1280 --output my.mp4
python3 starry_night_enhanced.py --no-oil --no-vignette   # cleaner look
python3 starry_night_enhanced.py --image ../assets/starry-night.jpg
```

What changed vs. the original script:

| Original function | This version |
|---|---|
| `enhance_colors_dynamically` | **Removed** — LAB saturation/contrast oscillation drifted the colors away from the painting |
| `create_starry_particles` | **Rewritten** — particles emit from the real star positions instead of random spots all over the frame |
| `create_dynamic_swirl_flow_field` | **Improved** — vortices centered on the painting's twin swirls, sky-only influence, gentler strength |
| `advanced_oil_painting` | **Vectorized** — from minutes per frame (pure Python loops) to tens of milliseconds |
| `add_glow_effect` | **Improved** — bloom-style glow sampled from the painting's own bright areas |
| `smooth_unfold_effect` | **Kept** as the intro reveal |
| — | **New** `detect_elements` + `animate_stars` / `animate_moon` / `sway_cypress` / `flicker_windows` |

## How element detection works

`python/analyze_elements.py` (the web build uses the same pipeline):

1. **Cypress** — the darkest tall olive-green blob in the left region; a
   sway-amplitude map (`masks.png`) is generated from treetop (1) to roots (0).
2. **Moon** — the largest bright blob in the upper right; the centroid of its
   V > 238 pixels marks the crescent core.
3. **Stars** — bright warm blobs in the sky (cypress excluded), with
   morphological merging so halos (Venus splits into fragments) join up.
4. **Window lights** — small warm blobs in the village band.

The result is then curated by hand/vision-model review and stored in
`assets/elements.json`.

## Technical notes

- The web animation is a single **WebGL1 fragment shader**: a displacement
  field (twin vortices + pseudo curl-noise advection + cypress sway + halo
  breathing) warps the painting's texture coordinates first, then brightness
  pulsing is applied on top.
- Glow colors are extracted at runtime from the painting's own pixels
  (mean of the brightest 30% per element), so halos always match the artwork.
- Performance: one pass, no post-processing chain — 60 FPS on ordinary hardware.
- Compatibility: Chrome / Edge / Firefox / Safari; graceful fallback to the
  static painting when WebGL is unavailable.

## Credits

- Original painting: Vincent van Gogh, *The Starry Night*, 1889 — MoMA, New York (public domain)
- Image: Wikimedia Commons (Google Art Project digitization)
- Base project: [CathyKernel/Animated-version-of-Van-Gogh-Starry-Night](https://github.com/CathyKernel/Animated-version-of-Van-Gogh-Starry-Night)
- Code in this repository: MIT License (see `LICENSE`)
