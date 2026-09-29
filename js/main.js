/* ============================================================
 * The Starry Night — interactive animation, page logic
 * Loads embedded assets -> boots the WebGL engine -> binds UI
 * ============================================================ */
(function () {
  'use strict';

  const DATA = window.__STARRY__;
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const canvas = $('#starry-canvas');
  const stage = $('#stage');
  const loading = $('#loading');
  const fallback = $('#fallback');
  const playBtn = $('#btn-play');
  const playLabel = playBtn ? playBtn.querySelector('span') : null;
  const speedSlider = $('#slider-speed');
  const speedVal = $('#speed-val');
  const intensitySlider = $('#slider-intensity');
  const intensityVal = $('#intensity-val');
  const fpsEl = $('#fps');
  const errBox = $('#error-box');

  let engine = null;

  function showError(msg) {
    if (errBox) {
      errBox.textContent = msg;
      errBox.style.display = 'block';
    }
    if (loading) loading.style.display = 'none';
    console.error(msg);
  }

  function fmtPct(v) { return Math.round(v * 100) + '%'; }

  async function boot() {
    if (!window.WebGLRenderingContext) {
      fallback.style.display = 'flex';
      fallback.querySelector('img').src = DATA.paintingData;
      if (loading) loading.style.display = 'none';
      return;
    }
    try {
      engine = new StarryEngine(canvas, {
        elements: DATA.elements,
        paintingSrc: DATA.paintingData,
        masksSrc: DATA.masksData,
        onFPS: (f) => { if (fpsEl) fpsEl.textContent = f + ' FPS'; },
        onError: showError
      });
      await engine.init();
      engine.start();

      // Hide the loading overlay
      if (loading) {
        loading.classList.add('done');
        setTimeout(() => loading.remove(), 700);
      }

      bindControls();
      bindKeyboard();
      window.addEventListener('resize', () => engine.resize());
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) engine.stop();
        else engine.start();
      });
    } catch (e) {
      showError('The animation engine failed to start: ' + e.message +
        ' (please try the latest Chrome, Edge or Safari)');
      fallback.style.display = 'flex';
      fallback.querySelector('img').src = DATA.paintingData;
    }
  }

  /* ---------------- Controls ---------------- */
  function bindControls() {
    // Play / pause
    playBtn.addEventListener('click', () => {
      const p = engine.params;
      p.playing = !p.playing;
      playBtn.setAttribute('aria-label', p.playing ? 'Pause' : 'Play');
      playBtn.setAttribute('aria-pressed', String(p.playing));
      if (playLabel) playLabel.textContent = p.playing ? 'Pause' : 'Play';
      playBtn.classList.toggle('paused', !p.playing);
    });

    // Speed
    speedSlider.addEventListener('input', () => {
      const v = parseFloat(speedSlider.value);
      engine.setParams({ speed: v });
      speedVal.textContent = v.toFixed(1) + '×';
    });

    // Intensity
    intensitySlider.addEventListener('input', () => {
      const v = parseFloat(intensitySlider.value);
      engine.setParams({ intensity: v });
      intensityVal.textContent = fmtPct(v);
    });

    // Effect toggles
    $$('.chip[data-toggle]').forEach((chip) => {
      chip.addEventListener('click', () => {
        const key = chip.dataset.toggle;
        const on = !chip.classList.contains('off');
        chip.classList.toggle('off', on);   // invert on click
        chip.setAttribute('aria-pressed', String(!on));
        engine.setToggle(key, !on);
      });
    });

    // Reset
    $('#btn-reset').addEventListener('click', () => {
      engine.setParams({ speed: 1.0, intensity: 1.0 });
      engine.params.playing = true;
      speedSlider.value = '1.0';
      intensitySlider.value = '1';
      speedVal.textContent = '1.0×';
      intensityVal.textContent = '100%';
      playBtn.classList.remove('paused');
      playBtn.setAttribute('aria-label', 'Pause');
      playBtn.setAttribute('aria-pressed', 'true');
      if (playLabel) playLabel.textContent = 'Pause';
      $$('.chip[data-toggle]').forEach((chip) => {
        chip.classList.remove('off');
        chip.setAttribute('aria-pressed', 'true');
      });
      const tg = engine.params.toggles;
      Object.keys(tg).forEach((k) => { tg[k] = true; });
      engine.animTime = 0;
      engine.introT = 0;
    });

    // Fullscreen
    $('#btn-full').addEventListener('click', () => {
      if (!document.fullscreenElement) {
        (stage.requestFullscreen || stage.webkitRequestFullscreen).call(stage);
      } else {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      }
    });

    // Click on the painting -> summon a shooting star
    canvas.addEventListener('click', (e) => {
      if (!engine.params.toggles.meteor) return;
      const uv = engine.clientToUV(e.clientX, e.clientY);
      if (uv.y < 0.62) engine.spawnMeteor(uv.x, Math.min(uv.y, 0.5));
    });
  }

  function bindKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.code === 'Space') {
        e.preventDefault();
        playBtn.click();
      } else if (e.key === 'f' || e.key === 'F') {
        $('#btn-full').click();
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
