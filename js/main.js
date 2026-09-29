/* ============================================================
 * The Starry Night — interactive animation, page logic
 * Loads embedded assets -> boots the WebGL engine -> wires up
 * the generative soundtrack and a few quiet interactions.
 * ============================================================ */
(function () {
  'use strict';

  const DATA = window.__STARRY__;

  const canvas = document.getElementById('starry-canvas');
  const stage = document.getElementById('stage');
  const loading = document.getElementById('loading');
  const fallback = document.getElementById('fallback');
  const musicBtn = document.getElementById('music-btn');
  const soundHint = document.getElementById('sound-hint');
  const errBox = document.getElementById('error-box');

  let engine = null;
  let music = null;
  let musicStarted = false;

  function showError(msg) {
    if (errBox) {
      errBox.textContent = msg;
      errBox.style.display = 'block';
    }
    if (loading) loading.style.display = 'none';
    console.error(msg);
  }

  function showFallback() {
    if (fallback) {
      fallback.hidden = false;
      fallback.style.display = 'flex';
      fallback.querySelector('img').src = DATA.paintingData;
    }
    if (loading) loading.style.display = 'none';
  }

  async function boot() {
    if (!window.WebGLRenderingContext) {
      showFallback();
      return;
    }
    try {
      engine = new StarryEngine(canvas, {
        elements: DATA.elements,
        paintingSrc: DATA.paintingData,
        masksSrc: DATA.masksData,
        onError: showError
      });
      window.__engine = engine; // exposed for debugging / testing only
      await engine.init();

      // Respect visitors who prefer less motion
      if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) {
        engine.setParams({ speed: 0.35 });
      }

      engine.start();

      if (loading) {
        loading.classList.add('done');
        setTimeout(() => loading.remove(), 700);
      }

      initMusic();
      bindInteractions();
    } catch (e) {
      showError('The animation engine failed to start: ' + e.message +
        ' (please try the latest Chrome, Edge or Safari)');
      showFallback();
    }
  }

  /* ---------------- Music (starts on first interaction) ---------------- */

  function setMusicUI(on) {
    if (!musicBtn) return;
    musicBtn.classList.toggle('on', on);
    musicBtn.setAttribute('aria-pressed', String(on));
    musicBtn.setAttribute('aria-label', on ? 'Pause music' : 'Play music');
    musicBtn.title = (on ? 'Pause' : 'Play') + ' music (M)';
  }

  function dismissHint() {
    if (!soundHint) return;
    soundHint.classList.add('hide');
    setTimeout(() => { if (soundHint.parentNode) soundHint.remove(); }, 1400);
  }

  function wakeMusic() {
    if (musicStarted || !window.StarryMusic) return;
    musicStarted = true;
    try {
      music = music || new StarryMusic();
      window.__music = music; // exposed for debugging / testing only
      music.start();
      setMusicUI(true);
      musicBtn.classList.add('started');
    } catch (e) {
      console.warn('Music unavailable:', e.message);
      setMusicUI(false);
    }
    dismissHint();
  }

  function toggleMusic() {
    if (!musicStarted) { wakeMusic(); return; }
    music.toggle().then(() => setMusicUI(music.playing));
  }

  function initMusic() {
    if (!musicBtn) return;

    // The first tap/click anywhere starts the soundtrack
    window.addEventListener('pointerdown', (e) => {
      if (e.target && e.target.closest && e.target.closest('#music-btn')) return; // button handles itself
      wakeMusic();
    }, { passive: true });

    musicBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleMusic();
    });
  }

  /* ---------------- Quiet interactions ---------------- */

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      (stage.requestFullscreen || stage.webkitRequestFullscreen || function () {}).call(stage);
    } else {
      (document.exitFullscreen || document.webkitExitFullscreen || function () {}).call(document);
    }
  }

  function bindInteractions() {
    // Click on the sky -> summon a shooting star
    canvas.addEventListener('click', (e) => {
      if (!engine || !engine.params.toggles.meteor) return;
      const uv = engine.clientToUV(e.clientX, e.clientY);
      if (uv.y < 0.62) engine.spawnMeteor(uv.x, Math.min(uv.y, 0.5));
    });

    // Keyboard: M music, F fullscreen, Space pause/resume the painting
    window.addEventListener('keydown', (e) => {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === 'm' || e.key === 'M') {
        toggleMusic();
      } else if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
      } else if (e.code === 'Space') {
        e.preventDefault();
        if (engine) engine.params.playing = !engine.params.playing;
      }
    });

    // Keep the painting's true aspect ratio at every size
    const onResize = () => { if (engine) engine.resize(); };
    window.addEventListener('resize', onResize);
    window.addEventListener('orientationchange', onResize);
    if (window.ResizeObserver) {
      new ResizeObserver(onResize).observe(canvas);
    }

    // Save the GPU / battery when the tab is hidden
    document.addEventListener('visibilitychange', () => {
      if (!engine) return;
      if (document.hidden) engine.stop();
      else engine.start();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
