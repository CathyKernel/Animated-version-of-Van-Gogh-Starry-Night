/* ============================================================
 * The Starry Night — page bootstrap
 * ------------------------------------------------------------
 * Loads the neural pipeline bundle (embedded scene-data.js for
 * offline / file:// use, assets/ folder over http), boots the
 * Three.js engine, wires the control panel, the generative
 * soundtrack and the quiet keyboard interactions.
 * ============================================================ */
(function () {
  'use strict';

  var DATA = window.__STARRY__;
  var canvas = document.getElementById('app');
  var loading = document.getElementById('loading');
  var fallback = document.getElementById('fallback');
  var errBox = document.getElementById('error-box');
  var musicBtn = document.getElementById('music-btn');
  var soundHint = document.getElementById('sound-hint');

  var engine = null;
  var music = null;

  function showError(msg) {
    if (errBox) {
      errBox.textContent = msg;
      errBox.hidden = false;
    }
    if (loading) { loading.classList.add('done'); }
    console.error(msg);
  }

  function showFallback() {
    if (fallback && DATA) {
      fallback.hidden = false;
      var f = DATA.files || {};
      var p = f['painting.jpg'] || f['painting.png'];
      if (p) {
        document.getElementById('fallback-img').src = 'data:image/jpeg;base64,' + p;
      }
    }
    if (loading) { loading.classList.add('done'); }
  }

  /* ---------------- asset resolution ---------------------------------- */

  function dataURL(b64, mime) {
    return 'data:' + mime + ';base64,' + b64;
  }

  function resolveSources() {
    if (DATA && DATA.files) {
      var f = DATA.files;
      return Promise.resolve({
        painting: dataURL(f['painting.jpg'], 'image/jpeg'),
        maskStars: dataURL(f['mask_stars.png'], 'image/png'),
        maskMoon: dataURL(f['mask_moon.png'], 'image/png'),
        maskTree: dataURL(f['mask_tree.png'], 'image/png'),
        maskVillage: dataURL(f['mask_village.png'], 'image/png'),
        depth: f['depth.png'] ? dataURL(f['depth.png'], 'image/png') : null,
        flow: f['flow.png'] ? dataURL(f['flow.png'], 'image/png') : null,
        elements: DATA.elements || {},
        flowScale: DATA.flowScale || 4.0,
        shaders: (DATA.shaders) || (window.__SHADERS__ || {})
      });
    }
    // http mode: read the asset registry
    return fetch('assets/manifest.json').then(function (r) { return r.json(); })
      .then(function (reg) {
        var a = 'assets/';
        return {
          painting: a + reg.assets.painting,
          maskStars: a + reg.assets.mask_stars,
          maskMoon: a + reg.assets.mask_moon,
          maskTree: a + reg.assets.mask_tree,
          maskVillage: a + reg.assets.mask_village,
          depth: reg.assets.depth ? a + reg.assets.depth : null,
          flow: reg.assets.flow ? a + reg.assets.flow : null,
          elements: reg.elements || {},
          flowScale: reg.flowScale || 4.0,
          shaders: window.__SHADERS__ || {}
        };
      });
  }

  /* ---------------- boot ------------------------------------------------ */

  function boot() {
    if (!window.WebGLRenderingContext) {
      showFallback();
      return;
    }
    resolveSources().then(function (src) {
      engine = new StarryEngine(canvas, {
        elements: src.elements,
        flowScale: src.flowScale,
        shaders: src.shaders,
        src: src,
        onError: showError,
        onStats: function (fps, w, h) {
          var f = document.getElementById('stat-fps');
          var r = document.getElementById('stat-res');
          if (f) { f.textContent = fps + ' fps'; }
          if (r) { r.textContent = w + '\u00d7' + h; }
        }
      });
      window.__engine = engine;   // debugging handle
      return engine.init();
    }).then(function () {
      if (loading) { loading.classList.add('done'); }
      // Respect visitors who prefer less motion
      if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) {
        engine.setParams({ speed: 0.35 });
      }
      wireUI();
    }).catch(function (err) {
      if (String(err && err.message || err).indexOf('WebGL') >= 0 ||
          !window.WebGLRenderingContext) {
        showFallback();
      } else {
        showError('Renderer failed: ' + (err && err.message ? err.message : err));
      }
    });
  }

  /* ---------------- panel wiring ----------------------------------------- */

  function bindRange(id, fn) {
    var el = document.getElementById(id);
    if (!el) { return; }
    el.addEventListener('input', function () {
      fn(parseInt(el.value, 10) / 100);
    });
  }

  function wireUI() {
    bindRange('ctl-parallax', function (v) { engine.setParams({ parallax: v }); });
    bindRange('ctl-flow', function (v) { engine.setParams({ flow: v }); });
    bindRange('ctl-speed', function (v) { engine.setParams({ speed: v }); });
    bindRange('ctl-sway', function (v) { engine.setParams({ sway: v }); });
    bindRange('ctl-twinkle', function (v) { engine.setParams({ twinkle: v }); });
    bindRange('ctl-moon', function (v) { engine.setParams({ moonGlow: v }); });
    bindRange('ctl-flicker', function (v) { engine.setParams({ flicker: v }); });

    var layerIds = ['sky', 'stars', 'moon', 'tree', 'village'];
    layerIds.forEach(function (name) {
      var el = document.getElementById('vis-' + name);
      if (!el) { return; }
      el.addEventListener('change', function () {
        var p = {};
        p[name] = el.checked;
        engine.setParams(p);
      });
    });

    var cruise = document.getElementById('ctl-cruise');
    if (cruise) {
      cruise.addEventListener('change', function () {
        engine.setParams({ cruise: cruise.checked });
      });
    }

    var pauseBtn = document.getElementById('ctl-pause');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', function () {
        engine.paused = !engine.paused;
        pauseBtn.textContent = engine.paused ? 'Resume' : 'Pause';
      });
    }

    var meteorBtn = document.getElementById('ctl-meteor');
    if (meteorBtn) {
      meteorBtn.addEventListener('click', function () {
        engine.summonMeteor(0.15 + Math.random() * 0.5, 0.1 + Math.random() * 0.25);
      });
    }

    var resetBtn = document.getElementById('ctl-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        engine.setParams({
          parallax: 0.55, flow: 1, speed: 1, sway: 1, twinkle: 1,
          moonGlow: 1, flicker: 1, cruise: true
        });
        ['ctl-parallax', 'ctl-flow', 'ctl-speed', 'ctl-sway',
         'ctl-twinkle', 'ctl-moon', 'ctl-flicker'].forEach(function (id) {
          var el = document.getElementById(id);
          if (el) { el.value = 100; }
        });
        if (cruise) { cruise.checked = true; }
      });
    }

    var toggle = document.getElementById('panel-toggle');
    var panel = document.getElementById('panel');
    if (toggle && panel) {
      toggle.addEventListener('click', function () {
        panel.classList.toggle('collapsed');
      });
    }
  }

  /* ---------------- sound + keys ------------------------------------------ */

  function ensureMusic() {
    if (!music) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { return null; }
      music = new StarryMusic(new AC());
    }
    return music;
  }

  function toggleMusic() {
    var m = ensureMusic();
    if (!m) { return; }
    if (m.playing) {
      m.stop();
      musicBtn.classList.add('off');
    } else {
      m.start();
      musicBtn.classList.remove('off');
    }
    if (soundHint) { soundHint.hidden = true; }
  }

  if (musicBtn) {
    musicBtn.classList.add('off');
    musicBtn.addEventListener('click', toggleMusic);
  }

  var woke = false;
  function wake() {
    if (woke) { return; }
    woke = true;
    if (soundHint) {
      soundHint.hidden = false;
      setTimeout(function () { soundHint.hidden = true; }, 6000);
    }
  }
  window.addEventListener('pointerdown', wake, { once: true });

  window.addEventListener('keydown', function (e) {
    if (e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) { return; }
    if (e.key === 'm' || e.key === 'M') { toggleMusic(); }
    if (e.key === ' ') {
      e.preventDefault();
      var pauseBtn = document.getElementById('ctl-pause');
      if (engine) {
        engine.paused = !engine.paused;
        if (pauseBtn) { pauseBtn.textContent = engine.paused ? 'Resume' : 'Pause'; }
      }
    }
    if (e.key === 'f' || e.key === 'F') {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        document.documentElement.requestFullscreen();
      }
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
