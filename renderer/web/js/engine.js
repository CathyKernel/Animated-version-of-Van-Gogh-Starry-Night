/* ============================================================
 * The Starry Night — neural rendering engine (Three.js + GLSL)
 * ------------------------------------------------------------
 * A single full-screen quad driven by depth_parallax.frag (with the
 * RAFT flow module concatenated on top). All element animation
 * angles are computed here in JavaScript and wrapped to [0, 2*pi]
 * so the motion loops forever without precision drift.
 *
 * Textures (from scene-data.js or assets/):
 *   painting.jpg   shared RGB  — every layer samples this one texture
 *   mask_*.png     per-layer alpha (stars / moon / tree / village)
 *   depth.png      SAM+MiDaS fused depth (per-pixel sky parallax)
 *   flow.png       RAFT-measured dense flow (RG-encoded)
 * ============================================================ */
(function (global) {
  'use strict';

  const TAU = Math.PI * 2;

  function StarryEngine(canvas, opts) {
    this.canvas = canvas;
    this.opts = opts || {};
    this.elements = this.opts.elements || {};
    this.flowScale = this.opts.flowScale || 4.0;
    this.shaders = this.opts.shaders || (global.__SHADERS__ || {});
    this.onError = this.opts.onError || function () {};

    // ---- params (bound to the UI panel) -----------------------------
    this.params = {
      parallax: 0.55,
      flow: 1.0,
      speed: 1.0,
      sway: 1.0,
      twinkle: 1.0,
      moonGlow: 1.0,
      flicker: 1.0,
      cruise: true,
      visSky: 1, visStars: 1, visMoon: 1, visTree: 1, visVillage: 1
    };

    // ---- animation state ---------------------------------------------
    this.clock = new THREE.Clock();
    this.animTime = 0;          // speed-scaled animation time
    this.paused = false;
    this.parallax = { x: 0, y: 0 };
    this.parallaxTarget = { x: 0, y: 0 };
    this.lastMouse = -10;       // seconds since last pointer input
    this.meteor = null;
    this.introT = 0;
    this.frameEMA = 16;
    this.qualityLadder = [1.5, 1.25, 1.0, 0.8, 0.66];
    this.qualityIdx = Math.min(2, this.qualityLadder.length - 1);
    this.fpsCounter = { frames: 0, t: 0, fps: 0 };
  }

  /* ---------------- texture loading -------------------------------- */

  StarryEngine.prototype._texFromImage = function (img) {
    var tex = new THREE.Texture(img);
    // NPOT textures in WebGL1: clamp + linear, no mipmaps.
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    return tex;
  };

  StarryEngine.prototype._loadImage = function (src) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error('image failed: ' + src.slice(0, 48))); };
      img.src = src;
    });
  };

  StarryEngine.prototype.init = function () {
    var self = this;

    if (!global.WebGLRenderingContext) {
      return Promise.reject(new Error('WebGL unavailable'));
    }

    // ---- renderer -----------------------------------------------------
    try {
      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas,
        antialias: false,
        alpha: false,
        powerPreference: 'high-performance'
      });
    } catch (e) {
      return Promise.reject(e);
    }
    this.renderer.setClearColor(0x05060a, 1);

    // ---- shaders -------------------------------------------------------
    var flowGLSL = this.shaders.flow || '';
    var mainGLSL = this.shaders.parallax || '';
    if (!mainGLSL) {
      return Promise.reject(new Error('shaders missing (renderer/shaders/*.frag)'));
    }
    var fragSrc = flowGLSL + '\n' + mainGLSL;
    var vertSrc = [
      // NOTE: `position` (vec3) and `uv` are declared by Three's shader
      // prefix — redeclaring them is a compile error on the WebGL2 path.
      'varying vec2 vUV;',
      'void main() {',
      '  vUV = uv;',
      '  gl_Position = vec4(position.xy, 0.0, 1.0);',
      '}'
    ].join('\n');

    // ---- textures -----------------------------------------------------
    var src = this.opts.src || {};   // { painting, maskStars, ... }
    var loads = {
      painting: this._loadImage(src.painting),
      stars: this._loadImage(src.maskStars),
      moon: this._loadImage(src.maskMoon),
      tree: this._loadImage(src.maskTree),
      village: this._loadImage(src.maskVillage),
      depth: src.depth ? this._loadImage(src.depth) : null,
      flow: src.flow ? this._loadImage(src.flow) : null
    };

    return Promise.all(Object.keys(loads).map(function (k) {
      return loads[k] ? loads[k].then(function (img) { return [k, img]; }) : null;
    })).then(function (pairs) {
      var imgs = {};
      pairs.forEach(function (p) { if (p) imgs[p[0]] = p[1]; });

      var uniforms = self._buildUniforms(imgs);
      self.material = new THREE.ShaderMaterial({
        vertexShader: vertSrc,
        fragmentShader: fragSrc,
        uniforms: uniforms
      });

      // ---- scene: one full-screen quad --------------------------------
      self.scene = new THREE.Scene();
      self.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      var quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), self.material);
      quad.frustumCulled = false;
      self.scene.add(quad);

      self._resize();
      global.addEventListener('resize', function () { self._resize(); });

      // ---- interaction --------------------------------------------------
      self._wireInput();
      self._loop();
      return self;
    });
  };

  /* ---------------- uniforms from the manifest ------------------------ */

  StarryEngine.prototype._buildUniforms = function (imgs) {
    var el = this.elements;
    var MAX_STARS = 16, MAX_WINS = 20;

    var stars = (el.stars || []).slice(0, MAX_STARS);
    var wins = (el.windows || []).slice(0, MAX_WINS);
    var vortices = (el.vortices || []).slice(0, 2);
    var moon = el.moon || { x: 0.926, y: 0.137, r: 0.055, halo: 0.135 };
    var cyp = el.cypress || { top: 0.05, bottom: 0.90 };

    var uStarsVal = [], uStarAngVal = [];
    for (var i = 0; i < MAX_STARS; i++) {
      var s = stars[i] || { x: -9, y: -9, r: 0.001, amp: 0 };
      uStarsVal.push(new THREE.Vector4(s.x, s.y, s.r, s.amp || 0.3));
      uStarAngVal.push(0);
    }
    var uWinsVal = [];
    for (var j = 0; j < MAX_WINS; j++) {
      var w = wins[j] || { x: -9, y: -9, r: 0.001, area: 60 };
      // candlelight speed from window size: small windows tremble faster
      var sp = 2.2 + 2.6 * (1 - Math.min(w.area, 110) / 110) + 0.9 * ((j * 37 % 11) / 11);
      uWinsVal.push(new THREE.Vector4(w.x, w.y, w.r, sp));
    }
    var uVortVal = [
      new THREE.Vector4(
        vortices[0] ? vortices[0].x : 0.42, vortices[0] ? vortices[0].y : 0.25,
        vortices[0] ? vortices[0].r : 0.30, vortices[0] ? (vortices[0].dir || 1) : 1),
      new THREE.Vector4(
        vortices[1] ? vortices[1].x : 0.63, vortices[1] ? vortices[1].y : 0.19,
        vortices[1] ? vortices[1].r : 0.20, vortices[1] ? (vortices[1].dir || -1) : -1)
    ];

    // element angular speeds: whirlpools 46 s / 62 s a turn
    this.vortexSpeeds = [TAU / 46, TAU / 62];
    this.vortexDirs = [
      vortices[0] ? (vortices[0].dir || 1) : 1,
      vortices[1] ? (vortices[1].dir || -1) : -1
    ];
    // stars 14-24 s a turn, alternating direction (deterministic)
    this.starSpeeds = [];
    for (var k = 0; k < stars.length; k++) {
      var period = 14 + (k * 53 % 11) / 11 * 10;
      this.starSpeeds.push({
        w: TAU / period,
        dir: (k % 2 === 0) ? 1 : -1
      });
    }
    this.moonSpeed = TAU / 68;
    this.moonBreathPeriod = 14;

    var self = this;
    function tex(img, fallback) {
      return { value: img ? self._texFromImage(img) : fallback || null };
    }
    var white = this._whiteTexture();

    this.uniforms = {
      uPainting: tex(imgs.painting, white),
      uMaskStars: tex(imgs.stars, white),
      uMaskMoon: tex(imgs.moon, white),
      uMaskTree: tex(imgs.tree, white),
      uMaskVillage: tex(imgs.village, white),
      uDepth: tex(imgs.depth, white),
      uFlow: tex(imgs.flow, white),

      uTime: { value: 0 },
      uAspect: { value: 1.2623 },
      uParallax: { value: new THREE.Vector2(0, 0) },
      uParallaxAmt: { value: this.params.parallax },
      uFlowStrength: { value: this.params.flow },
      uTwinkle: { value: this.params.twinkle },
      uWinFlicker: { value: this.params.flicker },
      uMoonGlow: { value: this.params.moonGlow },
      uIntro: { value: 0 },
      uIntensity: { value: 1 },

      uVisSky: { value: 1 }, uVisStars: { value: 1 }, uVisMoon: { value: 1 },
      uVisTree: { value: 1 }, uVisVillage: { value: 1 },

      uStars: { value: uStarsVal },
      uStarAng: { value: uStarAngVal },
      uStarCount: { value: stars.length },
      uVortices: { value: uVortVal },
      uSwirlAng: { value: new THREE.Vector2(0, 0) },
      uMoon: { value: new THREE.Vector4(moon.x, moon.y, moon.r, moon.halo) },
      uMoonAng: { value: 0 },
      uMoonBreath: { value: 1 },
      uWins: { value: uWinsVal },
      uWinCount: { value: wins.length },
      uCyp: { value: new THREE.Vector2(cyp.top || 0.05, cyp.bottom || 0.9) },
      uSway: { value: this.params.sway },
      uMeteor: { value: new THREE.Vector4(0, 0, 1, 0.4) },
      uMeteorLife: { value: -1 },
      uFlowScale: { value: this.flowScale }
    };
    return this.uniforms;
  };

  StarryEngine.prototype._whiteTexture = function () {
    var data = new Uint8Array([255, 255, 255, 255]);
    var t = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
    t.needsUpdate = true;
    return t;
  };

  /* ---------------- input ---------------------------------------------- */

  StarryEngine.prototype._wireInput = function () {
    var self = this;
    var cvs = this.canvas;

    function pointer(e) {
      var r = cvs.getBoundingClientRect();
      var x = ((e.clientX - r.left) / r.width) * 2 - 1;
      var y = -(((e.clientY - r.top) / r.height) * 2 - 1);
      self.parallaxTarget.x = Math.max(-1, Math.min(1, x));
      self.parallaxTarget.y = Math.max(-1, Math.min(1, y));
      self.lastMouse = self.animTime;
    }
    cvs.addEventListener('pointermove', pointer);
    cvs.addEventListener('pointerdown', function (e) {
      pointer(e);
      var r = cvs.getBoundingClientRect();
      var u = (e.clientX - r.left) / r.width;
      var v = 1 - (e.clientY - r.top) / r.height;
      self.summonMeteor(u, v);
    });
  };

  StarryEngine.prototype.summonMeteor = function (u, v) {
    var ang = Math.PI * (0.62 + Math.random() * 0.25);  // down-right sweep
    this.meteor = {
      x: Math.min(u, 0.86), y: Math.min(v, 0.72),
      dirx: Math.cos(ang), diry: -Math.abs(Math.sin(ang)),
      life: 0
    };
  };

  StarryEngine.prototype._autoMeteor = function () {
    if (Math.random() < 1 / 2600) {
      this.summonMeteor(0.12 + Math.random() * 0.6, 0.05 + Math.random() * 0.3);
    }
  };

  /* ---------------- frame loop ------------------------------------------ */

  StarryEngine.prototype._loop = function () {
    var self = this;
    requestAnimationFrame(function () { self._loop(); });

    var dt = Math.min(this.clock.getDelta(), 0.1);
    var raw = dt * 1000;

    // fps + adaptive quality
    this.frameEMA = this.frameEMA * 0.94 + raw * 0.06;
    this.fpsCounter.frames++;
    this.fpsCounter.t += dt;
    if (this.fpsCounter.t >= 0.5) {
      this.fpsCounter.fps = Math.round(this.fpsCounter.frames / this.fpsCounter.t);
      this.fpsCounter.frames = 0;
      this.fpsCounter.t = 0;
      this._adaptQuality();
      if (this.opts.onStats) {
        this.opts.onStats(this.fpsCounter.fps, this.renderer.domElement.width,
                          this.renderer.domElement.height);
      }
    }

    if (!this.paused) {
      this.animTime += dt * this.params.speed;
      this._autoMeteor();
    }
    this.introT = Math.min(1, this.introT + dt / 2.8);

    // parallax: mouse for 4 s after input, then auto-cruise resumes
    var t = this.animTime;
    if (this.params.cruise && (t - this.lastMouse) > 4) {
      this.parallaxTarget.x = 0.62 * Math.sin(t * 0.11);
      this.parallaxTarget.y = 0.45 * Math.sin(t * 0.17 + 1.3);
    }
    this.parallax.x += (this.parallaxTarget.x - this.parallax.x) * Math.min(1, dt * 3.2);
    this.parallax.y += (this.parallaxTarget.y - this.parallax.y) * Math.min(1, dt * 3.2);

    // ---- element angles (wrapped to [0, 2*pi], no drift) -------------
    var u = this.uniforms;
    u.uTime.value = t;
    u.uParallax.value.set(this.parallax.x, this.parallax.y);
    u.uParallaxAmt.value = this.params.parallax;
    u.uFlowStrength.value = this.params.flow;
    u.uTwinkle.value = this.params.twinkle;
    u.uWinFlicker.value = this.params.flicker;
    u.uMoonGlow.value = this.params.moonGlow;
    u.uSway.value = this.params.sway;
    u.uVisSky.value = this.params.visSky;
    u.uVisStars.value = this.params.visStars;
    u.uVisMoon.value = this.params.visMoon;
    u.uVisTree.value = this.params.visTree;
    u.uVisVillage.value = this.params.visVillage;
    u.uIntro.value = this.introT;

    u.uSwirlAng.value.set(
      ((t * this.vortexSpeeds[0]) % TAU + TAU) % TAU,
      ((t * this.vortexSpeeds[1]) % TAU + TAU) % TAU);

    for (var i = 0; i < this.starSpeeds.length; i++) {
      var a = (t * this.starSpeeds[i].w) % TAU;
      u.uStarAng.value[i] = (a + TAU) % TAU;
    }

    u.uMoonAng.value = ((t * this.moonSpeed) % TAU + TAU) % TAU;
    u.uMoonBreath.value = 1 + 0.03 * Math.sin(TAU * t / this.moonBreathPeriod);

    // meteor lifecycle
    if (this.meteor) {
      this.meteor.life += dt / 1.6;
      if (this.meteor.life >= 1) {
        this.meteor = null;
        u.uMeteorLife.value = -1;
      } else {
        u.uMeteor.value.set(this.meteor.x, this.meteor.y,
                            this.meteor.dirx, this.meteor.diry);
        u.uMeteorLife.value = this.meteor.life;
      }
    }

    this.renderer.render(this.scene, this.camera);
  };

  /* ---------------- sizing / quality ------------------------------------ */

  StarryEngine.prototype._resize = function () {
    var r = this.canvas.getBoundingClientRect();
    var dpr = global.devicePixelRatio || 1;
    var q = this.qualityLadder[this.qualityIdx];
    var scale = Math.min(dpr, 2) * q;
    var w = Math.max(2, Math.round(r.width * scale));
    var h = Math.max(2, Math.round(r.height * scale));
    this.renderer.setSize(w, h, false);
    this.uniforms.uAspect.value = w / h;
  };

  StarryEngine.prototype._adaptQuality = function () {
    var before = this.qualityIdx;
    if (this.frameEMA > 24 && this.qualityIdx < this.qualityLadder.length - 1) {
      this.qualityIdx++;
    } else if (this.frameEMA < 11 && this.qualityIdx > 0) {
      this.qualityIdx--;
    }
    if (before !== this.qualityIdx) {
      this._resize();
    }
  };

  StarryEngine.prototype.setParams = function (p) {
    for (var k in p) {
      if (Object.prototype.hasOwnProperty.call(p, k)) {
        this.params[k] = p[k];
      }
    }
    // visibility flags arrive as booleans from checkboxes
    var map = { sky: 'visSky', stars: 'visStars', moon: 'visMoon',
                tree: 'visTree', village: 'visVillage' };
    for (var m in map) {
      if (typeof p[m] === 'boolean') {
        this.params[map[m]] = p[m] ? 1 : 0;
      }
    }
  };

  global.StarryEngine = StarryEngine;
})(window);
