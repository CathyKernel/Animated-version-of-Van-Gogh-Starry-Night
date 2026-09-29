/* ============================================================
 * The Starry Night — WebGL animation engine (StarryEngine)
 * ------------------------------------------------------------
 * Design principles:
 *   1. Every color is 100% sampled from the original painting
 *      texture — the animation only modulates brightness and
 *      displaces positions, never the painting's hues.
 *   2. Element-level animation:
 *      - Sky: twin-whirlpool flow field + curl-noise advection,
 *        the brushwork streams like a current
 *      - Stars: 13 real star positions (incl. Venus) twinkling
 *        with independent phases + breathing halos
 *      - Moon: crescent brightness breathing + radial halo pulse
 *      - Cypress: crown sway (wide at the top, steady at the
 *        roots) + high-frequency leaf tremor
 *      - Village: 13 window lights flickering like candlelight
 *        (two-frequency rhythm)
 *      - Meteor: occasional shooting stars + click-to-summon
 *   3. Pure WebGL1 (GLSL ES 1.00), zero dependencies, works when
 *      opened directly from file://
 * ============================================================ */
(function (global) {
  'use strict';

  /* ---------------- Shaders ---------------- */

  const VERT_SRC = [
    'attribute vec2 aPos;',
    'varying vec2 vUV;',
    'void main() {',
    '  vUV = aPos * 0.5 + 0.5;',   // (-1,-1)->(0,0); uv.y=0 is the canvas bottom
    '  gl_Position = vec4(aPos, 0.0, 1.0);',
    '}'
  ].join('\n');

  const FRAG_SRC = [
    '#ifdef GL_FRAGMENT_PRECISION_HIGH',
    'precision highp float;',
    '#else',
    'precision mediump float;',
    '#endif',

    'varying vec2 vUV;',

    'uniform sampler2D uPainting;',   // the original painting
    'uniform sampler2D uMasks;',      // R channel = cypress sway-amplitude map
    'uniform float uTime;',           // animation time (seconds, speed applied)
    'uniform float uIntensity;',      // global intensity 0~2
    'uniform float uIntro;',          // intro reveal 0~1
    'uniform vec4  uTgl;',            // toggles: x sky, y stars, z moon, w cypress
    'uniform vec4  uTgl2;',           // toggles: x windows, y meteor, z brush shimmer

    'const int MAX_STARS = 13;',
    'const int MAX_WINS  = 13;',
    'uniform vec4 uStars[MAX_STARS];',      // x, y, r, amp
    'uniform vec4 uStarsB[MAX_STARS];',     // glowR, glowG, glowB, speedMod
    'uniform int  uStarCount;',
    'uniform vec4 uMoon;',                   // x, y, coreR, haloR
    'uniform vec3 uMoonCol;',
    'uniform vec2 uMoonAnim;',               // speed, phase
    'uniform vec4 uVortex[2];',              // x, y, R, dir (signed angular speed)
    'uniform vec4 uWins[MAX_WINS];',         // x, y, r, amp
    'uniform vec4 uWinCol;',                 // window warm base color rgb + strength
    'uniform int  uWinCount;',
    'uniform vec4 uMeteor;',                 // x, y, dirX, dirY
    'uniform float uMeteorLife;',            // 0~1; <0 means no meteor

    /* ---- hash & noise ---- */
    'float hash21(vec2 p) {',
    '  p = fract(p * vec2(234.34, 435.345));',
    '  p += dot(p, p + 34.23);',
    '  return fract(p.x * p.y);',
    '}',
    'float vnoise(vec2 p) {',
    '  vec2 i = floor(p); vec2 f = fract(p);',
    '  f = f * f * (3.0 - 2.0 * f);',
    '  float a = hash21(i);',
    '  float b = hash21(i + vec2(1.0, 0.0));',
    '  float c = hash21(i + vec2(0.0, 1.0));',
    '  float d = hash21(i + vec2(1.0, 1.0));',
    '  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);',
    '}',

    /* ---- vortex displacement: radius-sheared oscillating rotation ---- */
    'vec2 vortexDisp(vec2 uv, vec4 v, float t) {',
    '  vec2 d = uv - v.xy;',
    '  float r = length(d);',
    '  float infl = 1.0 - smoothstep(v.z * 0.12, v.z, r);',,
    '  if (infl <= 0.001) return vec2(0.0);',
    '  float ang = v.w * (0.55 + 0.30 * sin(t * 0.33)) * infl * sin(t * 0.8 + r * 17.0);',
    '  float ca = cos(ang), sa = sin(ang);',
    '  vec2 rd = vec2(d.x * ca - d.y * sa, d.x * sa + d.y * ca);',
    '  return (rd - d) * 0.32;',
    '}',

    'void main() {',
    '  vec2 uv = vUV;',
    '  uv.y = 1.0 - uv.y;',   // flip so uv.y=0 is the top of the image (matches element coords)

    /* ============ 1. Displacement field ============ */
    '  float cyp = texture2D(uMasks, uv).r;',                      // cypress sway amplitude
    '  float skyness = 1.0 - smoothstep(0.42, 0.62, uv.y);',       // upper-sky weight
    '  skyness *= (1.0 - cyp);',
    '  float vgn = 1.0 - smoothstep(0.70, 1.0, length(uv - vec2(0.5, 0.45)) * 1.35);',

    '  vec2 disp = vec2(0.0);',

    /* 1a. Sky flow: pseudo curl-noise advection (brushwork slowly crawls) */
    '  if (uTgl.x > 0.5) {',
    '    vec2 fp = uv * vec2(4.5, 5.5);',
    '    float n1 = vnoise(fp + vec2(uTime * 0.11, uTime * 0.06));',
    '    float n2 = vnoise(fp + vec2(17.0 - uTime * 0.09, 9.0 + uTime * 0.05));',
    '    vec2 flow = vec2(n1 - 0.5, n2 - 0.5);',
    '    disp += flow * 0.010 * skyness * uIntensity;',
    /* cloud-band horizontal shear drift */
    '    disp.x += 0.0028 * sin(uTime * 0.22 + uv.y * 7.5) * skyness * uIntensity;',
    /* twin whirlpools */
    '    disp += vortexDisp(uv, uVortex[0], uTime) * uIntensity * skyness;',
    '    disp += vortexDisp(uv, uVortex[1], uTime * 1.25) * uIntensity * skyness;',
    '  }',

    /* 1b. Cypress sway: low-frequency swing + high-frequency leaf tremor, widest at the top */
    '  if (uTgl.w > 0.5) {',
    '    float ph = uTime * 1.15;',
    '    disp.x += cyp * 0.0085 * uIntensity * sin(ph + uv.y * 4.2);',
    '    disp.x += cyp * 0.0032 * uIntensity * sin(ph * 2.63 + uv.y * 12.0);',
    '    disp.y += cyp * 0.0022 * uIntensity * cos(ph * 0.87 + uv.y * 6.5);',
    '  }',

    /* 1c. Star halo breathing (radial push-pull) */
    '  if (uTgl.y > 0.5) {',
    '    for (int i = 0; i < MAX_STARS; i++) {',
    '      if (i >= uStarCount) break;',
    '      vec4 s = uStars[i];',
    '      vec2 d = uv - s.xy;',
    '      float r = length(d);',
    '      if (r > s.z * 3.0) continue;',
    '      float h = hash21(s.xy + 0.7);',
    '      float tw = sin(uTime * (1.2 + h * 1.8) * uStarsB[i].w + h * 6.2832);',
    '      float fall = 1.0 - smoothstep(s.z * 0.4, s.z * 2.6, r);',
    '      disp += (d / max(r, 1e-5)) * tw * fall * 0.0035 * uIntensity;',
    '    }',
    '  }',

    /* 1d. Moon halo breathing displacement */
    '  if (uTgl.z > 0.5) {',
    '    vec2 md = uv - uMoon.xy;',
    '    float mr = length(md);',
    '    float mring = sin(uTime * uMoonAnim.x + uMoonAnim.y);',
    '    float mfall = 1.0 - smoothstep(uMoon.w * 0.25, uMoon.w * 1.6, mr);',
    '    disp += (md / max(mr, 1e-5)) * mring * mfall * 0.004 * uIntensity;',
    '  }',

    /* 1e. Whole-canvas brush shimmer (subtle life in the paint texture) */
    '  if (uTgl2.z > 0.5) {',
    '    vec2 bp = uv * vec2(8.0, 10.0);',
    '    disp += vec2(vnoise(bp + uTime * 0.06) - 0.5, vnoise(bp + 31.7 - uTime * 0.05) - 0.5) * 0.0016 * uIntensity;',
    '  }',

    '  vec2 suv = uv + disp;',

    /* ============ 2. Sample the original painting ============ */
    '  vec3 col = texture2D(uPainting, clamp(suv, vec2(0.001), vec2(0.999))).rgb;',

    /* ============ 3. Brightness animation (hue preserved) ============ */

    /* 3a. Star twinkle + self-colored glow */
    '  if (uTgl.y > 0.5) {',
    '    for (int i = 0; i < MAX_STARS; i++) {',
    '      if (i >= uStarCount) break;',
    '      vec4 s = uStars[i];',
    '      vec4 sb = uStarsB[i];',
    '      vec2 d = uv - s.xy;',
    '      float r = length(d) / max(s.z, 1e-5);',
    '      if (r > 3.2) continue;',
    '      float h = hash21(s.xy + 0.7);',
    '      float tw = sin(uTime * (1.2 + h * 1.8) * sb.w + h * 6.2832);',
    '      float core = exp(-r * r * 1.1);',
    '      float halo = exp(-r * 1.35);',
    '      col *= 1.0 + s.w * tw * (core * 0.85 + halo * 0.30);',
    '      col += sb.rgb * max(tw, 0.0) * halo * 0.16 * s.w;',
    '    }',
    '  }',

    /* 3b. Moon breathing */
    '  if (uTgl.z > 0.5) {',
    '    vec2 md = uv - uMoon.xy;',
    '    float mr = length(md) / max(uMoon.w, 1e-5);',
    '    if (mr < 2.2) {',
    '      float mtw = sin(uTime * uMoonAnim.x + uMoonAnim.y);',
    '      float core = exp(-mr * mr * 9.0);',
    '      float halo = exp(-mr * 1.6);',
    '      col *= 1.0 + 0.12 * mtw * (core + halo * 0.4);',
    '      col += uMoonCol * max(mtw, 0.0) * halo * 0.10;',
    '    }',
    '  }',

    /* 3c. Village window lights (candlelight two-frequency flicker) */
    '  if (uTgl2.x > 0.5) {',
    '    for (int i = 0; i < MAX_WINS; i++) {',
    '      if (i >= uWinCount) break;',
    '      vec4 wn = uWins[i];',
    '      vec2 d = uv - wn.xy;',
    '      float r = length(d) / max(wn.z, 1e-5);',
    '      if (r > 3.5) continue;',
    '      float h = hash21(wn.xy + 3.3);',
    '      float spd = 2.4 + h * 3.6;',
    '      float fl = 0.62 * sin(uTime * spd + h * 6.2832)',
    '               + 0.38 * sin(uTime * spd * 2.83 + h * 15.7);',
    '      float core = exp(-r * r * 1.6);',
    '      float halo = exp(-r * 2.6);',
    '      col *= 1.0 + wn.w * fl * (core * 0.9 + halo * 0.5);',
    '      col += uWinCol.rgb * max(fl, 0.0) * (core * 0.34 + halo * 0.16) * uWinCol.a * wn.w;',
    '    }',
    '  }',

    /* 3d. Meteor (occasional / click-summoned) */
    '  if (uMeteorLife >= 0.0 && uTgl2.y > 0.5) {',
    '    vec2 md = uv - uMeteor.xy;',
    '    float along = dot(md, uMeteor.zw);',
    '    float perp = length(md - uMeteor.zw * along);',
    '    float lifeEnv = sin(3.14159265 * clamp(uMeteorLife, 0.0, 1.0));',
    '    float head = exp(-perp * 220.0) * exp(-abs(along) * 90.0);',
    '    float trail = exp(-perp * 130.0) * exp(max(-along, 0.0) * -11.0)',
    '                * step(-0.16, along) * step(along, 0.012);',
    '    float mi = (head * 0.9 + trail * 0.5) * lifeEnv * skyness * vgn;',
    '    col += vec3(1.0, 0.95, 0.82) * mi * 0.85;',
    '  }',

    /* ============ 4. Intro reveal (a nod to the original smooth_unfold) ============ */
    '  if (uIntro < 1.0) {',
    '    float dvy = abs(uv.y - 0.44);',
    '    float wave = 0.012 * sin(uv.x * 9.0 + 2.0) * (1.0 - uIntro);',
    '    float edge = uIntro * 0.78;',
    '    float reveal = 1.0 - smoothstep(edge - 0.10 + wave, edge + wave, dvy);',
    '    col *= 0.22 + 0.78 * reveal;',
    '    col += vec3(1.0, 0.9, 0.7) * exp(-abs(dvy - edge) * 42.0) * 0.13 * (1.0 - uIntro);',
    '  }',

    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');

  /* ---------------- Engine class ---------------- */

  const MAX_STARS = 13;
  const MAX_WINS = 13;

  class StarryEngine {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {Object} opts
     *   opts.elements    element data (contents of elements.json)
     *   opts.paintingSrc painting dataURL / URL
     *   opts.masksSrc    cypress mask dataURL / URL
     *   opts.onReady     called when initialization finishes
     *   opts.onError     error callback
     */
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.opts = opts || {};
      this.gl = null;
      this.program = null;
      this.running = false;
      this.animTime = 0;          // animation time (speed applied)
      this.clock = 0;             // real time (meteor scheduling)
      this.lastTs = 0;
      this.introT = 0;            // intro progress
      this.params = {
        intensity: 1.0,
        speed: 1.0,
        playing: true,
        toggles: {
          sky: true, stars: true, moon: true, cypress: true,
          windows: true, meteor: true, grain: true
        }
      };
      this.meteor = null;         // {x,y,dx,dy,t,dur}
      this.nextMeteorAt = 6 + Math.random() * 8;
      this.uniforms = {};
      this.fps = 0;
      this._fpsAcc = 0; this._fpsN = 0; this._fpsLast = 0;
    }

    /* ---------- WebGL setup ---------- */
    async init() {
      const gl = this.canvas.getContext('webgl', { antialias: false, alpha: false })
        || this.canvas.getContext('experimental-webgl', { antialias: false, alpha: false });
      if (!gl) throw new Error('WebGL is not supported by this browser');
      this.gl = gl;

      this.program = this._buildProgram(VERT_SRC, FRAG_SRC);
      gl.useProgram(this.program);

      // fullscreen quad
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
      ]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(this.program, 'aPos');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

      // textures
      const [painting, masks] = await Promise.all([
        this._loadTexture(this.opts.paintingSrc),
        this._loadTexture(this.opts.masksSrc)
      ]);
      this.paintingImg = painting.image;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, painting.tex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, masks.tex);

      this._collectUniforms();
      this._setupElementUniforms();
      this.resize();

      this.canvas.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        this.stop();
        if (this.opts.onError) this.opts.onError('The WebGL context was lost — please reload the page');
      });

      if (this.opts.onReady) this.opts.onReady();
    }

    _buildProgram(vs, fs) {
      const gl = this.gl;
      const mk = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
          throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(sh));
        }
        return sh;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, mk(gl.VERTEX_SHADER, vs));
      gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error('Shader link failed: ' + gl.getProgramInfoLog(prog));
      }
      return prog;
    }

    _loadTexture(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const gl = this.gl;
          const tex = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          resolve({ tex, image: img });
        };
        img.onerror = () => reject(new Error('Texture failed to load: ' + String(src).slice(0, 60)));
        img.src = src;
      });
    }

    _collectUniforms() {
      const gl = this.gl, p = this.program;
      ['uPainting', 'uMasks', 'uTime', 'uIntensity', 'uIntro', 'uTgl', 'uTgl2',
        'uStars', 'uStarsB', 'uStarCount', 'uMoon', 'uMoonCol', 'uMoonAnim',
        'uVortex', 'uWins', 'uWinCol', 'uWinCount', 'uMeteor', 'uMeteorLife'
      ].forEach((name) => { this.uniforms[name] = gl.getUniformLocation(p, name); });
      gl.uniform1i(this.uniforms.uPainting, 0);
      gl.uniform1i(this.uniforms.uMasks, 1);
    }

    /* ---------- Extract each element's own color from the painting pixels (keeps glows faithful) ---------- */
    _setupElementUniforms() {
      const gl = this.gl, u = this.uniforms, el = this.opts.elements;

      // offscreen canvas for pixel reading
      const c = document.createElement('canvas');
      c.width = this.paintingImg.naturalWidth;
      c.height = this.paintingImg.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(this.paintingImg, 0, 0);
      const W = c.width, H = c.height;

      const sampleColor = (x, y, r) => {
        // average of the brightest 30% of pixels in the circle -> element core color
        const x0 = Math.max(0, (x - r) * W | 0), x1 = Math.min(W, (x + r) * W | 0);
        const y0 = Math.max(0, (y - r) * H | 0), y1 = Math.min(H, (y + r) * H | 0);
        const data = ctx.getImageData(x0, y0, x1 - x0, y1 - y0).data;
        const px = [];
        for (let i = 0; i < data.length; i += 4) {
          px.push([data[i], data[i + 1], data[i + 2], (data[i] + data[i + 1] + data[i + 2]) / 3]);
        }
        if (!px.length) return [1, 1, 1];
        px.sort((a, b) => b[3] - a[3]);
        const top = px.slice(0, Math.max(1, px.length * 0.3 | 0));
        let r_ = 0, g_ = 0, b_ = 0;
        top.forEach(p => { r_ += p[0]; g_ += p[1]; b_ += p[2]; });
        return [r_ / top.length / 255, g_ / top.length / 255, b_ / top.length / 255];
      };

      // --- stars ---
      const stars = (el.stars || []).slice(0, MAX_STARS);
      const sArr = new Float32Array(MAX_STARS * 4);
      const sArrB = new Float32Array(MAX_STARS * 4);
      stars.forEach((s, i) => {
        sArr.set([s.x, s.y, s.r, s.amp != null ? s.amp : 0.34], i * 4);
        const col = sampleColor(s.x, s.y, s.r * 0.9);
        sArrB.set([col[0], col[1], col[2], 1.0], i * 4);   // speedMod=1; the shader hashes per-star speed
      });
      gl.uniform4fv(u.uStars, sArr);
      gl.uniform4fv(u.uStarsB, sArrB);
      gl.uniform1i(u.uStarCount, stars.length);

      // --- moon ---
      const m = el.moon;
      const mcol = sampleColor(m.x, m.y, m.r);
      gl.uniform4f(u.uMoon, m.x, m.y, m.r, m.halo);
      gl.uniform3f(u.uMoonCol, mcol[0], mcol[1], mcol[2]);
      gl.uniform2f(u.uMoonAnim, m.speed || 0.45, m.phase || 0);

      // --- whirlpools ---
      (el.vortices || []).slice(0, 2).forEach((v, i) => {
        gl.uniform4f(u.uVortex[i], v.x, v.y, v.r, (v.dir || 1) * (v.speed || 0.6));
      });

      // --- window lights ---
      const wins = (el.windows || []).slice(0, MAX_WINS);
      const wArr = new Float32Array(MAX_WINS * 4);
      wins.forEach((wn, i) => {
        wArr.set([wn.x, wn.y, wn.r, 0.6], i * 4);
      });
      gl.uniform4fv(u.uWins, wArr);
      gl.uniform1i(u.uWinCount, wins.length);
      // window warm color: average of all window core colors (faithful to the painting)
      let wr = 0, wg = 0, wb = 0;
      wins.forEach(wn => {
        const cc = sampleColor(wn.x, wn.y, wn.r);
        wr += cc[0]; wg += cc[1]; wb += cc[2];
      });
      if (wins.length) { wr /= wins.length; wg /= wins.length; wb /= wins.length; }
      gl.uniform4f(u.uWinCol, wr, wg, wb, 1.0);

      gl.uniform4f(u.uMeteor, 0, 0, 0, 0);
      gl.uniform1f(u.uMeteorLife, -1);
    }

    /* ---------- Parameters ---------- */
    setParams(p) {
      Object.assign(this.params, p);
    }
    setToggle(key, val) {
      this.params.toggles[key] = val;
    }

    /* ---------- Sizing ---------- */
    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.gl.viewport(0, 0, w, h);
    }

    /* ---------- Meteors ---------- */
    spawnMeteor(x, y) {
      const ang = (0.55 + Math.random() * 0.5) * (Math.random() < 0.5 ? 1 : -1); // roughly 30-60 degrees
      const dir = [Math.sin(ang), Math.abs(Math.cos(ang))];  // falling downward
      const len = 0.22 + Math.random() * 0.12;
      const x0 = x != null ? x : (0.15 + Math.random() * 0.7);
      const y0 = y != null ? y : (0.06 + Math.random() * 0.22);
      this.meteor = {
        x: x0, y: y0,
        dx: dir[0] * (len / 1.2), dy: dir[1] * (len / 1.2),
        t: 0, dur: 1.05 + Math.random() * 0.5
      };
    }

    /* ---------- Render loop ---------- */
    start() {
      if (this.running) return;
      this.running = true;
      this.lastTs = performance.now();
      const loop = (ts) => {
        if (!this.running) return;
        const dt = Math.min((ts - this.lastTs) / 1000, 0.05);
        this.lastTs = ts;
        this.clock += dt;
        if (this.params.playing) this.animTime += dt * this.params.speed;
        this.introT = Math.min(1, this.introT + dt / 2.8);
        this._updateMeteor(dt);
        this._render();
        // FPS
        this._fpsAcc += dt; this._fpsN++;
        if (this._fpsAcc >= 0.5) {
          this.fps = Math.round(this._fpsN / this._fpsAcc);
          this._fpsAcc = 0; this._fpsN = 0;
          if (this.opts.onFPS) this.opts.onFPS(this.fps);
        }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
    stop() { this.running = false; }

    _updateMeteor(dt) {
      if (this.meteor) {
        this.meteor.t += dt;
        if (this.meteor.t >= this.meteor.dur) this.meteor = null;
      } else if (this.params.toggles.meteor && this.params.playing) {
        if (this.clock >= this.nextMeteorAt) {
          this.spawnMeteor();
          this.nextMeteorAt = this.clock + 14 + Math.random() * 18;
        }
      }
    }

    _render() {
      const gl = this.gl, u = this.uniforms, p = this.params, t = this.animTime;
      const ease = this.introT < 0.5
        ? 4 * this.introT ** 3
        : 1 - Math.pow(-2 * this.introT + 2, 3) / 2;

      gl.uniform1f(u.uTime, t);
      gl.uniform1f(u.uIntensity, p.intensity);
      gl.uniform1f(u.uIntro, ease);
      const tg = p.toggles;
      gl.uniform4f(u.uTgl, tg.sky ? 1 : 0, tg.stars ? 1 : 0, tg.moon ? 1 : 0, tg.cypress ? 1 : 0);
      gl.uniform4f(u.uTgl2, tg.windows ? 1 : 0, tg.meteor ? 1 : 0, tg.grain ? 1 : 0, 0);

      if (this.meteor) {
        const m = this.meteor, prog = Math.min(m.t / m.dur, 1);
        const ml = Math.hypot(m.dx, m.dy) || 1;
        gl.uniform4f(u.uMeteor, m.x + m.dx * prog, m.y + m.dy * prog, m.dx / ml, m.dy / ml);
        gl.uniform1f(u.uMeteorLife, prog);
      } else {
        gl.uniform1f(u.uMeteorLife, -1);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    /* Click on canvas -> image coordinates (used to summon meteors) */
    clientToUV(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: (clientX - rect.left) / rect.width,
        y: (clientY - rect.top) / rect.height
      };
    }
  }

  global.StarryEngine = StarryEngine;
})(typeof window !== 'undefined' ? window : this);
