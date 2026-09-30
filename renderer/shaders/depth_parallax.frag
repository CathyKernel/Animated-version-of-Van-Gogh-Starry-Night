// =====================================================================
// depth_parallax.frag — layer-aware neural composite (GLSL ES 1.00)
// ---------------------------------------------------------------------
// The main shader of the Van Gogh neural renderer. It composites the
// five SAM semantic layers back-to-front, each displaced by its own
// depth-scaled parallax, and animates every element the way the
// painting asks to be animated:
//
//   sky      per-pixel depth parallax + twin whirlpool rotation +
//            RAFT-flow brush advection + curl-noise night current
//   stars    halo spin around every detected star + twinkle
//   moon     halo rotation + crescent breathing glow
//   village  window lights flickering like candlelight
//   tree     cypress sway, crown sweeping, roots steady
//
// All layer colors come from ONE shared painting texture; layers only
// contribute alpha masks. Hues are therefore never altered — only
// positions and brightness, exactly as the project specification
// demands ("preserves original brush strokes and colors").
//
// The RAFT flow module (optical_flow.frag) is concatenated above this
// source by the exporter.
// =====================================================================

precision highp float;

varying vec2 vUV;

uniform sampler2D uPainting;   // shared RGB of the painting
uniform sampler2D uMaskStars;  // alpha masks (R channel)
uniform sampler2D uMaskMoon;
uniform sampler2D uMaskTree;
uniform sampler2D uMaskVillage;
uniform sampler2D uDepth;      // fused depth, near = white
uniform sampler2D uFlow;       // RAFT flow, RG-encoded

uniform float uTime;           // animation time (speed already applied)
uniform float uAspect;         // canvas w / h (= painting ratio ~1.2623)
uniform vec2  uParallax;       // camera offset, roughly -1..1
uniform float uParallaxAmt;    // parallax strength 0..1
uniform float uFlowStrength;   // RAFT brush advection strength 0..2
uniform float uTwinkle;        // star twinkle 0..2
uniform float uWinFlicker;     // window candlelight 0..2
uniform float uMoonGlow;       // moon breathing glow 0..2
uniform float uIntro;          // intro reveal 0..1
uniform float uIntensity;      // global brightness 0.4..1.6

uniform float uVisSky;
uniform float uVisStars;
uniform float uVisMoon;
uniform float uVisTree;
uniform float uVisVillage;

const int MAX_STARS = 16;
const int MAX_WINS  = 20;
uniform vec4  uStars[MAX_STARS];   // x, y, r, amp
uniform float uStarAng[MAX_STARS]; // current spin angle of each halo
uniform int   uStarCount;
uniform vec4  uVortices[2];        // x, y, r, dir(signed)
uniform vec2  uSwirlAng;           // rigid rotation angle of vortex 0/1
uniform vec4  uMoon;               // x, y, coreR, haloR
uniform float uMoonAng;            // halo rotation angle
uniform float uMoonBreath;         // breathing phase -> scale 0.97..1.03
uniform vec4  uWins[MAX_WINS];     // x, y, r, speed
uniform int   uWinCount;
uniform vec2  uCyp;                // cypress top, bottom (v coords)
uniform float uSway;               // cypress sway strength 0..2
uniform vec4  uMeteor;             // x, y, dirX, dirY (uv space)
uniform float uMeteorLife;         // 0..1, <0 = none
uniform float uFlowScale;          // RAFT texture scale (pixels)

// Layer depth priors from the SAM/depth stage (0 far .. 1 near).
const float D_SKY     = 0.08;
const float D_STARS   = 0.30;
const float D_MOON    = 0.42;
const float D_VILLAGE = 0.74;
const float D_TREE    = 1.00;

// ---- helpers ---------------------------------------------------------

// Mirror-wrap a uv coordinate: small overshoots past the texture edge
// reflect back inward instead of smearing the border column (clamp).
vec2 wrapUV(vec2 uv) {
  vec2 m = mod(uv, 2.0);            // fold into [0, 2)
  m = min(m, 2.0 - m);              // mirror into [0, 1]
  return clamp(m, vec2(0.0015), vec2(0.9985));
}

vec2 rot2(vec2 v, float a) {
  float c = cos(a), s = sin(a);
  return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Per-layer parallax: nearer layers counter-shift more, far less.
vec2 layerShift(float depth) {
  return uParallax * uParallaxAmt * (depth - 0.5) * 0.030;
}

// Rigid rotation displacement around a center with soft falloff.
// Aspect-corrected; the displacement vanishes as ang -> 2*pi so the
// loop is seamless. (See spinDisp of the original study.)
vec2 spinDisp(vec2 uv, vec2 center, float radius, float ang,
              float inner, float outer, float strength) {
  if (ang < 1e-4 && ang > -1e-4) return vec2(0.0);
  vec2 d = vec2((uv.x - center.x) * uAspect, uv.y - center.y);
  float r = length(d);
  float fall = 1.0 - smoothstep(radius * inner, radius * outer, r);
  if (fall <= 0.002) return vec2(0.0);
  vec2 rd = rot2(d, ang);
  vec2 disp = (rd - d) * fall * strength;
  return vec2(disp.x / uAspect, disp.y);
}

// Radial breathing displacement (the moon's halo expands and contracts).
vec2 radialDisp(vec2 uv, vec2 center, float radius, float scale,
                float strength) {
  vec2 d = vec2((uv.x - center.x) * uAspect, uv.y - center.y);
  float r = length(d);
  float fall = exp(-(r / radius) * (r / radius));
  vec2 disp = d * (scale - 1.0) * fall * strength;
  return vec2(disp.x / uAspect, disp.y);
}

// Deterministic per-index hash (twinkle phases, window phases).
float idxHash(float i) {
  return fract(sin(i * 127.1) * 43758.5453);
}

void main() {
  vec2 uv = vUV;
  vec2 P;  // working UV per layer
  vec3 col = vec3(0.0);
  float t = uTime;

  // ---------------- 1. SKY (opaque background) -------------------------
  {
    vec2 uvS = uv;
    // Depth-aware parallax, per pixel: MiDaS/SAM fused depth says how
    // far each patch of sky hangs behind the window.
    float d = texture2D(uDepth, clamp(uv, vec2(0.0), vec2(1.0))).r;
    uvS += uParallax * uParallaxAmt * (d - 0.5) * 0.030;
    // Twin whirlpools genuinely rotate (angles from JS, 2*pi-wrapped).
    for (int i = 0; i < 2; i++) {
      vec4 v = uVortices[i];
      float ang = uSwirlAng[i] * v.w;
      uvS += spinDisp(uvS, v.xy, v.z, ang, 0.10, 1.05, 1.0);
    }
    // RAFT-measured brush streaming + curl-noise night current.
    uvS -= flowAdvect(uvS, uFlow, uFlowScale, t, uFlowStrength);
    uvS += curlNoise(uv * 6.0, t) * 0.0016 * uFlowStrength;
    vec3 sky = texture2D(uPainting, wrapUV(uvS)).rgb;
    // Soft luminous ripple locked to the main whirlpool's turn.
    float ripple = 1.0 + 0.045 * uFlowStrength *
        sin(uSwirlAng.x * 3.0 + (uv.x + uv.y) * 18.0);
    col = sky * ripple * uVisSky;
  }

  // ---------------- 2. STARS (halos spin + twinkle) --------------------
  {
    vec2 uvL = uv + layerShift(D_STARS);
    for (int i = 0; i < MAX_STARS; i++) {
      if (i >= uStarCount) break;
      vec4 s = uStars[i];
      float dir = (idxHash(float(i) * 3.7) > 0.5) ? 1.0 : -1.0;
      uvL += spinDisp(uvL, s.xy, s.z * 2.2, uStarAng[i] * dir,
                      0.12, 1.6, s.w);
    }
    vec2 rg = texture2D(uMaskStars, wrapUV(uvL)).rg;
    float a = rg.r;
    // Twinkle: two-tone breathing per star, brightness only.
    float tw = 1.0;
    for (int i = 0; i < MAX_STARS; i++) {
      if (i >= uStarCount) break;
      vec4 s = uStars[i];
      vec2 dd = vec2((uvL.x - s.x) * uAspect, uvL.y - s.y);
      float r = length(dd);
      float fall = 1.0 - smoothstep(s.z * 0.2, s.z * 1.8, r);
      if (fall > 0.001) {
        float sp = 1.1 + idxHash(float(i)) * 1.6;
        float ph = idxHash(float(i) * 9.3) * 6.2831;
        tw += uTwinkle * 0.30 * fall *
            (0.62 * sin(t * sp + ph) + 0.38 * sin(t * sp * 2.33 + ph * 2.0));
      }
    }
    vec3 srgb = texture2D(uPainting, wrapUV(uvL)).rgb;
    col = mix(col, srgb * tw, a * uVisStars);
  }

  // ---------------- 3. MOON (rotating halo + breathing) -----------------
  {
    vec2 uvM = uv + layerShift(D_MOON);
    uvM += spinDisp(uvM, uMoon.xy, uMoon.w * 2.4, uMoonAng, 0.10, 1.7, 0.9);
    uvM += radialDisp(uvM, uMoon.xy, uMoon.w * 1.2, uMoonBreath, 1.0);
    float a = texture2D(uMaskMoon, wrapUV(uvM)).r;
    float glow = 1.0 + uMoonGlow * 0.20 *
        (sin((uMoonBreath - 1.0) * 160.0) * 0.5 + 0.5);
    vec3 mrgb = texture2D(uPainting, wrapUV(uvM)).rgb;
    col = mix(col, mrgb * glow, a * uVisMoon);
  }

  // ---------------- 4. VILLAGE (candlelit windows) ----------------------
  {
    vec2 uvV = uv + layerShift(D_VILLAGE);
    uvV -= flowAdvect(uvV, uFlow, uFlowScale, t, uFlowStrength) * 0.45;
    float a = texture2D(uMaskVillage, wrapUV(uvV)).r;
    // Two-frequency candlelight flicker over a slow warm swell.
    float fl = 1.0;
    for (int i = 0; i < MAX_WINS; i++) {
      if (i >= uWinCount) break;
      vec4 w = uWins[i];
      vec2 dd = vec2((uvV.x - w.x) * uAspect, uvV.y - w.y);
      float r = length(dd);
      float fall = 1.0 - smoothstep(w.z * 0.5, w.z * 3.2, r);
      if (fall > 0.001) {
        float ph = idxHash(float(i) * 5.1) * 6.2831;
        float sp = w.w;
        fl += uWinFlicker * 0.42 * fall *
            (0.60 * sin(t * sp + ph) + 0.40 * sin(t * sp * 2.7 + ph * 3.0));
      }
    }
    fl += uWinFlicker * 0.05 * sin(t * 0.35);  // slow warm swell
    vec3 vrgb = texture2D(uPainting, wrapUV(uvV)).rgb;
    col = mix(col, vrgb * fl, a * uVisVillage);
  }

  // ---------------- 5. TREE (cypress sways in the wind) -----------------
  {
    vec2 uvT = uv + layerShift(D_TREE);
    // Sway-amplitude ramp: treetop 1 -> roots 0 (from the SAM mask span).
    float ramp = clamp((uCyp.y - uvT.y) / max(uCyp.y - uCyp.x, 0.05), 0.0, 1.0);
    ramp = pow(ramp, 1.25);
    float sway = uSway * ramp *
        (0.0055 * sin(t * 0.7 + uvT.y * 4.2) +
         0.0028 * sin(t * 1.9 + uvT.y * 11.0));
    uvT.x += sway;
    float a = texture2D(uMaskTree, wrapUV(uvT)).r;
    vec3 trgb = texture2D(uPainting, wrapUV(uvT)).rgb;
    col = mix(col, trgb, a * uVisTree);
  }

  // ---------------- shooting star --------------------------------------
  if (uMeteorLife >= 0.0) {
    vec2 mp = uMeteor.xy + uMeteor.zw * (1.0 - uMeteorLife) * -0.34;
    vec2 d = vec2((uv.x - mp.x) * uAspect, uv.y - mp.y);
    vec2 dir = normalize(vec2(uMeteor.z * uAspect, uMeteor.w));
    float along = dot(d, dir);
    float perp = length(d - dir * along);
    float head = exp(-perp * perp * 5200.0) * exp(-max(-along, 0.0) * 9.0);
    float tail = exp(-perp * perp * 900.0) *
        smoothstep(0.0, 0.16, along) * exp(-along * 13.0);
    float life = sin(uMeteorLife * 3.14159);
    col += vec3(1.0, 0.97, 0.85) * (head * 0.9 + tail * 0.35) * life;
  }

  // ---------------- intro reveal, vignette, tone -----------------------
  {
    // Gentle unfold: brightness rises, a soft zoom settles.
    float rev = smoothstep(0.0, 1.0, uIntro);
    col *= mix(0.02, 1.0, pow(rev, 0.8));
    float r = length((vUV - 0.5) * vec2(uAspect, 1.0)) / length(vec2(uAspect, 1.0) * 0.5);
    col *= 1.0 - 0.16 * smoothstep(0.62, 1.05, r);
    col *= uIntensity;
    gl_FragColor = vec4(col, 1.0);
  }
}
