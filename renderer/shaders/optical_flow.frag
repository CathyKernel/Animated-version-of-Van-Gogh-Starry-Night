// =====================================================================
// optical_flow.frag — RAFT flow advection module (GLSL ES 1.00 chunk)
// ---------------------------------------------------------------------
// Concatenated by the exporter into the head of depth_parallax.frag.
// The RAFT stage measured the dense motion of the designed animation
// and packed it into an 8-bit RG texture:
//     u = (R * 2 - 1) * uFlowScale   (pixels)
//     v = (G * 2 - 1) * uFlowScale
// This module turns that measured field into living brushwork:
//   - flowAdvect()   streams UVs along the measured motion
//   - flowCurl()     spins small eddies where the field rotates
//   - curlNoise()    organic noise whose rotation follows the flow
//   - hash/vnoise    value-noise primitives
// Brightness and position only — the painting's hues never change.
// =====================================================================

// ---- value-noise primitives -----------------------------------------
float hash21(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ---- decoded RAFT field ----------------------------------------------
// uv: texture coordinates of the painting; returns pixel-space flow.
vec2 decodeFlow(vec2 uv, sampler2D flowTex, float uFlowScale) {
  vec2 rg = texture2D(flowTex, uv).rg;
  return (rg * 2.0 - 1.0) * uFlowScale;
}

// ---- brushwork streaming ----------------------------------------------
// Time-integrated wobble along the measured motion: the brush strokes
// visibly drift with the swirl, then return — a closed-loop oscillation
// so the painting never walks away from its true self.
vec2 flowAdvect(vec2 uv, sampler2D flowTex, float uFlowScale,
                float uTime, float uStrength) {
  if (uStrength < 1e-4) return vec2(0.0);
  vec2 f = decodeFlow(uv, flowTex, uFlowScale);
  // Normalize to a smooth direction field, keep magnitude as energy.
  float m = length(f);
  if (m < 1e-3) return vec2(0.0);
  vec2 dir = f / m;
  float energy = min(m / 5.0, 1.0);
  float phase = uTime * (0.55 + 0.45 * vnoise(uv * 7.0 + uTime * 0.03));
  float wave = sin(phase * 6.2831853) * 0.5 + 0.5;   // 0..1 breathing
  return dir * energy * uStrength * (0.35 + 0.65 * wave) * 0.028;
}

// ---- flow curl: where the measured field rotates, spin the strokes ----
float flowCurl(vec2 uv, sampler2D flowTex, float uFlowScale, float uAspect) {
  vec2 e = vec2(1.5 / 512.0, 1.5 / 512.0);
  vec2 f0 = decodeFlow(uv, flowTex, uFlowScale);
  vec2 fx = decodeFlow(uv + vec2(e.x, 0.0), flowTex, uFlowScale);
  vec2 fy = decodeFlow(uv + vec2(0.0, e.y), flowTex, uFlowScale);
  float dudy = (fy.y - f0.y) / e.y;
  float dvdx = (fx.x - f0.x) / e.x;
  return dudy - dvdx;
}

// ---- curl noise (divergence-free advection currents) -------------------
vec2 curlNoise(vec2 p, float uTime) {
  float e = 0.12;
  float n1 = vnoise(p + vec2(0.0, e) + uTime * 0.05);
  float n2 = vnoise(p - vec2(0.0, e) + uTime * 0.05);
  float n3 = vnoise(p + vec2(e, 0.0) + uTime * 0.05);
  float n4 = vnoise(p - vec2(e, 0.0) + uTime * 0.05);
  return vec2(n1 - n2, n4 - n3) / (2.0 * e);
}
