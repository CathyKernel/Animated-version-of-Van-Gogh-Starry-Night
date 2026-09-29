/* ============================================================
 * The Starry Night — procedural soundtrack (StarryMusic)
 * ------------------------------------------------------------
 * A gentle, dreamy loop generated live with the Web Audio API:
 * felt-piano style arpeggios, warm detuned pads, a soft bass
 * and occasional glassy chimes, all washed in a long reverb.
 *
 * Zero audio files, zero copyright, works offline and on any
 * static host (Netlify / GitHub Pages / file://).
 *
 * Browsers block autoplay, so the page starts the music on
 * the first user interaction (see main.js).
 * ============================================================ */
(function (global) {
  'use strict';

  /* ---------- helpers ---------- */

  const NOTE_OFFSET = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  function noteFreq(name) {
    const m = /^([A-G])([#b]?)(\d)$/.exec(name);
    if (!m) return 440;
    const semi = NOTE_OFFSET[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0);
    const midi = (parseInt(m[3], 10) + 1) * 12 + semi; // C4 = 60
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* 8-bar loop, 64 BPM — a slow "music box" breathing pace */
  const BPM = 64;
  const BEAT = 60 / BPM;          // 0.9375 s
  const BAR = BEAT * 4;           // 3.75 s
  const EIGHTH = BEAT / 2;        // 0.46875 s

  /* Dreamy diatonic progression in C major */
  const PROGRESSION = [
    { root: 'C3', pad: ['E4', 'G4', 'B4', 'D5'], arp: ['C4', 'E4', 'G4', 'B4', 'D5', 'B4', 'G4', 'E4'] },
    { root: 'A2', pad: ['C4', 'E4', 'G4', 'B4'], arp: ['A3', 'C4', 'E4', 'G4', 'B4', 'G4', 'E4', 'C4'] },
    { root: 'F2', pad: ['A3', 'C4', 'E4', 'G4'], arp: ['F3', 'A3', 'C4', 'E4', 'G4', 'E4', 'C4', 'A3'] },
    { root: 'G2', pad: ['B3', 'D4', 'E4', 'A4'], arp: ['G3', 'B3', 'D4', 'E4', 'A4', 'E4', 'D4', 'B3'] },
    { root: 'D3', pad: ['F3', 'A3', 'C4', 'E4'], arp: ['D3', 'F3', 'A3', 'C4', 'E4', 'C4', 'A3', 'F3'] },
    { root: 'E3', pad: ['G3', 'B3', 'D4', 'E4'], arp: ['E3', 'G3', 'B3', 'D4', 'E4', 'D4', 'B3', 'G3'] },
    { root: 'F2', pad: ['A3', 'C4', 'E4', 'G4'], arp: ['F3', 'A3', 'C4', 'E4', 'G4', 'E4', 'C4', 'A3'] },
    { root: 'G2', pad: ['B3', 'D4', 'G4', 'A4'], arp: ['G3', 'B3', 'D4', 'G4', 'A4', 'G4', 'D4', 'B3'] }
  ];

  /* sparse melody — C major pentatonic, always consonant */
  const MELODY = ['C5', 'D5', 'E5', 'G5', 'A5', 'C6', 'E6', 'G5', 'A5', 'D5'];

  class StarryMusic {
    constructor(opts) {
      this.playing = false;
      this.ctx = null;
      this.master = null;
      this.bus = null;
      this.conv = null;
      this.wet = null;
      this._timer = null;
      this._barIndex = 0;
      this._nextBarTime = 0;
      this._rng = mulberry32(20260929);
      this.volume = (opts && opts.volume != null) ? opts.volume : 0.85;
    }

    /* ---------- public API ---------- */

    async start() {
      if (this.playing) return;
      if (!this.ctx) this._initCtx();
      if (this.ctx.state === 'suspended') {
        try { await this.ctx.resume(); } catch (e) { /* ignore */ }
      }
      this.playing = true;
      const now = this.ctx.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setValueAtTime(this.master.gain.value, now);
      this.master.gain.linearRampToValueAtTime(this.volume, now + 2.6); // slow fade-in
      this._startScheduler();
    }

    async stop() {
      if (!this.playing) return;
      this.playing = false;
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
      const now = this.ctx.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setValueAtTime(this.master.gain.value, now);
      this.master.gain.linearRampToValueAtTime(0, now + 1.1); // gentle fade-out
      const ctx = this.ctx;
      setTimeout(() => { if (!this.playing && ctx.state === 'running') ctx.suspend(); }, 1400);
    }

    async toggle() {
      if (this.playing) await this.stop();
      else await this.start();
    }

    setVolume(v) {
      this.volume = v;
      if (this.ctx && this.playing) {
        const now = this.ctx.currentTime;
        this.master.gain.linearRampToValueAtTime(v, now + 0.4);
      }
    }

    /* ---------- audio graph ---------- */

    _initCtx() {
      const AC = global.AudioContext || global.webkitAudioContext;
      if (!AC) throw new Error('Web Audio API is not supported');
      const ctx = this.ctx = new AC({ latencyHint: 'playback' });

      // master chain: master -> gentle lowpass -> soft compressor -> speakers
      this.master = ctx.createGain();
      this.master.gain.value = 0;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass'; lp.frequency.value = 5200; lp.Q.value = 0.3;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -20; comp.knee.value = 12;
      comp.ratio.value = 2.5; comp.attack.value = 0.01; comp.release.value = 0.3;
      this.master.connect(lp); lp.connect(comp); comp.connect(ctx.destination);

      // dry bus
      this.bus = ctx.createGain();
      this.bus.gain.value = 0.9;
      this.bus.connect(this.master);

      // reverb (generated impulse response)
      this.conv = ctx.createConvolver();
      this.conv.buffer = this._makeIR(3.2, 2.4);
      this.wet = ctx.createGain();
      this.wet.gain.value = 0.38;
      this.bus.connect(this.conv);
      this.conv.connect(this.wet);
      this.wet.connect(this.master);
    }

    _makeIR(seconds, decay) {
      const rate = this.ctx.sampleRate;
      const len = Math.max(1, Math.floor(seconds * rate));
      const buf = this.ctx.createBuffer(2, len, rate);
      for (let ch = 0; ch < 2; ch++) {
        const d = buf.getChannelData(ch);
        for (let i = 0; i < len; i++) {
          const t = i / rate;
          const env = Math.pow(1 - i / len, decay) * Math.exp(-3.0 * t);
          d[i] = (Math.random() * 2 - 1) * env * 0.5;
        }
      }
      return buf;
    }

    _out(node, pan) {
      // route a voice through an optional stereo panner into the bus
      if (this.ctx.createStereoPanner) {
        const p = this.ctx.createStereoPanner();
        p.pan.value = Math.max(-1, Math.min(1, pan || 0));
        node.connect(p);
        p.connect(this.bus);
      } else {
        node.connect(this.bus);
      }
    }

    /* ---------- voices ---------- */

    /* felt-piano style pluck: sine + soft 2nd/3rd partials, quick attack */
    _pluck(freq, t, vel, dur, pan) {
      const ctx = this.ctx;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0004, t + dur);

      const f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = 2500; f.Q.value = 0.4;

      const o1 = ctx.createOscillator(); o1.type = 'sine'; o1.frequency.value = freq;
      const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 2;
      const g2 = ctx.createGain(); g2.gain.value = 0.32;
      const o3 = ctx.createOscillator(); o3.type = 'triangle'; o3.frequency.value = freq * 3.004;
      const g3 = ctx.createGain(); g3.gain.value = 0.09;

      o1.connect(g);
      o2.connect(g2); g2.connect(g);
      o3.connect(g3); g3.connect(g);
      g.connect(f);
      this._out(f, pan);

      const tEnd = t + dur + 0.08;
      [o1, o2, o3].forEach(o => { o.start(t); o.stop(tEnd); });
    }

    /* glassy chime for the sparse melody: brighter, longer, more reverb */
    _chime(freq, t, vel, dur, pan) {
      const ctx = this.ctx;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0003, t + dur);

      const o1 = ctx.createOscillator(); o1.type = 'sine'; o1.frequency.value = freq;
      const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 2.01;
      const g2 = ctx.createGain(); g2.gain.value = 0.18;
      const o3 = ctx.createOscillator(); o3.type = 'sine'; o3.frequency.value = freq * 3.02;
      const g3 = ctx.createGain(); g3.gain.value = 0.06;

      o1.connect(g);
      o2.connect(g2); g2.connect(g);
      o3.connect(g3); g3.connect(g);
      this._out(g, pan);
      // extra reverb send for the chimes
      const send = ctx.createGain(); send.gain.value = 0.9;
      g.connect(send); send.connect(this.conv);

      const tEnd = t + dur + 0.1;
      [o1, o2, o3].forEach(o => { o.start(t); o.stop(tEnd); });
    }

    /* warm detuned pad note, slow attack & release */
    _pad(freq, t, dur, level) {
      const ctx = this.ctx;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(level, t + 1.4);
      g.gain.setValueAtTime(level, t + dur - 0.6);
      g.gain.linearRampToValueAtTime(0, t + dur + 1.1);

      const f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = 1100; f.Q.value = 0.2;

      const o1 = ctx.createOscillator(); o1.type = 'triangle'; o1.frequency.value = freq; o1.detune.value = -5;
      const o2 = ctx.createOscillator(); o2.type = 'triangle'; o2.frequency.value = freq; o2.detune.value = 5;
      o1.connect(g); o2.connect(g);
      g.connect(f);
      f.connect(this.bus);

      const tEnd = t + dur + 1.3;
      [o1, o2].forEach(o => { o.start(t); o.stop(tEnd); });
    }

    /* soft sine bass */
    _bass(freq, t, vel, dur) {
      const ctx = this.ctx;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + 0.06);
      g.gain.exponentialRampToValueAtTime(0.0004, t + dur);

      const o1 = ctx.createOscillator(); o1.type = 'sine'; o1.frequency.value = freq;
      const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 2;
      const g2 = ctx.createGain(); g2.gain.value = 0.18;
      o1.connect(g);
      o2.connect(g2); g2.connect(g);
      g.connect(this.bus);

      const tEnd = t + dur + 0.05;
      [o1, o2].forEach(o => { o.start(t); o.stop(tEnd); });
    }

    /* ---------- scheduler (lookahead pattern) ---------- */

    _startScheduler() {
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
      this._nextBarTime = this.ctx.currentTime + 0.15;
      const tick = () => {
        if (!this.playing) return;
        while (this._nextBarTime < this.ctx.currentTime + 0.5) {
          this._scheduleBar(this._nextBarTime, this._barIndex);
          this._barIndex++;
          this._nextBarTime += BAR;
        }
      };
      this._timer = setInterval(tick, 120);
      tick();
    }

    _scheduleBar(t, idx) {
      const bar = idx % PROGRESSION.length;
      const ch = PROGRESSION[bar];
      const r = this._rng;

      // pad chord — slow attack, overlapping release
      ch.pad.forEach((n, i) => {
        this._pad(noteFreq(n), t + i * 0.02, BAR + 0.7, 0.028 - i * 0.002);
      });

      // bass: root on beat 1, soft fifth on beat 3
      this._bass(noteFreq(ch.root), t + 0.01, 0.11, BAR * 0.9);
      if (r() < 0.7) {
        const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const rootSemi = NOTE_OFFSET[ch.root[0]] + (ch.root[1] === '#' ? 1 : ch.root[1] === 'b' ? -1 : 0);
        const fifth = names[(rootSemi + 7) % 12];
        const oct = parseInt(ch.root.slice(-1), 10) + (rootSemi + 7 >= 12 ? 1 : 0);
        this._bass(noteFreq(fifth + oct), t + BEAT * 2, 0.055, BEAT * 1.4);
      }

      // arpeggio — felt piano, gently humanized, alternate stereo sides
      ch.arp.forEach((n, i) => {
        if (r() < 0.08) return; // small breaths keep it organic
        const tt = t + i * EIGHTH + (r() - 0.5) * 0.018;
        const swell = 0.72 + 0.28 * Math.sin((i / ch.arp.length) * Math.PI);
        const vel = 0.16 * swell * (0.9 + r() * 0.2);
        const pan = ((i % 2) * 2 - 1) * 0.22 + (r() - 0.5) * 0.1;
        this._pluck(noteFreq(n), tt, vel, 1.9, pan);
      });

      // sparse melody — glassy chimes over the harmony
      if ((bar === 2 || bar === 5 || bar === 7) && r() < 0.62) {
        const n = MELODY[(r() * MELODY.length) | 0];
        const off = EIGHTH * (2 + ((r() * 4) | 0));
        this._chime(noteFreq(n), t + off, 0.05, 3.2, (r() - 0.5) * 0.7);
        if (r() < 0.35) {
          const n2 = MELODY[(r() * MELODY.length) | 0];
          this._chime(noteFreq(n2), t + off + EIGHTH * 2, 0.04, 2.8, (r() - 0.5) * 0.7);
        }
      }
    }
  }

  global.StarryMusic = StarryMusic;
})(typeof window !== 'undefined' ? window : this);
