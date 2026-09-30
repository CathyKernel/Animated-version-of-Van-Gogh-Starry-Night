/* ============================================================
 * The Starry Night — generative soundtrack (StarryMusic)
 * ------------------------------------------------------------
 * A gentle ambient loop synthesized live with the Web Audio API:
 * felt-piano arpeggios, a warm detuned pad, a soft bass pulse
 * and occasional glassy chimes, washed in a generated reverb.
 * Zero audio files, zero licensing, works offline.
 *
 * Browsers require a user gesture before audio may start, so
 * main.js wakes this up on the first click / key press.
 * ============================================================ */
(function (global) {
  'use strict';

  var BPM = 64;
  var BEAT = 60 / BPM;
  var BAR = BEAT * 4;
  var LOOP_BARS = 8;
  var LOOKAHEAD = 0.35;          // seconds scheduled ahead of playback

  // Dorian-flavored progression under the night sky (Am add9 feel).
  var CHORDS = [
    ['A3', 'C4', 'E4', 'B4'],    // Am(add9)
    ['F3', 'A3', 'C4', 'E4'],    // Fmaj7
    ['C4', 'E4', 'G4', 'D5'],    // C(add9)
    ['G3', 'B3', 'D4', 'F#4']    // Gmaj (raised 7th shimmer)
  ];

  var NOTE_OFFSET = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  function noteFreq(name) {
    var m = /^([A-G])([#b]?)(\d)$/.exec(name);
    if (!m) { return 440; }
    var semi = NOTE_OFFSET[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0);
    var midi = (parseInt(m[3], 10) + 1) * 12 + semi;
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function StarryMusic(ctx) {
    this.ctx = ctx;
    this.rng = mulberry32(0xC0FFEE ^ (Date.now() & 0xffff));
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.bus = ctx.createGain();
    this.bus.gain.value = 0.9;

    // generated impulse -> long, soft cathedral-ish reverb
    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this._impulse(3.4, 2.6);
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.5;

    this.bus.connect(this.master);
    this.bus.connect(this.reverb);
    this.reverb.connect(this.wet);
    this.wet.connect(this.master);
    this.master.connect(ctx.destination);

    this.playing = false;
    this.nextBar = 0;
    this.barIndex = 0;
    this.timer = null;
  }

  StarryMusic.prototype._impulse = function (seconds, decay) {
    var rate = this.ctx.sampleRate;
    var len = Math.floor(seconds * rate);
    var buf = this.ctx.createBuffer(2, len, rate);
    for (var ch = 0; ch < 2; ch++) {
      var d = buf.getChannelData(ch);
      for (var i = 0; i < len; i++) {
        d[i] = (Math.random() * 2 - 1) *
            Math.pow(1 - i / len, decay) *
            (0.55 + 0.45 * Math.sin(i / len * Math.PI));
      }
    }
    return buf;
  };

  /* ---- voices --------------------------------------------------------- */

  StarryMusic.prototype._piano = function (freq, t, dur, vel) {
    var ctx = this.ctx;
    var g = ctx.createGain();
    var osc = ctx.createOscillator();
    var osc2 = ctx.createOscillator();
    osc.type = 'triangle';
    osc2.type = 'sine';
    osc.frequency.value = freq;
    osc2.frequency.value = freq * 2.001;      // soft inharmonic felt layer
    var lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 2400;
    lp.frequency.setValueAtTime(3000, t);
    lp.frequency.exponentialRampToValueAtTime(700, t + dur);

    var o2g = ctx.createGain();
    o2g.gain.value = 0.22;
    osc.connect(lp);
    osc2.connect(o2g);
    o2g.connect(lp);
    lp.connect(g);
    g.connect(this.bus);

    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vel, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.start(t); osc2.start(t);
    osc.stop(t + dur + 0.05); osc2.stop(t + dur + 0.05);
  };

  StarryMusic.prototype._pad = function (freq, t, dur) {
    var ctx = this.ctx;
    var g = ctx.createGain();
    var o1 = ctx.createOscillator();
    var o2 = ctx.createOscillator();
    o1.type = 'sawtooth';
    o2.type = 'sawtooth';
    o1.frequency.value = freq;
    o2.frequency.value = freq * 1.004;        // slow detuned beating
    var lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 520;
    lp.Q.value = 0.4;
    o1.connect(lp); o2.connect(lp);
    lp.connect(g);
    g.connect(this.bus);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.05, t + dur * 0.35);
    g.gain.linearRampToValueAtTime(0.0001, t + dur);
    o1.start(t); o2.start(t);
    o1.stop(t + dur); o2.stop(t + dur);
  };

  StarryMusic.prototype._bass = function (freq, t) {
    var ctx = this.ctx;
    var g = ctx.createGain();
    var o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = freq / 2;
    o.connect(g);
    g.connect(this.bus);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.09, t + 0.06);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.2);
    o.start(t); o.stop(t + 2.3);
  };

  StarryMusic.prototype._chime = function (freq, t) {
    var ctx = this.ctx;
    var g = ctx.createGain();
    var o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = freq;
    var o2 = ctx.createOscillator();
    o2.type = 'sine';
    o2.frequency.value = freq * 2.76;         // bell partial
    var g2 = ctx.createGain();
    g2.gain.value = 0.18;
    o.connect(g); o2.connect(g2); g2.connect(g);
    g.connect(this.wet);                       // chimes go straight to reverb
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.055, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 3.2);
    o.start(t); o2.start(t);
    o.stop(t + 3.3); o2.stop(t + 3.3);
  };

  /* ---- scheduling ------------------------------------------------------- */

  StarryMusic.prototype._scheduleBar = function (t, bar) {
    var chord = CHORDS[Math.floor(bar / 2) % CHORDS.length];
    var rng = this.rng;
    var beat = BEAT;

    this._bass(noteFreq(chord[0]), t);

    // arpeggio: 6 notes across the bar, upward drift with random skips
    var notes = chord.concat([chord[1]]);
    for (var i = 0; i < 6; i++) {
      if (rng() < 0.16) { continue; }          // breathing space
      var idx = (i + (bar % 2)) % notes.length;
      var oct = (rng() < 0.22) ? 2 : 1;
      var f = noteFreq(notes[idx]) * oct;
      this._piano(f, t + i * beat * 0.66, 2.4 + rng() * 1.4, 0.055 + rng() * 0.03);
    }

    // pad on bars 0 and 4 of each 8-bar loop
    if (bar % 4 === 0) {
      for (var c = 0; c < chord.length; c++) {
        this._pad(noteFreq(chord[c]) / 2, t, BAR * 4);
      }
    }

    // glassy chimes: sparse, high, only on some bars
    if (rng() < 0.45) {
      var n = chord[Math.floor(rng() * chord.length)];
      this._chime(noteFreq(n) * 4, t + beat * (1 + Math.floor(rng() * 3)));
    }
    if (rng() < 0.2) {
      this._chime(noteFreq(chord[2]) * 8, t + beat * 2.5);
    }
  };

  StarryMusic.prototype._tick = function () {
    var now = this.ctx.currentTime;
    while (this.nextBar < now + LOOKAHEAD) {
      this._scheduleBar(Math.max(this.nextBar, now + 0.05), this.barIndex);
      this.barIndex = (this.barIndex + 1) % LOOP_BARS;
      this.nextBar += BAR;
    }
  };

  StarryMusic.prototype.start = function () {
    if (this.playing) { return; }
    this.playing = true;
    this.ctx.resume();
    this.nextBar = this.ctx.currentTime + 0.08;
    this.master.gain.cancelScheduledValues(this.ctx.currentTime);
    this.master.gain.setValueAtTime(this.master.gain.value, this.ctx.currentTime);
    this.master.gain.linearRampToValueAtTime(0.9, this.ctx.currentTime + 2.5);
    var self = this;
    this.timer = setInterval(function () { self._tick(); }, 120);
    this._tick();
  };

  StarryMusic.prototype.stop = function () {
    if (!this.playing) { return; }
    this.playing = false;
    var ctx = this.ctx;
    this.master.gain.cancelScheduledValues(ctx.currentTime);
    this.master.gain.setValueAtTime(this.master.gain.value, ctx.currentTime);
    this.master.gain.linearRampToValueAtTime(0.0001, ctx.currentTime + 1.2);
    clearInterval(this.timer);
  };

  global.StarryMusic = StarryMusic;
})(window);
