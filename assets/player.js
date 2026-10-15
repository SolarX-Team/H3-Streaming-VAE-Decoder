/* Frame-synced video comparison, laid out like the SolarWM stage-2 showcase. */
(() => {
  "use strict";
  const { D, T, tr, h, bi, seg, onLang, whenVisible, clamp, sgn, register } = H3;

  const ICON = {
    play: '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>',
    pause: '<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zm8 0h4v16h-4z"/></svg>',
    back: '<svg viewBox="0 0 24 24"><path d="M6 5h2v14H6zm12 0v14l-9-7z"/></svg>',
    fwd: '<svg viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM6 5v14l9-7z"/></svg>',
    full: '<svg viewBox="0 0 24 24"><path d="M4 4h6v2H6v4H4zm10 0h6v6h-2V6h-4zM4 14h2v4h4v2H4zm14 0h2v6h-6v-2h4z"/></svg>',
  };
  const NAMES = {
    src: T("原视频", "Source"),
    official: T("官方 decode", "Official decode"),
    v1: T("V1 · 基础方案", "V1 · base version"),
    v2: T("V2 · 加亮度校正", "V2 · + brightness correction"),
    v3: T("V3 · 加 slot 0 重算", "V3 · + slot-0 refresh"),
    sr: T("StreamRender · 一步 2 个", "StreamRender · 2 / step"),
    sr1: T("StreamRender · 一步 1 个", "StreamRender · 1 / step"),
  };
  const RATES = [[0.25, "¼×"], [0.5, "½×"], [1, "1×"], [2, "2×"]];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function clipsOf(set) {
    if (set === "mcl") {
      return D.mcl.player.map(id => {
        const idx = D.mcl.clips.findIndex(c => c.short === id);
        const c = D.mcl.clips[idx];
        return {
          id, idx, label: c.name, dataset: "MCL-JCV", frames: c.frames, fps: c.fps, size: "1344×768",
          thumb: `media/thumbs/mcl/${id}.jpg`, file: v => `media/mcl/${v}/${id}.mp4`,
        };
      });
    }
    return D.solar.player.map(p => ({
      id: p.dataset, key: p.key, label: p.key, dataset: "SolarWM", frames: p.frames, fps: 16, size: "1344×768",
      thumb: `media/thumbs/solarwm/${p.dataset}.jpg`, file: v => `media/solarwm/${v}/${p.dataset}.mp4`,
    }));
  }

  function metric(spec, clip, v) {
    if (v === "src") return null;
    if (spec.set === "solar") {
      if (v === "official") return T("参照", "reference");
      const c = D.solar.clips.find(x => x.key === clip.key)[v];
      const t = `p95 ${c.steady_p95.toFixed(2)} · 0→1 ${sgn(c.jump01, 1)}`;
      return T(t, t);
    }
    if (spec.metric === "z1") {
      if (v === "official") return T("参照", "reference");
      const p = D.mcl.z1_psnr[clip.id][v].toFixed(1);
      return T(`z1 ${p} dB`, `z1 ${p} dB`);
    }
    const p = D.mcl.p95[v][clip.idx].toFixed(2);
    return T(`跳变 p95 ${p}`, `jump p95 ${p}`);
  }

  // Keeps several <video> elements on the same frame. The first one is the clock;
  // the others are nudged with playbackRate and re-seeked when they drift too far.
  // When any of them runs out of data, all pause until every one can play again.
  class SyncPlayer {
    constructor(onChange) {
      this.videos = [];
      this.fps = 30;
      this.frames = 1;
      this.want = false;
      this.stalled = false;
      this.rate = 1;
      this.loop = true;
      this.onChange = onChange;
      this.raf = 0;
      this.tick = this.tick.bind(this);
    }

    attach(videos, fps, frames) {
      this.videos = videos;
      this.fps = fps;
      this.frames = frames;
      for (const video of videos) {
        video.muted = true;
        video.playbackRate = this.rate;
        video.addEventListener("waiting", () => { if (this.want) this.stall(); });
        video.addEventListener("seeked", () => this.onChange());
      }
      videos[0].addEventListener("ended", () => this.ended());
    }

    detach() {
      for (const video of this.videos) {
        video.pause();
        video.removeAttribute("src");
        video.load();
      }
      this.videos = [];
      this.stalled = false;
    }

    get master() { return this.videos[0]; }

    get frame() {
      const m = this.master;
      if (!m) return 0;
      return Math.max(0, Math.min(this.frames - 1, Math.floor(m.currentTime * this.fps + 1e-3)));
    }

    ready() { return this.videos.length > 0 && this.videos.every(v => v.readyState >= 3); }

    play() {
      this.want = true;
      if (this.frame >= this.frames - 1) this.seekFrame(0);
      this.start();
      if (!this.raf) this.raf = requestAnimationFrame(this.tick);
      this.onChange();
    }

    halt() {
      this.want = false;
      this.stalled = false;
      for (const v of this.videos) v.pause();
    }

    pause() {
      this.halt();
      this.seekFrame(this.frame);
    }

    start() {
      if (!this.ready()) { this.stall(); return; }
      this.stalled = false;
      const time = this.master.currentTime;
      for (const v of this.videos) {
        if (Math.abs(v.currentTime - time) > 0.02) v.currentTime = time;
        v.playbackRate = this.rate;
        const p = v.play();
        if (p) p.catch(() => {});
      }
    }

    stall() {
      this.stalled = true;
      for (const v of this.videos) v.pause();
      this.onChange();
    }

    seekFrame(frame) {
      const target = Math.max(0, Math.min(this.frames - 1, Math.round(frame)));
      const time = (target + 0.5) / this.fps;
      for (const v of this.videos) v.currentTime = time;
      this.onChange(target);
    }

    step(delta) {
      const f = this.frame;
      if (this.want) this.halt();
      this.seekFrame(f + delta);
    }

    setRate(rate) {
      this.rate = rate;
      for (const v of this.videos) v.playbackRate = rate;
    }

    ended() {
      if (!this.want) return;
      if (this.loop) { this.seekFrame(0); this.start(); }
      else { this.halt(); this.onChange(); }
    }

    tick() {
      this.raf = 0;
      if (!this.videos.length || !this.want) return;
      this.raf = requestAnimationFrame(this.tick);
      if (this.stalled) {
        if (this.ready()) this.start();
        return;
      }
      const m = this.master;
      for (const v of this.videos.slice(1)) {
        if (v.paused && !m.paused && !v.ended) {
          const p = v.play();
          if (p) p.catch(() => {});
        }
        const drift = v.currentTime - m.currentTime;
        if (Math.abs(drift) > 0.15) {
          v.currentTime = m.currentTime;
          v.playbackRate = this.rate;
        } else if (Math.abs(drift) > 0.01) {
          v.playbackRate = this.rate * (1 - Math.max(-0.12, Math.min(0.12, drift * 2.5)));
        } else if (v.playbackRate !== this.rate) {
          v.playbackRate = this.rate;
        }
      }
      this.onChange();
    }
  }

  const players = [];
  let active = null;

  function mountPlayer(fig) {
    const spec = JSON.parse(fig.dataset.player);
    const clips = clipsOf(spec.set);
    const sources = spec.sources;
    const st = {
      clip: clips.find(c => c.id === spec.clip) || clips[0],
      mode: "side",
      rate: spec.rate ?? 1,
      loop: true,
      focus: null,
      wipeLeft: sources[sources.length - 2],
      wipeRight: sources[sources.length - 1],
      wipePos: spec.wipe ?? 0.5,
      auto: !reduced,
      live: false,
      visible: false,
    };
    const cap = fig.querySelector("figcaption");
    const add = node => fig.insertBefore(node, cap);
    let selects = [];

    const head = add(h("div", { class: "vp-head" }));
    const headL = h("div", null, head);
    const crumbs = h("div", { class: "vp-crumbs" }, headL);
    const tag = h("span", { class: "vp-tag" }, crumbs);
    const index = h("span", null, crumbs);
    const title = h("div", { class: "vp-title" }, headL);
    const facts = h("dl", { class: "vp-facts" }, head);

    const picker = add(h("div", { class: "vp-picker" }));
    const picks = clips.map(c => {
      const b = h("button", { type: "button", class: "vp-pick", title: c.label }, picker);
      h("img", { src: c.thumb, alt: "", loading: "lazy" }, b);
      h("div", { class: "nm" }, b, c.id);
      bi(h("div", { class: "mt" }, b), T(`${c.frames} 帧 · ${(c.frames / c.fps).toFixed(1)} s`, `${c.frames} frames · ${(c.frames / c.fps).toFixed(1)} s`));
      b.addEventListener("click", () => { b.blur(); select(c); });
      return b;
    });

    const body = add(h("div", { class: "vp-body" }));
    const stage = h("div", { class: "vp-stage" }, body);
    const buffering = bi(h("div", { class: "buffering" }), T("缓冲中…", "Buffering…"));
    const controls = h("div", { class: "controls" }, body);
    const row = h("div", { class: "btn-row" }, controls);
    const bBack = h("button", { type: "button", class: "btn", html: ICON.back, "aria-label": "previous frame" }, row);
    const bPlay = h("button", { type: "button", class: "btn primary", html: ICON.play, "aria-label": "play / pause" }, row);
    const bFwd = h("button", { type: "button", class: "btn", html: ICON.fwd, "aria-label": "next frame" }, row);
    const scrubBox = h("div", { class: "scrub" }, controls);
    const scrub = h("input", { type: "range", min: 0, max: 1, step: 1, value: 0, "aria-label": "frame" }, scrubBox);
    const readout = h("div", { class: "readout" }, scrubBox);
    const rFrame = h("span", null, readout);
    const rTime = h("span", null, readout);
    const gRate = bi(h("div", { class: "control-group" }, controls), T("速度", "Speed"));
    seg(gRate, RATES.map(([k, label]) => ({ key: k, label })), st.rate, k => { st.rate = +k; player.setRate(st.rate); });
    const gMode = bi(h("div", { class: "control-group" }, controls), T("视图", "View"));
    seg(gMode, [
      { key: "side", label: T(`${sources.length} 路并排`, "Side by side") },
      { key: "wipe", label: T("滑块对比", "Wipe") },
    ], st.mode, k => { if (k !== st.mode) { st.mode = k; mount(); } });
    const loopLab = h("label", { class: "toggle" }, controls);
    const loop = h("input", { type: "checkbox" }, loopLab);
    loop.checked = st.loop;
    bi(loopLab, T("循环", "Loop"));
    const bFull = h("button", { type: "button", class: "btn", html: ICON.full, "aria-label": "fullscreen" }, controls);

    if (spec.hints) {
      const hints = add(h("div", { class: "hints" }));
      h("span", { lang: "zh", html: "<kbd>空格</kbd> 播放 / 暂停 · <kbd>←</kbd> <kbd>→</kbd> 逐帧 · <kbd>Shift</kbd> + <kbd>←</kbd> <kbd>→</kbd> 前后 1 秒 · <kbd>[</kbd> <kbd>]</kbd> 上一段 / 下一段 · <kbd>F</kbd> 全屏 · 点击某一路画面可以放大。快捷键作用于最近点过、或者当前在屏幕上的那组视频。" }, hints);
      h("span", { lang: "en", html: "<kbd>Space</kbd> play / pause · <kbd>←</kbd> <kbd>→</kbd> one frame · <kbd>Shift</kbd> + <kbd>←</kbd> <kbd>→</kbd> one second · <kbd>[</kbd> <kbd>]</kbd> previous / next clip · <kbd>F</kbd> fullscreen · click a video to enlarge it. Shortcuts go to the player you used last, or the one on screen." }, hints);
    }

    let scrubbing = false, resume = false, shownWant = null, token = 0;
    const player = new SyncPlayer(render);

    function render(hint) {
      const c = st.clip;
      const frame = hint ?? player.frame;
      if (!scrubbing) scrub.value = String(frame);
      scrub.style.setProperty("--fill", `${(frame / Math.max(1, c.frames - 1)) * 100}%`);
      rFrame.innerHTML = tr(T(`第 <b>${frame}</b> 帧 / 共 ${c.frames} 帧`, `frame <b>${frame}</b> of ${c.frames}`));
      rTime.textContent = `${(frame / c.fps).toFixed(2)} s / ${(c.frames / c.fps).toFixed(2)} s`;
      if (shownWant !== player.want) {
        shownWant = player.want;
        bPlay.innerHTML = player.want ? ICON.pause : ICON.play;
      }
      buffering.classList.toggle("show", player.want && player.stalled);
    }

    function renderHead() {
      const c = st.clip;
      const i = clips.indexOf(c);
      tag.textContent = c.dataset;
      index.textContent = "";
      bi(index, T(`第 ${i + 1} / ${clips.length} 段`, `clip ${i + 1} of ${clips.length}`));
      title.textContent = c.label;
      facts.textContent = "";
      const fact = (k, v) => {
        const d = h("div", null, facts);
        bi(h("dt", null, d), k);
        bi(h("dd", null, d), v);
      };
      fact(T("帧数", "Frames"), String(c.frames));
      fact(T("时长", "Duration"), `${(c.frames / c.fps).toFixed(2)} s`);
      fact(T("帧率", "Frame rate"), `${c.fps} fps`);
      fact(T("分辨率", "Size"), c.size);
      picks.forEach((b, j) => b.classList.toggle("active", j === i));
      const b = picks[i];
      if (picker.scrollWidth > picker.clientWidth) {
        picker.scrollTo({ left: b.offsetLeft - picker.offsetLeft - (picker.clientWidth - b.offsetWidth) / 2, behavior: "smooth" });
      }
      scrub.max = String(c.frames - 1);
    }

    function makeVideo(src, cls) {
      const v = h("video", { class: cls, src, preload: "auto", playsinline: true, muted: true, disablepictureinpicture: true, poster: st.clip.thumb });
      v.muted = true;
      return v;
    }

    function setFocus(name) {
      st.focus = st.focus === name ? null : name;
      const tiles = stage.querySelector(".tiles");
      if (!tiles) return;
      tiles.classList.toggle("focus", Boolean(st.focus));
      for (const f of tiles.children) f.classList.toggle("is-focus", f.dataset.name === st.focus);
    }

    function buildSide() {
      const tiles = h("div", { class: "tiles" + (st.focus ? " focus" : "") });
      tiles.style.setProperty("--cols", String(Math.min(3, sources.length)));
      tiles.style.setProperty("--rest", String(Math.max(1, sources.length - 1)));
      const videos = sources.map(v => {
        const f = h("figure", { class: "tile" + (st.focus === v ? " is-focus" : ""), "data-name": v }, tiles);
        const video = makeVideo(st.clip.file(v));
        f.appendChild(video);
        video.addEventListener("click", () => setFocus(v));
        const fc = h("figcaption", null, f);
        h("span", { class: `dot ${v}` }, fc);
        bi(h("span", { class: "name" }, fc), NAMES[v]);
        const m = metric(spec, st.clip, v);
        if (m) bi(h("span", { class: "metric" }, fc), m);
        return video;
      });
      return { node: tiles, videos };
    }

    function buildWipe() {
      const L = st.wipeLeft, R = st.wipeRight;
      const bottom = makeVideo(st.clip.file(L));
      const top = makeVideo(st.clip.file(R), "top");
      const wipe = h("div", { class: "wipe" });
      wipe.append(bottom, top);
      h("span", null, h("div", { class: "divider" }, wipe));
      const tagL = h("span", { class: "wipe-tag left" }, wipe);
      h("span", { class: `dot ${L}` }, tagL);
      bi(tagL, NAMES[L]);
      const tagR = h("span", { class: "wipe-tag right" }, wipe);
      h("span", { class: `dot ${R}` }, tagR);
      bi(tagR, NAMES[R]);
      wipe.style.setProperty("--pos", `${st.wipePos * 100}%`);
      const move = e => {
        const r = wipe.getBoundingClientRect();
        st.wipePos = clamp((e.clientX - r.left) / r.width, 0, 1);
        wipe.style.setProperty("--pos", `${st.wipePos * 100}%`);
      };
      wipe.addEventListener("pointerdown", e => { wipe.setPointerCapture(e.pointerId); move(e); });
      wipe.addEventListener("pointermove", e => { if (wipe.hasPointerCapture(e.pointerId)) move(e); });
      selects = [];
      const pick = (value, onPick) => {
        const sel = h("select");
        for (const v of sources) {
          const o = h("option", { value: v }, sel, tr(NAMES[v]));
          o.selected = v === value;
        }
        sel.addEventListener("change", () => onPick(sel.value));
        selects.push(sel);
        return sel;
      };
      const label = h("div", { class: "wipe-label" });
      bi(h("span", null, label), T("左边", "Left"));
      label.append(pick(L, v => { st.wipeLeft = v; mount(); }));
      bi(h("span", null, label), T("右边", "Right"));
      label.append(pick(R, v => { st.wipeRight = v; mount(); }));
      bi(h("span", null, label), T("在画面上左右拖动分割线", "drag across the picture to move the divider"));
      const wrap = h("div", { class: "wipe-wrap" });
      wrap.append(wipe, label);
      return { node: wrap, videos: [bottom, top] };
    }

    function placeholder() {
      const tiles = h("div", { class: "tiles" });
      tiles.style.setProperty("--cols", String(Math.min(3, sources.length)));
      for (const v of sources) {
        const f = h("figure", { class: "tile" }, tiles);
        h("video", { poster: st.clip.thumb, preload: "none", muted: true }, f);
        const fc = h("figcaption", null, f);
        h("span", { class: `dot ${v}` }, fc);
        bi(h("span", { class: "name" }, fc), NAMES[v]);
        const m = metric(spec, st.clip, v);
        if (m) bi(h("span", { class: "metric" }, fc), m);
      }
      stage.replaceChildren(tiles);
    }

    function mount({ frame, play } = {}) {
      const startFrame = frame ?? player.frame;
      const shouldPlay = play ?? player.want;
      player.halt();
      player.detach();
      const built = st.mode === "wipe" ? buildWipe() : buildSide();
      stage.replaceChildren(built.node, buffering);
      player.loop = st.loop;
      player.setRate(st.rate);
      player.attach(built.videos, st.clip.fps, st.clip.frames);
      const my = ++token;
      Promise.all(built.videos.map(v => (v.readyState >= 1 ? 0 : new Promise(r => {
        v.addEventListener("loadedmetadata", r, { once: true });
        v.addEventListener("error", r, { once: true });
      })))).then(() => {
        if (my !== token) return;
        player.seekFrame(startFrame);
        if (shouldPlay) player.play();
      });
      render(startFrame);
    }

    function select(c, play) {
      st.clip = c;
      renderHead();
      if (st.live) mount({ frame: 0, play: play ?? (player.want || st.auto) });
      else placeholder();
    }

    function selectRelative(delta) {
      const i = clips.indexOf(st.clip);
      select(clips[(i + delta + clips.length) % clips.length]);
    }

    function toggle() {
      st.auto = !player.want;
      if (player.want) player.pause();
      else player.play();
    }

    function toggleFullscreen() {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (body.requestFullscreen) body.requestFullscreen();
    }

    const click = (btn, fn) => btn.addEventListener("click", e => { e.currentTarget.blur(); fn(); });
    click(bPlay, toggle);
    click(bBack, () => player.step(-1));
    click(bFwd, () => player.step(1));
    click(bFull, toggleFullscreen);
    loop.addEventListener("change", () => { st.loop = loop.checked; player.loop = st.loop; });
    scrub.addEventListener("input", () => {
      if (!scrubbing) {
        scrubbing = true;
        resume = player.want;
        if (player.want) player.halt();
      }
      player.seekFrame(+scrub.value);
    });
    scrub.addEventListener("change", () => {
      scrubbing = false;
      if (resume) player.play();
      resume = false;
    });

    const self = { fig, player, toggle, toggleFullscreen, selectRelative, fps: () => st.clip.fps };
    players.push(self);
    fig.addEventListener("pointerdown", () => { active = self; });
    fig.addEventListener("mouseenter", () => { active = self; });

    onLang(() => {
      render();
      for (const sel of selects) for (const o of sel.options) o.textContent = tr(NAMES[o.value]);
    });

    fig.classList.add("vp");
    renderHead();
    placeholder();
    render(0);
    whenVisible(fig, () => {
      if (st.live) return;
      st.live = true;
      mount({ frame: 0, play: st.auto && st.visible });
    }, null, { threshold: 0, margin: "500px 0px" });
    whenVisible(fig, () => {
      st.visible = true;
      if (st.live && st.auto && !player.want) player.play();
    }, () => {
      st.visible = false;
      if (player.want) player.halt();
    }, { threshold: 0.4 });
  }

  function onScreen(fig) {
    const r = fig.getBoundingClientRect();
    return Math.max(0, Math.min(innerHeight, r.bottom) - Math.max(0, r.top));
  }

  function target() {
    if (active && onScreen(active.fig) > 160) return active;
    let best = null, area = 160;
    for (const p of players) {
      const a = onScreen(p.fig);
      if (a > area) { best = p; area = a; }
    }
    return best;
  }

  document.addEventListener("keydown", e => {
    const t = e.target;
    if (t instanceof HTMLInputElement && t.type !== "checkbox") return;
    if (t instanceof HTMLButtonElement && (e.key === " " || e.key === "Enter")) return;
    if (t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement || e.metaKey || e.ctrlKey || e.altKey) return;
    const p = target();
    if (!p) return;
    const second = Math.round(p.fps());
    switch (e.key) {
      case " ": e.preventDefault(); p.toggle(); break;
      case "ArrowLeft": e.preventDefault(); p.player.step(e.shiftKey ? -second : -1); break;
      case "ArrowRight": e.preventDefault(); p.player.step(e.shiftKey ? second : 1); break;
      case "[": p.selectRelative(-1); break;
      case "]": p.selectRelative(1); break;
      case "f": case "F": p.toggleFullscreen(); break;
      default:
    }
  });

  window.__players = players;
  register("figure[data-player]", mountPlayer, false);
})();
