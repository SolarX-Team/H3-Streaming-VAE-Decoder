/* Explainer animations: the official windowed decode, the causal stream, and V3's first three steps.
   Each animation is a pure draw(t) over pre-built SVG nodes, driven by a small timeline. */
(() => {
  "use strict";
  const { T, tr, h, s, bi, seg, onLang, whenVisible, register, clamp } = H3;

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zm8 0h4v16h-4z"/></svg>';

  const ease = x => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
  const p01 = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
  const ep = (t, a, b) => ease(p01(t, a, b));
  const lerp = (a, b, k) => a + (b - a) * k;

  const C = {
    accent: "#6a9bcc", off: "#d97757", ours: "#788c5d",
    text: "#141413", muted: "#5e5d59", faint: "#87867f", line2: "#c2c0b6", dash: "#b0aea5",
    cell: "#ffffff", cellLine: "#d1cfc5", cellText: "#3d3d3a", latText: "#28466a", offText: "#94452a",
  };
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  };
  const set = (el, attrs) => { for (const k in attrs) el.setAttribute(k, typeof attrs[k] === "number" ? attrs[k].toFixed(2) : attrs[k]); };
  const show = (el, o) => {
    const vis = o > 0.002;
    el.setAttribute("opacity", vis ? Math.min(1, o).toFixed(3) : "0");
    el.style.display = vis ? "" : "none";
  };
  const shift = (el, dx, dy) => el.setAttribute("transform", `translate(${dx.toFixed(2)} ${dy.toFixed(2)})`);
  const scaleAbout = (el, x, y, k) => el.setAttribute("transform", `translate(${x} ${y}) scale(${k.toFixed(3)}) translate(${-x} ${-y})`);

  /* SVG text in both languages; CSS hides the inactive one. */
  function label(parent, x, y, text, attrs = {}) {
    const g = s("g", null, parent);
    if (text && typeof text === "object") {
      s("text", { x, y, lang: "zh", ...attrs }, g, text.zh);
      s("text", { x, y, lang: "en", ...attrs }, g, text.en);
    } else s("text", { x, y, ...attrs }, g, text);
    return g;
  }

  function bracket(parent, x1, x2, y, text, anchor = "middle") {
    s("path", { d: `M${x1},${y - 6}V${y}H${x2}V${y - 6}`, fill: "none", stroke: C.muted, "stroke-width": 1 }, parent);
    const x = anchor === "end" ? x2 : anchor === "start" ? x1 : (x1 + x2) / 2;
    label(parent, x, y + 17, text, { fill: C.muted, "font-size": 12, "text-anchor": anchor });
  }

  /* A slot of the decoder window: cached / computing look, latent name, a small status line. */
  function tile(parent, x, y, w, hh, name, head) {
    const g = s("g", null, parent);
    s("rect", { x, y, width: w, height: hh, rx: 8, fill: rgba(C.accent, head ? 0.24 : 0.12), stroke: rgba(C.accent, 0.6) }, g);
    const hot = s("rect", { x, y, width: w, height: hh, rx: 8, fill: rgba(C.accent, 0.34), stroke: C.accent, "stroke-width": 1.5, opacity: 0 }, g);
    s("text", { x: x + w / 2, y: y + hh / 2 + 1, "text-anchor": "middle", "font-size": 14, fill: C.latText, class: "mono" }, g, name);
    const bar = s("rect", { x: x + 8, y: y + hh - 7, width: 0, height: 2, rx: 1, fill: C.accent, opacity: 0 }, g);
    const st = {
      kv: label(g, x + w / 2, y + hh - 9, T("K/V 已缓存", "cached K/V"), { "text-anchor": "middle", "font-size": 10.5, fill: C.faint }),
      run: label(g, x + w / 2, y + hh - 11, T("计算中", "computing"), { "text-anchor": "middle", "font-size": 10.5, fill: C.latText }),
      pre: label(g, x + w / 2, y + hh - 11, "prefill", { "text-anchor": "middle", "font-size": 10.5, fill: C.latText }),
      re: label(g, x + w / 2, y + hh - 11, T("重算", "recompute"), { "text-anchor": "middle", "font-size": 10.5, fill: C.latText }),
    };
    return { g, hot, bar, st, x, w };
  }
  function tileState(tl, hot, prog, kind, idle = "kv") {
    tl.hot.setAttribute("opacity", hot.toFixed(3));
    set(tl.bar, { width: (tl.w - 16) * prog });
    tl.bar.setAttribute("opacity", hot > 0.05 && prog > 0 && prog < 1 ? "1" : "0");
    const busy = hot > 0.5;
    for (const k in tl.st) show(tl.st[k], k === (busy ? kind : idle) ? 1 : 0);
  }

  function arc(parent, x1, x2, y) {
    const d = Math.abs(x2 - x1);
    return s("path", {
      d: `M${x1},${y}Q${(x1 + x2) / 2},${y - 14 - d * 0.16} ${x2},${y}`,
      fill: "none", stroke: rgba(C.accent, 0.75), "stroke-width": 1.2,
    }, parent);
  }

  const LEGEND = {
    hot: [`background:${rgba(C.accent, 0.3)};box-shadow:inset 0 0 0 1.5px ${C.accent}`, T("正在计算的格", "slot being computed")],
    kv: [`background:${rgba(C.accent, 0.12)};box-shadow:inset 0 0 0 1px ${rgba(C.accent, 0.45)}`, T("已缓存 K/V 的格", "slot with cached K/V")],
    empty: [`background:none;border:1px dashed ${C.dash}`, T("空格（latent 还没生成）", "empty slot (latent not generated yet)")],
    ours: [`background:${C.ours}`, T("流式输出的帧", "frames from the stream")],
    off: [`background:${C.off}`, T("官方输出的帧", "frames from official decode")],
  };
  const legend = (...keys) => el => {
    for (const k of keys) {
      const sp = h("span", null, el);
      h("i", { class: "box", style: LEGEND[k][0] }, sp);
      bi(sp, LEGEND[k][1]);
    }
  };

  /* Timeline: play / pause, a scrubber, chapter buttons and one caption line per chapter. */
  function timeline(fig, o) {
    const box = fig.querySelector(".anim");
    const stage = h("div", { class: "stage" }, box);
    const svg = s("svg", { viewBox: `0 0 ${o.W} ${o.H}`, role: "img" }, stage);
    const draw = o.build(svg);
    const bar = h("div", { class: "bar" }, box);
    const btn = h("button", { type: "button", class: "btn primary", html: ICON_PLAY, "aria-label": "play / pause" }, bar);
    const sc = h("div", { class: "scrub" }, bar);
    const range = h("input", { type: "range", min: 0, max: 1000, step: 1, value: 0, "aria-label": "time" }, sc);
    const chapters = seg(bar, o.chapters.map((c, i) => ({ key: i, label: c.label })), 0, i => seek(o.chapters[+i].t));
    const say = h("p", { class: "say" }, box);
    if (o.legend) o.legend(h("div", { class: "legend" }, box));
    let t = 0, playing = false, last = 0, raf = 0, auto = !reduced, shown = "";

    function paint() {
      draw(t);
      let k = 0;
      o.chapters.forEach((c, i) => { if (t >= c.t - 1e-6) k = i; });
      chapters.sync(k);
      const key = k + H3.lang();
      if (key !== shown) { shown = key; say.innerHTML = tr(o.chapters[k].say); }
      range.value = String(Math.round((1000 * t) / o.duration));
      range.style.setProperty("--fill", `${(100 * t) / o.duration}%`);
    }
    function frame(now) {
      raf = 0;
      if (!playing) return;
      t += Math.min(0.05, (now - last) / 1000);
      last = now;
      if (t >= o.duration) t = 0;
      paint();
      raf = requestAnimationFrame(frame);
    }
    function play() {
      if (playing) return;
      playing = true;
      btn.innerHTML = ICON_PAUSE;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    }
    function pause() {
      playing = false;
      btn.innerHTML = ICON_PLAY;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    }
    function seek(x) {
      t = clamp(x, 0, o.duration - 1e-3);
      paint();
    }
    btn.addEventListener("click", () => {
      btn.blur();
      if (playing) { auto = false; pause(); }
      else { auto = true; play(); }
    });
    range.addEventListener("input", () => { auto = false; pause(); seek((+range.value / 1000) * o.duration); });
    onLang(paint);
    whenVisible(box, () => { if (auto) play(); }, pause, { threshold: 0.35 });
    seek(reduced ? o.duration - 0.01 : 0);
  }

  /* ---------------------------------------------------------------------------------------- */
  /* Official decoder: 17-frame chunks padded to 20 → 5 latents; 7 latents → 28 frames → slice. */
  function officialScene(svg) {
    const X0 = 132, CW = 24, CG = 4, GW = 4 * CW + 3 * CG, GG = 14;
    const gx = g => X0 + g * (GW + GG);
    const cx = (g, j) => gx(g) + j * (CW + CG);
    const Y_IN = 58, Y_LAT = 154, Y_OUT = 252, CH = 30, LW = 64, LH = 34;
    const lx = g => gx(g) + (GW - LW) / 2;
    const RIGHT = gx(6) + GW;
    const frameNo = (g, j) => (g === 0 ? (j < 3 ? null : 0) : g < 5 ? 1 + (g - 1) * 4 + j : g === 5 ? (j < 3 ? null : 17) : 18 + j);
    const BLEND = { "5,3": 1, "6,0": 0.8, "6,1": 0.6, "6,2": 0.4, "6,3": 0.2 };

    const rowLabel = (y, text) => label(svg, 0, y, text, { fill: C.muted, "font-size": 13 });
    rowLabel(Y_IN + 20, T("视频帧", "Frames"));
    rowLabel(Y_LAT + 22, "latent");
    rowLabel(Y_OUT + 20, T("decoder 输出", "Decoder output"));
    const divX = gx(5) - GG / 2;
    s("line", { x1: divX, x2: divX, y1: 22, y2: 346, stroke: C.line2, "stroke-dasharray": "3 4" }, svg);
    label(svg, (gx(0) + gx(4) + GW) / 2, 36, T("第 1 段：17 帧", "chunk 1: 17 frames"), { fill: C.faint, "font-size": 12, "text-anchor": "middle" });
    label(svg, (gx(5) + RIGHT) / 2, 36, T("第 2 段的开头", "start of chunk 2"), { fill: C.faint, "font-size": 12, "text-anchor": "middle" });

    const cells = [];
    for (let g = 0; g < 7; g++) for (let j = 0; j < 4; j++) {
      const n = frameNo(g, j), pad = n == null;
      const inG = s("g", null, svg);
      s("rect", { x: cx(g, j), y: Y_IN, width: CW, height: CH, rx: 4, fill: pad ? "none" : C.cell, stroke: pad ? C.dash : C.cellLine, "stroke-dasharray": pad ? "3 3" : null }, inG);
      s("text", { x: cx(g, j) + CW / 2, y: Y_IN + 19.5, "text-anchor": "middle", "font-size": 11, fill: pad ? C.faint : C.cellText, class: "mono" }, inG, pad ? "P" : String(n));
      const enc = s("rect", { rx: 3, fill: rgba(C.accent, 0.38) }, svg);
      const dec = s("rect", { rx: 3, fill: rgba(C.off, 0.4) }, svg);
      const outG = s("g", null, svg);
      s("rect", { x: cx(g, j), y: Y_OUT, width: CW, height: CH, rx: 4, fill: rgba(C.off, 0.15), stroke: rgba(C.off, pad ? 0.4 : 0.62), "stroke-dasharray": pad ? "3 3" : null }, outG);
      s("text", { x: cx(g, j) + CW / 2, y: Y_OUT + 19.5, "text-anchor": "middle", "font-size": 11, fill: pad ? C.faint : C.offText, class: "mono" }, outG, pad ? "p" : String(n));
      cells.push({ g, j, i: g * 4 + j, pad, inG, enc, dec, outG, blend: BLEND[`${g},${j}`] });
    }

    const lats = [];
    for (let g = 0; g < 7; g++) {
      const lg = s("g", null, svg);
      const head = g === 0 || g === 5;
      s("rect", { x: lx(g), y: Y_LAT, width: LW, height: LH, rx: 6, fill: rgba(C.accent, head ? 0.34 : 0.14), stroke: rgba(C.accent, 0.7) }, lg);
      s("text", { x: lx(g) + LW / 2, y: Y_LAT + 22, "text-anchor": "middle", "font-size": 13, fill: C.latText, class: "mono" }, lg, "z" + g);
      lats.push(lg);
    }
    const win = s("g", null, svg);
    s("rect", { x: X0 - 10, y: Y_LAT - 12, width: RIGHT - X0 + 20, height: LH + 24, rx: 10, fill: "none", stroke: C.line2, "stroke-dasharray": "5 4" }, win);
    label(win, X0 - 4, Y_LAT - 20, T("decoder 窗口：7 个 latent，36 层双向 attention", "decoder window: 7 latents, 36 layers of bidirectional attention"), { fill: C.muted, "font-size": 12 });

    const sliceG = s("g", null, svg);
    bracket(sliceG, cx(0, 0), cx(4, 3) + CW, 300, T("[0:20]，丢掉前 3 帧", "[0:20], first 3 dropped"));
    bracket(sliceG, cx(5, 0), cx(6, 3) + CW, 300, T("[20:28]，丢掉前 3 帧", "[20:28], first 3 dropped"), "end");

    const counts = [[cx(0, 3) + CW / 2, "1"], ...[1, 2, 3, 4].map(g => [gx(g) + GW / 2, "4"]), [cx(5, 3) + CW / 2, "1"], [gx(6) + GW / 2, "4"]]
      .map(([x, n]) => s("text", { x, y: 306, "text-anchor": "middle", "font-size": 16, "font-weight": 650, fill: C.text }, svg, n));
    const resG = s("g", null, svg);
    bracket(resG, cx(0, 3), cx(4, 3) + CW, 326, T("直接输出 17 帧", "17 frames, output directly"));
    bracket(resG, cx(5, 3), cx(6, 3) + CW, 326, T("5 帧，与下一窗口线性混合", "5 frames, blended with the next window"), "end");
    const wText = label(svg, RIGHT, 362, T("本窗口权重 1 → 0.2，下一窗口 0 → 0.8", "weights 1 → 0.2 here, 0 → 0.8 in the next window"), { fill: C.faint, "font-size": 11.5, "text-anchor": "end" });

    const ENC = g => 1.2 + 0.12 * g, DEC = g => 3.4 + 0.11 * g;
    return t => {
      for (const c of cells) {
        const a = 0.05 + 0.025 * c.i, k = ep(t, a, a + 0.3);
        show(c.inG, k);
        shift(c.inG, 0, (1 - k) * -6);

        const e0 = ENC(c.g), ke = ep(t, e0, e0 + 0.7);
        const sub = LW / 4;
        set(c.enc, { x: lerp(cx(c.g, c.j), lx(c.g) + c.j * sub + 1, ke), y: lerp(Y_IN, Y_LAT, ke), width: lerp(CW, sub - 2, ke), height: lerp(CH, LH, ke) });
        show(c.enc, 0.9 * Math.min(p01(t, e0, e0 + 0.08), 1 - p01(t, e0 + 0.5, e0 + 0.7)));

        const d0 = DEC(c.g), kd = ep(t, d0, d0 + 0.7);
        set(c.dec, { x: lerp(lx(c.g) + c.j * sub + 1, cx(c.g, c.j), kd), y: lerp(Y_LAT, Y_OUT, kd), width: lerp(sub - 2, CW, kd), height: lerp(LH, CH, kd) });
        show(c.dec, 0.9 * Math.min(p01(t, d0, d0 + 0.08), 1 - p01(t, d0 + 0.5, d0 + 0.7)));

        let o = ep(t, d0 + 0.5, d0 + 0.75), dy = 0;
        if (c.pad) { const k2 = ep(t, 5.9, 6.4); o *= 1 - 0.75 * k2; dy = 6 * k2; }
        if (c.blend != null) o *= 1 - (1 - c.blend) * ep(t, 8.0, 8.5);
        show(c.outG, o);
        shift(c.outG, 0, dy);
      }
      lats.forEach((lg, g) => {
        const a = ENC(g) + 0.5, k = ep(t, a, a + 0.3);
        show(lg, k);
        scaleAbout(lg, lx(g) + LW / 2, Y_LAT + LH / 2, 0.86 + 0.14 * k);
      });
      show(win, ep(t, 2.8, 3.2));
      show(sliceG, Math.min(ep(t, 5.6, 5.9), 1 - ep(t, 6.9, 7.2)));
      counts.forEach((c, i) => show(c, ep(t, 7.2 + 0.08 * i, 7.5 + 0.08 * i)));
      show(resG, ep(t, 7.8, 8.1));
      show(wText, ep(t, 8.1, 8.4));
    };
  }

  register('[data-anim="official"]', fig => timeline(fig, {
    W: 1000, H: 372, duration: 11.5, build: officialScene,
    chapters: [
      { t: 0, label: T("编码", "Encode"), say: T(
        "<b>编码</b>：视频按 17 帧一段，每段左边补 3 帧（P）凑成 20 帧，因果 3D 卷积在时间上压缩 4 倍，得到 5 个 latent。所以每段第一个 latent（z0、z5，颜色更深）只含 1 帧真实画面。",
        "<b>Encode</b>: the video is cut into 17-frame chunks, each left-padded with 3 frames (P) to 20 frames, and the causal 3D CNN compresses time by 4×, giving 5 latents. So the first latent of each chunk (z0 and z5, darker) holds a single real frame.") },
      { t: 2.8, label: T("解码", "Decode"), say: T(
        "<b>解码</b>：decoder 一次读 7 个 latent，本段 5 个加下一段 2 个，36 层都是双向 attention。<code>proj_out</code> 给每个 latent 均匀地输出 4 帧，一共 28 帧。",
        "<b>Decode</b>: the decoder reads 7 latents at a time, 5 from this chunk and 2 from the next, with bidirectional attention in all 36 layers. <code>proj_out</code> emits 4 frames for every latent, 28 frames in total.") },
      { t: 5.5, label: T("切片", "Slice"), say: T(
        "<b>切片</b>：<code>decode_temporal</code> 取 [0:20] 和 [20:28]，各丢掉开头 3 帧，丢掉的正是两个段首 latent 解出来的补帧（p）。",
        "<b>Slice</b>: <code>decode_temporal</code> takes [0:20] and [20:28] and drops the first 3 frames of each, which are exactly the padding frames (p) decoded from the two chunk-start latents.") },
      { t: 6.9, label: T("结果", "Result"), say: T(
        "<b>结果</b>：前 17 帧（1 4 4 4 4）直接输出；后 5 帧（1 4）与下一个窗口的前 5 帧线性混合。下一个窗口从 z5 开始，所以段首 latent 总在 slot 0 和 slot 5。",
        "<b>Result</b>: the first 17 frames (1 4 4 4 4) are output directly, and the last 5 (1 4) are blended linearly with the first 5 frames of the next window. The next window starts at z5, so chunk-start latents always sit at slots 0 and 5.") },
    ],
  }), false);

  /* ---------------------------------------------------------------------------------------- */
  /* Causal stream: latents arrive one by one; each computes its own slot and emits at once. */
  function streamScene(svg) {
    const X0 = 150;
    const GY = 30, GWd = 50, GH = 28, GGap = 8;
    const gxk = k => X0 + k * (GWd + GGap);
    const WY = 120, SW = 88, SH = 54, SG = 12;
    const sx = sl => X0 + sl * (SW + SG);
    const OY1 = 254, OY2 = 300, FW = 14, FG = 3, FH = 26;
    const fx = f => X0 + f * (FW + FG);
    const NF = 39, NK = 12;
    const A = k => (k <= 7 ? 0.6 + k : 0.6 + 7 + 1.8 + (k - 8));
    const A7 = A(7);
    const tok = f => { const c = Math.floor(f / 17), r = f % 17; return r === 0 ? 5 * c : 5 * c + 1 + Math.floor((r - 1) / 4); };
    const first = k => 17 * Math.floor(k / 5) + (k % 5 === 0 ? 0 : 1 + 4 * ((k % 5) - 1));
    const emit = k => (k === 7 ? A7 + 1.6 : A(k) + 0.95);
    const OFF = [A(6) + 0.85, A(11) + 0.85];
    const offAt = f => (f <= 16 ? OFF[0] + 0.012 * f : OFF[1] + 0.012 * (f - 17));
    const oursAt = f => emit(tok(f)) + 0.04 * (f - first(tok(f)));
    const compute = k => (k === 7 ? [A7 + 1.15, A7 + 1.6] : [A(k) + 0.6, A(k) + 0.95]);

    label(svg, 0, GY + 19, T("生成的 latent", "Generated latents"), { fill: C.muted, "font-size": 13 });
    label(svg, 0, WY + 24, T("decoder 窗口", "Decoder window"), { fill: C.muted, "font-size": 13 });
    const range0 = s("text", { x: 0, y: WY + 44, fill: C.faint, "font-size": 12, class: "mono" }, svg, "z0 – z6");
    const range1 = s("text", { x: 0, y: WY + 44, fill: C.faint, "font-size": 12, class: "mono" }, svg, "z5 – z11");
    s("circle", { cx: 4, cy: OY1 + 13.5, r: 4, fill: C.ours }, svg);
    label(svg, 14, OY1 + 18, T("流式输出", "Stream output"), { fill: C.muted, "font-size": 13 });
    s("circle", { cx: 4, cy: OY2 + 13.5, r: 4, fill: C.off }, svg);
    label(svg, 14, OY2 + 18, T("官方 decode", "Official decode"), { fill: C.muted, "font-size": 13 });

    for (let i = 0; i < 7; i++) {
      s("rect", { x: sx(i), y: WY, width: SW, height: SH, rx: 8, fill: "none", stroke: C.dash, "stroke-dasharray": "4 4" }, svg);
      s("text", { x: sx(i) + SW / 2, y: WY + SH + 17, "text-anchor": "middle", "font-size": 11, fill: C.faint }, svg, `slot ${i}`);
    }
    for (const f of [17, 34]) {
      const x = fx(f) - FG / 2;
      s("line", { x1: x, x2: x, y1: OY1 - 6, y2: OY2 + FH + 6, stroke: C.line2, "stroke-dasharray": "3 4" }, svg);
    }
    label(svg, (fx(0) + fx(16) + FW) / 2, OY2 + FH + 24, T("第 1 段", "chunk 1"), { fill: C.faint, "font-size": 11.5, "text-anchor": "middle" });
    label(svg, (fx(17) + fx(33) + FW) / 2, OY2 + FH + 24, T("第 2 段", "chunk 2"), { fill: C.faint, "font-size": 11.5, "text-anchor": "middle" });

    const arcs = [];
    for (let k = 0; k < NK; k++) {
      const g = s("g", null, svg);
      const pairs = k === 7 ? [[1, 0], [2, 0], [2, 1]] : Array.from({ length: k < 7 ? k : k - 5 }, (_, j) => [k < 7 ? k : k - 5, j]);
      for (const [a, b] of pairs) arc(g, sx(a) + SW / 2, sx(b) + SW / 2, WY - 3);
      arcs.push({ k, g });
    }

    const gen = [], travel = [];
    for (let k = 0; k < NK; k++) {
      const head = k % 5 === 0;
      const mk = () => {
        const g = s("g", null, svg);
        s("rect", { x: 0, y: 0, width: GWd, height: GH, rx: 6, fill: rgba(C.accent, head ? 0.32 : 0.14), stroke: rgba(C.accent, 0.62) }, g);
        s("text", { x: GWd / 2, y: GH / 2 + 4.5, "text-anchor": "middle", "font-size": 12, fill: C.latText, class: "mono" }, g, "z" + k);
        return g;
      };
      gen.push(mk());
      travel.push(mk());
    }

    const tiles = [];
    const addTile = (k, w) => {
      const slot = k - 5 * w;
      const tl = tile(svg, 0, WY, SW, SH, "z" + k, k % 5 === 0);
      tiles.push({ k, w, slot, tl });
    };
    for (let k = 0; k <= 6; k++) addTile(k, 0);
    for (let k = 5; k <= 11; k++) addTile(k, 1);

    const ours = [], offs = [];
    for (let f = 0; f < NF; f++) {
      ours.push(s("rect", { x: fx(f), y: OY1, width: FW, height: FH, rx: 3, fill: C.ours, opacity: 0 }, svg));
      if (f < 34) offs.push(s("rect", { x: fx(f), y: OY2, width: FW, height: FH, rx: 3, fill: C.off, opacity: 0 }, svg));
    }
    const wait = [
      label(svg, fx(0) + 2, OY2 + 17, T("等待 z6 生成…", "waiting for z6…"), { fill: C.faint, "font-size": 12 }),
      label(svg, fx(0) + 2, OY2 + 17, T("解码 z0–z6…", "decoding z0–z6…"), { fill: C.faint, "font-size": 12 }),
      label(svg, fx(17) + 2, OY2 + 17, T("等待 z11 生成…", "waiting for z11…"), { fill: C.faint, "font-size": 12 }),
      label(svg, fx(17) + 2, OY2 + 17, T("解码 z5–z11…", "decoding z5–z11…"), { fill: C.faint, "font-size": 12 }),
    ];
    const prog = [0, 17].map(f0 => s("rect", { x: fx(f0), y: OY2 + FH + 3, width: 0, height: 2, rx: 1, fill: C.off }, svg));
    const cnt = [OY1, OY2].map(y => ({
      zh: s("text", { x: fx(NF) + 12, y: y + 18, lang: "zh", fill: C.muted, "font-size": 12.5 }, svg),
      en: s("text", { x: fx(NF) + 12, y: y + 18, lang: "en", fill: C.muted, "font-size": 12.5 }, svg),
      n: -1,
    }));
    const count = (c, n) => {
      if (c.n === n) return;
      c.n = n;
      c.zh.textContent = `${n} 帧`;
      c.en.textContent = n === 1 ? "1 frame" : `${n} frames`;
    };

    return t => {
      for (let k = 0; k < NK; k++) {
        const a = A(k);
        const [t0, t1] = k === 7 ? [A7 + 0.8, A7 + 1.15] : [a + 0.2, a + 0.6];
        const kg = ep(t, a, a + 0.3);
        show(gen[k], kg * (1 - 0.5 * ep(t, t1 - 0.1, t1 + 0.15)));
        shift(gen[k], gxk(k), GY - 8 * (1 - kg));
        const kt = ep(t, t0, t1);
        const slot = k < 7 ? k : k - 5;
        shift(travel[k], lerp(gxk(k), sx(slot) + (SW - GWd) / 2, kt), lerp(GY, WY + (SH - GH) / 2, kt));
        show(travel[k], Math.min(p01(t, t0, t0 + 0.06), 1 - p01(t, t1 - 0.04, t1 + 0.06)));
        const [c0, c1] = compute(k);
        show(arcs[k].g, 0.9 * Math.min(ep(t, c0, c0 + 0.1), 1 - ep(t, c1, c1 + 0.15)));
      }
      for (const { k, w, slot, tl } of tiles) {
        let o, x = sx(slot), hot, pr, kind = "run";
        if (w === 0) {
          const arrive = A(k) + 0.55;
          o = p01(t, arrive, arrive + 0.08);
          if (k <= 4) o *= 1 - ep(t, A7 + 0.1, A7 + 0.45);
          else if (t >= A7 + 0.45) o = 0;
          const [c0, c1] = compute(k);
          hot = Math.min(ep(t, c0 - 0.05, c0 + 0.05), 1 - ep(t, c1, c1 + 0.1));
          pr = p01(t, c0, c1);
        } else if (k <= 6) {
          o = t >= A7 + 0.45 ? 1 : 0;
          x = lerp(sx(k), sx(slot), ep(t, A7 + 0.45, A7 + 1.05));
          const [c0, c1] = compute(7);
          hot = Math.min(ep(t, c0 - 0.05, c0 + 0.05), 1 - ep(t, c1, c1 + 0.1));
          pr = p01(t, c0, c1);
          kind = "pre";
        } else {
          const arrive = k === 7 ? A7 + 1.12 : A(k) + 0.55;
          o = p01(t, arrive, arrive + 0.08);
          const [c0, c1] = compute(k);
          hot = Math.min(ep(t, c0 - 0.05, c0 + 0.05), 1 - ep(t, c1, c1 + 0.1));
          pr = p01(t, c0, c1);
        }
        show(tl.g, o);
        shift(tl.g, x, 0);
        tileState(tl, hot, pr, kind);
      }
      const r1 = ep(t, A7 + 0.45, A7 + 0.75);
      show(range0, 1 - r1);
      show(range1, r1);

      let nOurs = 0, nOff = 0;
      for (let f = 0; f < NF; f++) {
        const ao = oursAt(f), ko = ep(t, ao, ao + 0.2);
        show(ours[f], 0.9 * ko);
        shift(ours[f], 0, -6 * (1 - ko));
        if (t >= ao) nOurs++;
        if (!offs[f]) continue;
        const af = offAt(f), kf = ep(t, af, af + 0.2);
        show(offs[f], 0.9 * kf);
        shift(offs[f], 0, -6 * (1 - kf));
        if (t >= af) nOff++;
      }
      count(cnt[0], nOurs);
      count(cnt[1], nOff);
      const d0 = [A(6) + 0.15, A(11) + 0.15];
      show(wait[0], 1 - ep(t, d0[0] - 0.1, d0[0]));
      show(wait[1], Math.min(ep(t, d0[0], d0[0] + 0.1), 1 - ep(t, OFF[0] - 0.05, OFF[0] + 0.05)));
      show(wait[2], Math.min(ep(t, OFF[0] + 0.4, OFF[0] + 0.6), 1 - ep(t, d0[1] - 0.1, d0[1])));
      show(wait[3], Math.min(ep(t, d0[1], d0[1] + 0.1), 1 - ep(t, OFF[1] - 0.05, OFF[1] + 0.05)));
      prog.forEach((p, i) => {
        set(p, { width: (fx(16) + FW - fx(0)) * p01(t, d0[i], OFF[i]) });
        show(p, Math.min(p01(t, d0[i], d0[i] + 0.05), 1 - ep(t, OFF[i], OFF[i] + 0.3)));
      });
    };
  }

  register('[data-anim="stream"]', fig => timeline(fig, {
    W: 1000, H: 362, duration: 16.4, build: streamScene, legend: legend("hot", "kv", "empty", "ours", "off"),
    chapters: [
      { t: 0, label: "z0", say: T(
        "<b>z0 到达</b>：只算 slot 0，马上输出第 0 帧。后面 6 格对应的 latent 还没生成，就空着，不放任何东西。",
        "<b>z0 arrives</b>: only slot 0 is computed, and frame 0 is out right away. The latents for the other 6 slots do not exist yet, so those slots stay empty.") },
      { t: 1.6, label: "z1–z4", say: T(
        "<b>z1–z4</b>：每来一个 latent 只算它自己那一格，attention 读前面各格缓存的 K/V（上方的弧线），算完马上输出 4 帧，之后不再修改。",
        "<b>z1–z4</b>: each new latent computes only its own slot, attention reads the cached K/V of the earlier slots (the arcs above), and its 4 frames are out as soon as it is done, never to be revised.") },
      { t: 5.6, label: "z5–z6", say: T(
        "<b>z5、z6</b> 落在 slot 5、6，一样是一到就出帧。官方 decode 要等 z6 生成之后才能解第一个窗口，解完一次给出 17 帧。",
        "<b>z5 and z6</b> land in slots 5 and 6 and are decoded on arrival as well. The official decode can only start the first window once z6 exists, and then delivers 17 frames at once.") },
      { t: 7.6, label: T("换窗", "Shift"), say: T(
        "<b>换窗</b>：z7 到达时窗口右移 5 格。z5、z6 搬到 slot 0、1，在新位置重算一次 K/V（prefill，不出帧），z7 放进 slot 2。",
        "<b>Window shift</b>: when z7 arrives the window moves right by 5. z5 and z6 move to slots 0 and 1 and their K/V is recomputed at the new positions (prefill, no frames), and z7 goes into slot 2.") },
      { t: 9.4, label: "z8–z11", say: T(
        "<b>z8–z11</b>：之后都一样，每个 latent 一到就出帧。官方的第二个窗口要等 z11 生成以后才能解。",
        "<b>z8–z11</b>: from here on it is the same, every latent is decoded on arrival. The official decoder's second window has to wait until z11 exists.") },
    ],
  }), false);

  /* ---------------------------------------------------------------------------------------- */
  /* V3 start-up: z0 computed alone, slot 0 recomputed together with z1, then z2 on its own. */
  function v3Scene(svg) {
    const X0 = 176, SW = 78, SH = 50, SG = 10;
    const sx = i => X0 + i * (SW + SG);
    const NX = sx(6) + SW + 20;
    const rows = [
      { title: T("z0 到达", "z0 arrives"), sub: T("只算它自己这一格", "only its own slot"),
        notes: [T("输出第 0 帧", "frame 0 is out"), T("K/V 是独自算出来的", "its K/V was computed alone")] },
      { title: T("z1 到达", "z1 arrives"), sub: T("slot 0 和 z1 一起算", "slot 0 recomputed with z1"),
        notes: [T("slot 0 换成新的 K/V", "slot 0 gets new K/V"), T("输出 z1 的 4 帧", "z1's 4 frames are out")] },
      { title: T("z2 到达", "z2 arrives"), sub: T("只算 z2 这一格", "only z2's slot"),
        notes: [T("z0、z1 不再重算", "z0 and z1 are not recomputed"), T("输出 z2 的 4 帧", "z2's 4 frames are out")] },
    ];
    rows.forEach((r, k) => {
      const y = 30 + 102 * k;
      r.at = 0.2 + 3 * k;
      r.head = s("g", null, svg);
      label(r.head, 0, y + 22, r.title, { fill: C.text, "font-size": 13.5, "font-weight": 600 });
      label(r.head, 0, y + 41, r.sub, { fill: C.faint, "font-size": 12 });
      for (let i = 0; i < 7; i++) s("rect", { x: sx(i), y, width: SW, height: SH, rx: 8, fill: "none", stroke: C.dash, "stroke-dasharray": "4 4" }, svg);
      r.arcs = s("g", null, svg);
      for (let i = 0; i < k; i++) arc(r.arcs, sx(k) + SW / 2, sx(i) + SW / 2, y - 3);
      r.tiles = Array.from({ length: k + 1 }, (_, i) => tile(svg, 0, y, SW, SH, "z" + i, i === 0));
      r.notes = r.notes.map((n, i) => label(svg, NX, y + 20 + 19 * i, n, { fill: C.muted, "font-size": 12.5 }));
      if (k === 2) for (let i = 0; i < 7; i++) s("text", { x: sx(i) + SW / 2, y: y + SH + 18, "text-anchor": "middle", "font-size": 11, fill: C.faint }, svg, `slot ${i}`);
    });

    return t => {
      rows.forEach((r, k) => {
        const a = r.at, c0 = a + 0.75, c1 = a + 1.75;
        show(r.head, 0.35 + 0.65 * ep(t, a, a + 0.3));
        const hot = Math.min(ep(t, c0 - 0.1, c0), 1 - ep(t, c1, c1 + 0.1));
        r.tiles.forEach((tl, i) => {
          const isNew = i === k;
          const o = isNew ? ep(t, a + 0.35, a + 0.65) : ep(t, a, a + 0.3);
          show(tl.g, o);
          shift(tl.g, sx(i), isNew ? -8 * (1 - o) : 0);
          const busy = isNew || (k === 1 && i === 0);
          tileState(tl, busy ? hot : 0, p01(t, c0, c1), isNew ? "run" : "re", isNew && t < c0 ? null : "kv");
        });
        show(r.arcs, 0.9 * Math.min(ep(t, c0 - 0.1, c0), 1 - ep(t, c1, c1 + 0.15)));
        r.notes.forEach((n, i) => show(n, ep(t, c1 + 0.1 * i, c1 + 0.3 + 0.1 * i)));
      });
    };
  }

  register('[data-anim="v3start"]', fig => timeline(fig, {
    W: 1000, H: 316, duration: 9.5, build: v3Scene, legend: legend("hot", "kv", "empty"),
    chapters: [
      { t: 0, label: "z0", say: T(
        "<b>z0 到达</b>：窗口里只有它自己，只算这一格，马上输出第 0 帧。这一格的 K/V 是独自算出来的，没有任何上下文。",
        "<b>z0 arrives</b>: it is alone in the window, so only its slot is computed, and frame 0 is out right away. Its K/V is computed alone, with no context at all.") },
      { t: 3, label: "z1", say: T(
        "<b>z1 到达</b>：V1、V2 在这一步只算 z1，读的是 z0 独自算出来的 K/V；V3 把 slot 0 和 z1 一起重算，cache 里 slot 0 的 K/V 换成看得见 z1 的这一份。第 0 帧已经输出，不再修改；z1 的 4 帧照常输出。",
        "<b>z1 arrives</b>: V1 and V2 compute only z1 here and read the K/V that z0 computed alone. V3 recomputes slot 0 together with z1, and slot 0's K/V in the cache is replaced by this version, which has seen z1. Frame 0 has already been shown and stays as it is; z1's 4 frames are out as usual.") },
      { t: 6, label: "z2", say: T(
        "<b>z2 到达</b>：只算 z2 这一格，attention 读 slot 0、1 缓存的 K/V（上方的弧线），z0、z1 都不再重算。从这里开始和 V1、V2 完全一样，整段视频只在 z1 那一步多算了 1 格。",
        "<b>z2 arrives</b>: only z2's slot is computed, attention reads the cached K/V of slots 0 and 1 (the arcs above), and neither z0 nor z1 is recomputed. From here on it is exactly the same as V1 and V2; the whole video costs one extra slot, in the z1 step.") },
    ],
  }), false);
})();
