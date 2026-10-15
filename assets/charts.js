/* Data charts (SVG) and the latency simulation. Data: window.BLOG_DATA. Charts re-render on a language switch. */
(() => {
  "use strict";
  const { D, VAR, T, tr, h, s, bi, seg, svgBox, axes, legend, clamp, num, sgn, range, pathOf, register } = H3;
  const W = 960;
  const mean = a => a.reduce((p, q) => p + q, 0) / a.length;
  const controls = elm => h("div", { class: "fig-controls" }, elm);
  const vline = (svg, x, y1, y2, dash = "3 4") => s("line", { x1: x, x2: x, y1, y2, stroke: "var(--line-2)", "stroke-dasharray": dash }, svg);

  /* Per-frame mean-luma offset against the official decode, one clip at a time. */
  function offsetChart(elm) {
    const variants = (elm.dataset.variants || "v1").split(",");
    const clips = D.mcl.player;
    const st = elm._st || (elm._st = { clip: elm.dataset.clip || clips[0] });
    elm.textContent = "";
    seg(controls(elm), clips.map(c => ({ key: c, label: c })), st.clip, c => { st.clip = c; offsetChart(elm); });
    const series = D.mcl.series[st.clip];
    const n = series[variants[0]].length;
    const lim = +(elm.dataset.lim || 4);
    const H = 250, m = { l: 46, r: 16, t: 14, b: 40 };
    const svg = svgBox(elm, W, H);
    const { X, Y } = axes(svg, {
      W, H, m,
      x: { min: 0, max: n - 1, ticks: range(0, n - 1, 17), label: T("帧号（虚线是每个 17 帧 chunk 的第一帧）", "frame (dashed: first frame of each 17-frame chunk)") },
      y: { min: -lim, max: lim, ticks: range(-lim, lim, lim > 3 ? 2 : 1), fmt: v => sgn(v, 0), label: T("相对官方（灰阶）", "vs official (levels)") },
    });
    for (let f = 17; f < n; f += 17) vline(svg, X(f), m.t, H - m.b);
    for (const v of variants) {
      const ys = series[v], color = VAR[v].color;
      s("path", { d: pathOf(ys.map((y, f) => [X(f), Y(clamp(y, -lim, lim))])), fill: "none", stroke: color, "stroke-width": 1.6, "stroke-linejoin": "round" }, svg);
      ys.forEach((y, f) => {
        if (Math.abs(y) <= lim) return;
        const yy = Y(y > 0 ? lim : -lim);
        s("circle", { cx: X(f), cy: yy, r: 3, fill: color }, svg);
        s("text", { x: X(f) + 7, y: y > 0 ? yy + 12 : yy - 5, fill: color }, svg, sgn(y, 1));
      });
    }
    legend(elm, variants.map(v => ({ label: T(`${VAR[v].zh} − 官方 decode`, `${VAR[v].en} − official decode`), color: VAR[v].color })));
  }
  register('[data-chart="offset"]', offsetChart);

  /* Mean offset at each of the 17 frame positions of a chunk, grouped by the slot that produced it. */
  const SLOT_BANDS = [[0, 0, 5], [1, 4, 6], [5, 8, 2], [9, 12, 3], [13, 16, 4]];

  function profileChart(elm) {
    const keys = elm.dataset.sets.split(",").map(k => k.split(":")[1]);
    const perclip = !!elm.dataset.perclip;
    elm.textContent = "";
    const H = 290, m = { l: 46, r: 16, t: 38, b: 42 };
    const vals = keys.flatMap(v => D.mcl.profile[v]).concat(perclip ? D.mcl.profile_clips_v1.flat() : []);
    const hi = Math.max(1, Math.ceil(Math.max(...vals) * 2) / 2);
    const lo = Math.min(-1, Math.floor(Math.min(...vals) * 2) / 2);
    const svg = svgBox(elm, W, H);
    const BX = r => m.l + ((W - m.l - m.r) * (r + 0.5)) / 17;
    SLOT_BANDS.forEach(([a, b, slot], i) => {
      if (i % 2) s("rect", { x: BX(a - 0.5), y: m.t - 22, width: BX(b + 0.5) - BX(a - 0.5), height: H - m.b - m.t + 22, fill: "var(--band)" }, svg);
      s("text", { x: (BX(a - 0.5) + BX(b + 0.5)) / 2, y: m.t - 8, "text-anchor": "middle", class: "t-title" }, svg, `slot ${slot}`);
    });
    const { X, Y } = axes(svg, {
      W, H, m,
      x: { min: -0.5, max: 16.5, ticks: range(0, 16), label: T("chunk 内的帧位置（0 是 chunk 的第一帧）", "frame position inside the chunk (0 is its first frame)") },
      y: { min: lo, max: hi, ticks: range(lo, hi, 0.5), fmt: v => sgn(v, 1), label: T("相对官方（灰阶）", "vs official (levels)") },
    });
    if (perclip) {
      for (const c of D.mcl.profile_clips_v1) {
        s("path", { d: pathOf(c.map((y, r) => [X(r), Y(y)])), fill: "none", stroke: VAR.v1.color, "stroke-width": 1, opacity: 0.16 }, svg);
      }
    }
    const bw = 0.7 / keys.length;
    keys.forEach((v, j) => {
      const last = j === keys.length - 1;
      D.mcl.profile[v].forEach((y, r) => {
        const x0 = X(r - 0.35 + j * bw), x1 = X(r - 0.35 + (j + 1) * bw);
        const y0 = Y(0), y1 = Y(y);
        s("rect", { x: x0 + 0.5, y: Math.min(y0, y1), width: Math.max(1, x1 - x0 - 1), height: Math.max(1, Math.abs(y1 - y0)), rx: 1.5, fill: VAR[v].color, opacity: last ? 0.9 : 0.4 }, svg);
      });
      if (!last) return;
      for (const [a, b, slot] of SLOT_BANDS) {
        const mu = D.mcl.slot_stats[v][slot].mean;
        const ys = D.mcl.profile[v].slice(a, b + 1);
        const y = mu >= 0 ? Y(Math.max(0, ...ys)) - 7 : Y(Math.min(0, ...ys)) + 15;
        s("text", { x: (BX(a - 0.5) + BX(b + 0.5)) / 2, y, "text-anchor": "middle", class: "t-strong" }, svg, sgn(mu, 2));
      }
    });
    const items = keys.map((v, j) => ({ label: VAR[v], color: VAR[v].color, kind: "box", opacity: j === keys.length - 1 ? 1 : 0.45 }));
    if (perclip) items.push({ label: T("细线：每段视频", "thin lines: individual clips"), color: VAR.v1.color, opacity: 0.5 });
    items.push({ label: T(`数字：${VAR[keys[keys.length - 1]].zh} 每个 slot 的平均`, `numbers: ${VAR[keys[keys.length - 1]].en} mean per slot`), color: "transparent" });
    legend(elm, items);
  }
  register('[data-chart="profile"]', profileChart);

  /* The probed slot's offset follows the share of chunk-start positions (slots 0 and 5) among what it can see. */
  function shareChart(elm) {
    elm.textContent = "";
    const k = D.share.scale, base = 2 / 7;
    const H = 330, m = { l: 46, r: 16, t: 18, b: 44 };
    const svg = svgBox(elm, W, H);
    const { X, Y } = axes(svg, {
      W, H, m,
      x: { min: -0.02, max: 0.53, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt: v => v.toFixed(1), label: T("被测 slot 看得见的位置里，段首位置（slot 0、5）所占的比例", "share of chunk-start positions (slots 0 and 5) among the slots the probed slot can see") },
      y: { min: -3, max: 3, ticks: range(-3, 3, 1), fmt: v => sgn(v, 0), label: T("亮度偏移（灰阶）", "offset (levels)") },
    });
    s("line", { x1: X(base), x2: X(base), y1: m.t, y2: H - m.b, stroke: "var(--line-2)" }, svg);
    s("text", { x: X(base) + 6, y: m.t + 10 }, svg, tr(T("完整窗口：2/7", "full window: 2/7")));
    s("line", { x1: X(-0.02), y1: Y(k * (-0.02 - base)), x2: X(0.53), y2: Y(k * (0.53 - base)), stroke: "var(--muted)", "stroke-width": 1.3, "stroke-dasharray": "6 5" }, svg);
    const labels = new Map();
    for (const r of D.share.rows) {
      const px = X(r.share), py = Y(clamp(r.g, -3, 3));
      const color = r.adj ? "var(--faint)" : r.kind === "trunc" ? VAR.v1.color : "var(--accent)";
      const a = { fill: r.adj ? "none" : color, stroke: color, "stroke-width": 1.3, opacity: r.adj ? 1 : 0.85 };
      if (r.window === "W0") s("circle", { cx: px, cy: py, r: 4, ...a }, svg);
      else if (r.window === "Wstatic") s("rect", { x: px - 3.6, y: py - 3.6, width: 7.2, height: 7.2, ...a }, svg);
      else s("path", { d: `M${px},${py - 5}L${px + 5},${py}L${px},${py + 5}L${px - 5},${py}Z`, ...a }, svg);
      if (r.kind === "trunc" && r.window === "W0") {
        const key = r.share.toFixed(3), l = labels.get(key) || { px, ys: [], slots: [] };
        l.ys.push(py);
        l.slots.push(r.slot);
        labels.set(key, l);
      }
    }
    for (const l of labels.values()) {
      s("text", { x: l.px + 9, y: mean(l.ys) - 7, fill: VAR.v1.color }, svg, "slot " + [...new Set(l.slots)].sort().join(" / "));
    }
    const fitted = D.share.rows.filter(r => !r.adj).length;
    s("text", { x: W - m.r - 4, y: m.t + 10, "text-anchor": "end", class: "t-title" }, svg,
      `${tr(T("偏移", "offset"))} ≈ ${num(k, 1)} × (${tr(T("比例", "share"))} − 2/7) · r = ${num(D.share.r, 2)} · n = ${fitted}`);
    legend(elm, [
      { label: T("流式截断：slot s 只看得见 0..s，也就是 V1 算 slot s 时的情形", "streaming truncation: slot s sees only 0..s, which is V1's situation"), color: VAR.v1.color, kind: "dot" },
      { label: T("其他遮挡方式", "other masks"), color: "var(--accent)", kind: "dot" },
      { label: T("遮住的是紧挨着的 slot 0（不参与拟合）", "the hidden slot is the adjacent slot 0 (not fitted)"), kind: "hollow" },
      { label: T("拟合直线", "fitted line"), color: "var(--muted)", dash: true },
      { label: T("形状：● 原窗口 · ◆ slot 0、5 换成普通 latent · ■ 7 格放同一个 latent", "shapes: ● original window · ◆ slots 0 and 5 swapped for ordinary latents · ■ the same latent in all 7 slots"), color: "transparent" },
    ]);
  }
  register('[data-chart="share"]', shareChart);

  /* Offset against local brightness for one slot, on the held-out clips. */
  function gainChart(elm) {
    const st = elm._st || (elm._st = { slot: elm.dataset.slot || "4" });
    elm.textContent = "";
    seg(controls(elm), ["3", "4", "5"].map(k => ({ key: k, label: `slot ${k}` })), st.slot, k => { st.slot = k; gainChart(elm); });
    const sc = D.linear.scatter[st.slot], meta = D.linear.slots[st.slot];
    const ys = [...sc.y].sort((a, b) => a - b);
    const q = p => ys[Math.floor(p * (ys.length - 1))];
    let y0 = Math.min(q(0.015), ...sc.bins.map(b => b[2]), 0), y1 = Math.max(q(0.985), ...sc.bins.map(b => b[3]), 0);
    const pad = (y1 - y0) * 0.08;
    y0 = Math.floor((y0 - pad) * 2) / 2;
    y1 = Math.ceil((y1 + pad) * 2) / 2;
    const stepY = y1 - y0 > 4 ? 1 : 0.5;
    const H = 310, m = { l: 50, r: 16, t: 14, b: 44 };
    const svg = svgBox(elm, W, H);
    const { X, Y } = axes(svg, {
      W, H, m,
      x: { min: 0, max: 255, ticks: [0, 50, 100, 150, 200, 250], label: T("64×64 区域的亮度（8-bit luma）", "brightness of the 64×64 cell (8-bit luma)") },
      y: { min: y0, max: y1, ticks: range(Math.ceil(y0 / stepY) * stepY, y1, stepY), fmt: v => sgn(v, 1), label: T("流式 − 官方（灰阶）", "stream − official (levels)") },
    });
    sc.x.forEach((x, i) => s("circle", { cx: X(x), cy: Y(clamp(sc.y[i], y0, y1)), r: 1.8, fill: "var(--accent)", opacity: 0.22 }, svg));
    s("line", { x1: X(0), x2: X(255), y1: Y(meta.mean_offset_train), y2: Y(meta.mean_offset_train), stroke: "var(--muted)", "stroke-width": 1.3, "stroke-dasharray": "6 5" }, svg);
    const [a, b] = sc.fit;
    s("line", { x1: X(0), x2: X(255), y1: Y(clamp(a, y0, y1)), y2: Y(clamp(a + 255 * b, y0, y1)), stroke: VAR.v2.color, "stroke-width": 2 }, svg);
    for (const [bx, mu, lo, hi] of sc.bins) {
      s("line", { x1: X(bx), x2: X(bx), y1: Y(lo), y2: Y(hi), stroke: "var(--text)", "stroke-width": 1.2, opacity: 0.6 }, svg);
      s("circle", { cx: X(bx), cy: Y(mu), r: 3.4, fill: "var(--bg)", stroke: "var(--text)", "stroke-width": 1.5 }, svg);
    }
    legend(elm, [
      { label: T("点：64×64 区域（留出的视频里随机 700 个）", "dots: 64×64 cells (700 random, held-out clips)"), color: "var(--accent)", kind: "dot", opacity: 0.6 },
      { label: T("圆圈与竖线：按亮度分组的均值和 25–75% 分位", "rings and bars: binned mean and 25–75% range"), kind: "hollow" },
      { label: T("a + b·x（SRC01–14 拟合）", "a + b·x (fitted on SRC01–14)"), color: VAR.v2.color },
      { label: T("只减一个常数", "a single constant"), color: "var(--muted)", dash: true },
    ]);
    const rd = h("div", { class: "readings" }, elm);
    h("span", { html: tr(T(
      `slot ${st.slot}，留出的 SRC15–30 上帧平均残差 RMS：不校正 <b>${meta.frame_rms_none.toFixed(2)}</b>，减常数 <b>${meta.frame_rms_const.toFixed(2)}</b>，减 a + b·x <b>${meta.frame_rms_prior.toFixed(2)}</b>`,
      `slot ${st.slot}, frame-mean residual RMS on held-out SRC15–30: none <b>${meta.frame_rms_none.toFixed(2)}</b>, constant <b>${meta.frame_rms_const.toFixed(2)}</b>, a + b·x <b>${meta.frame_rms_prior.toFixed(2)}</b>`)) }, rd);
  }
  register('[data-chart="gain"]', gainChart);

  /* Offset over the first window of a video: the start-up transient. */
  function firstChart(elm) {
    const variants = (elm.dataset.variants || "v1,v2,v3").split(",");
    elm.textContent = "";
    const F = D.mcl.first_window, n = F[variants[0]].mean.length;
    const lo = +(elm.dataset.lo || -4), hi = +(elm.dataset.hi || 8);
    const H = 300, m = { l: 46, r: 16, t: 34, b: 44 };
    const svg = svgBox(elm, W, H);
    const { X, Y } = axes(svg, {
      W, H, m,
      x: { min: -0.5, max: n - 0.5, ticks: range(0, n - 1), label: T("视频开头的帧号", "frame index from the start of the video") },
      y: { min: lo, max: hi, ticks: range(lo, hi, 2), fmt: v => sgn(v, 0), label: T("相对官方（灰阶）", "vs official (levels)") },
    });
    for (let t = 0, f = 0; f < n; t++) {
      const next = Math.min(n, f + (t % 5 === 0 ? 1 : 4));
      if (f > 0) vline(svg, X(f - 0.5), m.t - 16, H - m.b);
      const attrs = { x: X((f + next - 1) / 2), y: m.t - 18, "text-anchor": "middle" };
      if (t % 5 === 0) attrs.fill = "var(--accent)";
      s("text", attrs, svg, "z" + t);
      f = next;
    }
    for (const v of variants) {
      const g = F[v], color = VAR[v].color;
      const up = g.p90.map((y, f) => [X(f), Y(clamp(y, lo, hi))]);
      const dn = g.p10.map((y, f) => [X(f), Y(clamp(y, lo, hi))]).reverse();
      s("path", { d: pathOf(up.concat(dn)) + "Z", fill: color, opacity: 0.1 }, svg);
      s("path", { d: pathOf(g.mean.map((y, f) => [X(f), Y(clamp(y, lo, hi))])), fill: "none", stroke: color, "stroke-width": 1.7, "stroke-linejoin": "round" }, svg);
      g.mean.forEach((y, f) => {
        s("circle", { cx: X(f), cy: Y(clamp(y, lo, hi)), r: 2.4, fill: color }, svg);
        if (y >= lo && y <= hi) return;
        s("text", { x: X(f) + 7, y: y < lo ? Y(lo) - 6 : Y(hi) + 13, fill: color }, svg, `${VAR[v].en} ${sgn(y, 1)}`);
      });
    }
    const items = variants.map(v => ({ label: VAR[v], color: VAR[v].color }));
    items.push({ label: T("色带：30 段的 10–90% 分位", "bands: 10–90% range over 30 clips"), color: "var(--faint)", kind: "box", opacity: 0.5 });
    legend(elm, items);
  }
  register('[data-chart="first"]', firstChart);

  /* SolarWM: two metrics per subset, V1 against V3. */
  function subsetChart(elm) {
    elm.textContent = "";
    const subs = D.solar.subsets;
    const rowH = 26, m = { l: 150, r: 34, t: 36, b: 28 };
    const H = m.t + m.b + rowH * subs.length;
    const svg = svgBox(elm, W, H);
    const gap = 48, pw = (W - m.l - m.r - gap) / 2;
    const panels = [
      { key: "steady_p95", x0: m.l, min: 0, max: 3.5, ticks: [0, 1, 2, 3], title: T("稳态亮度跳变 p95（越低越好）", "steady luma-jump p95 (lower is better)") },
      { key: "jump01", x0: m.l + pw + gap, min: -3, max: 21, ticks: [0, 5, 10, 15, 20], title: T("第 0→1 帧亮度跳变（越接近 0 越好）", "frame 0→1 luma jump (closer to 0 is better)") },
    ];
    subs.forEach((d, i) => {
      if (i % 2) s("rect", { x: 0, y: m.t + i * rowH, width: W, height: rowH, fill: "var(--band)" }, svg);
      s("text", { x: m.l - 14, y: m.t + i * rowH + 17, "text-anchor": "end", class: "t-strong" }, svg, d);
    });
    for (const p of panels) {
      const X = v => p.x0 + (pw * (clamp(v, p.min, p.max) - p.min)) / (p.max - p.min);
      s("text", { x: p.x0, y: 14, class: "t-title" }, svg, tr(p.title));
      for (const t of p.ticks) {
        s("line", { x1: X(t), x2: X(t), y1: m.t - 6, y2: H - m.b, class: t === 0 ? "axis" : "grid" }, svg);
        s("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, svg, String(t));
      }
      subs.forEach((d, i) => {
        const y = m.t + i * rowH;
        ["v1", "v3"].forEach((v, j) => {
          const val = D.solar.per_subset[v][d][p.key];
          const xa = X(Math.min(0, val)), xb = X(Math.max(0, val));
          s("rect", { x: xa, y: y + 5 + j * 9, width: Math.max(1.5, xb - xa), height: 7, rx: 1.5, fill: VAR[v].color }, svg);
          s("text", { x: val >= 0 ? xb + 5 : xa - 5, y: y + 11.5 + j * 9, "text-anchor": val >= 0 ? "start" : "end", "font-size": 10, fill: VAR[v].color }, svg, num(val, 2));
        });
      });
    }
    legend(elm, [{ label: VAR.v1, color: VAR.v1.color, kind: "box" }, { label: VAR.v3, color: VAR.v3.color, kind: "box" }]);
  }
  register('[data-chart="solar-subsets"]', subsetChart);

  /* When does each frame become visible? Official windowed decode against the per-latent stream (V3). */
  function latencyChart(elm) {
    const st = elm._st || (elm._st = { mode: "latent", T: 0.6 });
    elm.textContent = "";
    const ctl = controls(elm);
    seg(ctl, [
      { key: "latent", label: T("逐 latent 生成", "latent by latent") },
      { key: "chunk", label: T("按 chunk 生成（SolarWM）", "chunk by chunk (SolarWM)") },
    ], st.mode, k => { st.mode = k; st.T = k === "chunk" ? 3 : 0.6; latencyChart(elm); });
    const grp = h("div", { class: "control-group" }, ctl);
    bi(h("span", null, grp), T("每个 latent 的生成耗时", "generation time per latent"));
    const sc = h("div", { class: "scrub", style: "min-width:150px;max-width:220px" }, grp);
    const inp = h("input", { type: "range", min: 0.1, max: 4, step: 0.05, value: st.T, "aria-label": "seconds per latent" }, sc);
    const out = h("span", { class: "readout" }, grp);
    const host = h("div", null, elm);
    const rd = h("div", { class: "readings" }, elm);
    inp.addEventListener("input", () => { st.T = +inp.value; draw(); });

    const N = 27;
    const first = D.mcl.timing.v3.emit_median.slice(0, 7);
    const slot = D.mcl.timing.v3.steady_slot_p50;
    const windowSec = D.mcl.timing.official.first;
    const frames = t => (t % 5 === 0 ? 1 : 4);
    const total = Array.from({ length: N }, (_, t) => frames(t)).reduce((a, b) => a + b, 0);

    function simulate() {
      const arrive = [];
      for (let t = 0; t < N; t++) arrive.push(st.mode === "latent" ? (t + 1) * st.T : Math.min(Math.floor(t / 5) * 5 + 5, N) * st.T);
      const ours = [];
      let free = 0;
      for (let t = 0; t < N; t++) {
        const cost = t < 7 ? first[t] : slot[String(((t - 7) % 5) + 2)];
        free = Math.max(arrive[t], free) + cost;
        ours.push(free);
      }
      const windows = Math.ceil((N + 3) / 5) - 1;
      const off = new Array(N).fill(0);
      let busy = 0;
      for (let k = 0; k < windows; k++) {
        busy = Math.max(arrive[Math.min(5 * k + 6, N - 1)], busy) + windowSec;
        for (let t = 5 * k; t < (k === windows - 1 ? N : 5 * k + 5); t++) off[t] = busy;
      }
      const stats = avail => {
        let sum = 0, cnt = 0, worst = 0;
        for (let t = 0; t < N; t++) {
          const d = avail[t] - arrive[t];
          sum += d * frames(t);
          cnt += frames(t);
          worst = Math.max(worst, d);
        }
        return { first: avail[0] - arrive[0], mean: sum / cnt, worst };
      };
      return { arrive, ours, off, so: stats(ours), sf: stats(off) };
    }

    function draw() {
      inp.style.setProperty("--fill", `${(100 * (st.T - 0.1)) / 3.9}%`);
      out.innerHTML = st.mode === "chunk" ? `<b>${st.T.toFixed(2)} s</b>&nbsp;·&nbsp;${(5 * st.T).toFixed(1)} s / chunk` : `<b>${st.T.toFixed(2)} s</b>`;
      const r = simulate();
      host.textContent = "";
      const H = 280, m = { l: 46, r: 16, t: 14, b: 42 };
      const tmax = Math.max(r.off[N - 1], r.ours[N - 1]) * 1.03;
      const svg = svgBox(host, W, H);
      const stepX = tmax > 120 ? 20 : tmax > 60 ? 10 : tmax > 30 ? 5 : tmax > 12 ? 2 : 1;
      const { X, Y } = axes(svg, {
        W, H, m,
        x: { min: 0, max: tmax, ticks: range(0, tmax, stepX), fmt: v => v + " s", label: T("从开始生成算起的时间", "time since generation started") },
        y: { min: 0, max: total, ticks: [0, 17, 34, 51, 68, 85], label: T("已能播放的帧数", "frames ready to show") },
      });
      const curve = avail => {
        const pts = [[X(0), Y(0)]];
        let cum = 0;
        avail.map((a, t) => [a, t]).sort((p, q) => p[0] - q[0]).forEach(([a, t]) => {
          pts.push([X(a), Y(cum)]);
          cum += frames(t);
          pts.push([X(a), Y(cum)]);
        });
        pts.push([X(tmax), Y(cum)]);
        return pathOf(pts);
      };
      s("path", { d: curve(r.arrive), fill: "none", stroke: "var(--faint)", "stroke-width": 1.2, "stroke-dasharray": "4 4" }, svg);
      s("path", { d: curve(r.off), fill: "none", stroke: VAR.official.color, "stroke-width": 1.8 }, svg);
      s("path", { d: curve(r.ours), fill: "none", stroke: VAR.v3.color, "stroke-width": 1.8 }, svg);
      legend(host, [
        { label: T("latent 生成出来的时刻（解码不花时间的理想情况）", "latents generated (ideal: decoding takes no time)"), color: "var(--faint)", dash: true },
        { label: VAR.official, color: VAR.official.color },
        { label: T("流式 V3", "stream, V3"), color: VAR.v3.color },
      ]);
      rd.textContent = "";
      const pair = (label, a, b) => h("span", {
        html: `${tr(label)}：<span class="dot official"></span> <b>${a.toFixed(2)} s</b>&nbsp;&nbsp;<span class="dot v3"></span> <b>${b.toFixed(2)} s</b>`,
      }, rd);
      pair(T("第一帧", "first frame"), r.sf.first, r.so.first);
      pair(T("每帧平均等待", "mean wait per frame"), r.sf.mean, r.so.mean);
      pair(T("最坏等待", "worst wait"), r.sf.worst, r.so.worst);
      const steady = [2, 3, 4, 5, 6].reduce((a, k) => a + slot[String(k)], 0) / 5;
      if (st.mode === "latent" && st.T < steady) {
        h("span", { style: "color:var(--faint)", html: tr(T(
          `生成比流式解码还快（稳态平均每个 latent ${steady.toFixed(2)} s），解码会排队，延迟随视频变长而累积。`,
          `The generator is faster than the stream decoder here (${steady.toFixed(2)} s per latent in steady state), so decoding queues up and latency grows with video length.`)) }, rd);
      }
    }
    draw();
  }
  register('[data-chart="latency"]', latencyChart);
})();
