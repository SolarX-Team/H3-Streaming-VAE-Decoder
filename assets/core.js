/* Shared helpers: language switch, DOM/SVG builders, number formatting, module registry. */
window.H3 = (() => {
  "use strict";
  const D = window.BLOG_DATA;
  const root = document.documentElement;
  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "\u2212";
  const langHooks = [];
  const modules = [];

  const store = {
    get: (k, fallback = null) => { try { const v = localStorage.getItem(k); return v === null ? fallback : v; } catch (e) { return fallback; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  const lang = () => (root.dataset.lang === "en" ? "en" : "zh");
  const T = (zh, en) => ({ zh, en });
  const tr = x => (x == null ? "" : typeof x === "object" ? (x[lang()] ?? x.zh) : String(x));
  const onLang = fn => (langHooks.push(fn), fn);

  function setLang(l, save = true) {
    root.dataset.lang = l === "en" ? "en" : "zh";
    root.lang = l === "en" ? "en" : "zh-CN";
    if (save) store.set("h3blog.lang", root.dataset.lang);
    document.querySelectorAll("[data-set-lang]").forEach(b => b.classList.toggle("on", b.dataset.setLang === root.dataset.lang));
    const meta = document.querySelector(`meta[name="title-${root.dataset.lang}"]`);
    if (meta) document.title = meta.content;
    if (save) {
      const u = new URL(location.href);
      u.searchParams.set("lang", root.dataset.lang);
      history.replaceState(null, "", u);
    }
    langHooks.forEach(fn => fn());
  }

  function h(tag, attrs, parent, text) {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "style") e.style.cssText = v;
      else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v === true ? "" : v);
    }
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  function s(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if ((k === "fill" || k === "stroke") && typeof v === "string" && v.startsWith("var(")) e.style.setProperty(k, v);
      else e.setAttribute(k, v);
    }
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  /* Append text in both languages; CSS shows the active one, so nothing needs re-rendering on a switch. */
  function bi(parent, x) {
    if (x == null) return parent;
    if (typeof x !== "object") { parent.append(String(x)); return parent; }
    h("span", { lang: "zh" }, parent, x.zh);
    h("span", { lang: "en" }, parent, x.en ?? x.zh);
    return parent;
  }

  function seg(parent, items, current, onPick) {
    const box = h("div", { class: "segmented", role: "group" }, parent);
    const sync = v => box.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.value === String(v)));
    for (const it of items) {
      const b = h("button", { type: "button", "data-value": String(it.key), title: it.title ? tr(it.title) : null }, box);
      bi(b, it.label);
      b.addEventListener("click", () => { b.blur(); sync(it.key); onPick(it.key); });
    }
    sync(current);
    return { el: box, sync };
  }

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const num = (v, d = 2) => (v < 0 && Math.abs(v) >= 0.5 * 10 ** -d ? MINUS : "") + Math.abs(v).toFixed(d);
  const sgn = (v, d = 2) => {
    const t = Math.abs(v).toFixed(d);
    if (+t === 0) return t;
    return (v > 0 ? "+" : MINUS) + t;
  };
  const range = (a, b, step = 1) => {
    const out = [];
    for (let i = 0; a + i * step <= b + 1e-9; i++) out.push(+(a + i * step).toFixed(10));
    return out;
  };
  const pathOf = pts => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("");

  const VAR = {
    src: { color: "var(--c-src)", zh: "原视频", en: "Source" },
    official: { color: "var(--c-off)", zh: "官方 decode", en: "Official decode" },
    v1: { color: "var(--c-v1)", zh: "V1", en: "V1" },
    v2: { color: "var(--c-v2)", zh: "V2", en: "V2" },
    v3: { color: "var(--c-v3)", zh: "V3", en: "V3" },
  };

  function svgBox(parent, W, H, cls = "chart") {
    const wrap = h("div", { class: "chart-scroll" }, parent);
    return s("svg", { class: cls, viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" }, wrap);
  }

  function axes(svg, o) {
    const { W, H, m, x, y } = o;
    const X = v => m.l + (W - m.l - m.r) * (v - x.min) / (x.max - x.min);
    const Y = v => H - m.b - (H - m.t - m.b) * (v - y.min) / (y.max - y.min);
    for (const t of y.ticks || []) {
      s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, svg, y.fmt ? y.fmt(t) : String(t));
    }
    for (const t of x.ticks || []) {
      if (o.xgrid) s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "grid" }, svg);
      s("text", { x: X(t), y: H - m.b + 17, "text-anchor": "middle" }, svg, x.fmt ? x.fmt(t) : String(t));
    }
    s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    if (o.zero !== false && y.min < 0 && y.max > 0) s("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    if (x.label) s("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "t-title" }, svg, tr(x.label));
    if (y.label) {
      const cy = (m.t + H - m.b) / 2;
      s("text", { x: 12, y: cy, transform: `rotate(-90 12 ${cy})`, "text-anchor": "middle", class: "t-title" }, svg, tr(y.label));
    }
    return { X, Y };
  }

  function legend(parent, items) {
    const box = h("div", { class: "legend" }, parent);
    for (const it of items) {
      const sp = h("span", null, box);
      const i = h("i", { class: it.kind || "" }, sp);
      if (it.kind !== "hollow") {
        i.style.background = it.dash ? `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)` : it.color;
      } else if (it.color) i.style.borderColor = it.color;
      if (it.opacity != null) i.style.opacity = it.opacity;
      sp.append(tr(it.label));
    }
    return box;
  }

  /* Modules re-render on a language switch unless they say they handle it themselves. */
  function register(selector, fn, relang = true) { modules.push([selector, fn, relang]); }

  function start() {
    const q = new URLSearchParams(location.search).get("lang");
    setLang(q || root.dataset.lang || "zh", !!q);
    document.querySelectorAll("[data-set-lang]").forEach(b => (b.onclick = () => setLang(b.dataset.setLang)));
    for (const [selector, fn, relang] of modules) {
      document.querySelectorAll(selector).forEach(elm => {
        try {
          const render = () => fn(elm);
          render();
          if (relang) onLang(render);
        } catch (err) {
          console.error(selector, err);
          elm.textContent = "render error: " + err.message;
        }
      });
    }
    navUi();
  }

  /* Top-bar section links mark the current h2; the side contents and the Method dropdown mark the current heading. */
  function navUi() {
    const heads = [...document.querySelectorAll(".toc a")].map(a => document.getElementById(a.hash.slice(1))).filter(Boolean);
    const links = [...document.querySelectorAll(".toc a, .nav a")];
    let ticking = false;
    const update = () => {
      ticking = false;
      let cur = "", sec = "";
      for (const el of heads) {
        if (el.getBoundingClientRect().top >= 160) break;
        cur = el.id;
        if (el.tagName === "H2") sec = el.id;
      }
      for (const a of links) a.classList.toggle("on", a.hash.slice(1) === (a.matches(".nav > a, .nav-item > a") ? sec : cur));
    };
    addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();

    /* Keep the dropdown shut after a pick until the pointer leaves, otherwise it hangs over the target. */
    document.querySelectorAll(".nav-item").forEach(item => {
      item.addEventListener("click", e => { if (e.target.closest(".nav-sub a")) item.classList.add("closed"); });
      item.addEventListener("mouseleave", () => item.classList.remove("closed"));
    });

    const toc = document.querySelector(".toc"), btn = document.querySelector(".toc-btn");
    if (!toc || !btn) return;
    const open = v => { toc.classList.toggle("open", v); btn.setAttribute("aria-expanded", String(v)); };
    btn.addEventListener("click", e => { e.stopPropagation(); open(!toc.classList.contains("open")); });
    document.addEventListener("click", e => {
      if (toc.classList.contains("open") && (!toc.contains(e.target) || e.target.closest("a"))) open(false);
    });
    document.addEventListener("keydown", e => { if (e.key === "Escape" && toc.classList.contains("open")) { open(false); btn.focus(); } });
  }

  /* isIntersecting is true from the first visible pixel, so compare the ratio against the threshold. */
  function whenVisible(elm, onShow, onHide, opts = {}) {
    if (!("IntersectionObserver" in window)) { onShow(); return; }
    const th = opts.threshold ?? 0.25;
    let state = null;
    new IntersectionObserver(entries => {
      for (const e of entries) {
        const vis = e.isIntersecting && e.intersectionRatio >= th - 1e-3;
        if (vis === state) continue;
        state = vis;
        if (vis) onShow();
        else if (onHide) onHide();
      }
    }, { threshold: th > 0 ? [0, th] : [0], rootMargin: opts.margin || "0px" }).observe(elm);
  }

  return {
    D, NS, MINUS, VAR, root, store, lang, T, tr, onLang, setLang, h, s, bi, seg, clamp, num, sgn, range, pathOf,
    svgBox, axes, legend, register, start, whenVisible,
  };
})();
