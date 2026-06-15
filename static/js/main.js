/* ==========================================================================
   GAM project page — interactions
   ========================================================================== */
(function () {
  "use strict";

  /* ------------------------------------------------------- theme toggle */
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("gam-theme", next); } catch (_) {}
    });
  }

  /* ------------------------------------------------- nav + progress bar */
  const nav = document.querySelector(".nav");
  const bar = document.querySelector(".progress-bar");
  const onScroll = () => {
    const y = window.scrollY;
    nav.classList.toggle("is-scrolled", y > 8);
    const h = document.documentElement.scrollHeight - window.innerHeight;
    if (bar) bar.style.width = (h > 0 ? (y / h) * 100 : 0) + "%";
  };
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------------------------------------------------- scroll reveal */
  const revealIO = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          revealIO.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  document.querySelectorAll(".reveal").forEach((el) => revealIO.observe(el));

  /* ------------------------------------------------- latency bar chart */
  const latIO = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.querySelectorAll(".lat-fill").forEach((f) => {
          f.style.width = f.dataset.w;
          const val = f.parentElement.querySelector(".lat-val");
          if (val) val.style.setProperty("--w", f.dataset.w);
        });
        latIO.unobserve(e.target);
      }
    },
    { threshold: 0.35 }
  );
  document.querySelectorAll(".latency").forEach((el) => latIO.observe(el));

  /* ------------------------------------------------- SVG line charts */
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function animateLines(root, lines) {
    if (reduceMotion) return;
    lines.forEach((p) => {
      const len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
    });
    new IntersectionObserver((entries, io) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      lines.forEach((p, i) => {
        p.style.transition = `stroke-dashoffset 1.1s cubic-bezier(.2,.6,.2,1) ${i * 130}ms`;
        p.style.strokeDashoffset = 0;
      });
      io.disconnect();
    }, { threshold: 0.3 }).observe(root);
  }

  /* renders a line chart; no tooltips by design */
  function renderLineChart(root, cfg) {
    const { series, W, H, L, R: RR, T, B, gridStep = 20, labelStep = 20, dotR = 5, xTitle, yTitle } = cfg;
    const pw = W - L - RR, ph = H - T - B;
    const x = (i) => L + (pw * i) / 4;
    const y = (v) => T + ph * (1 - v / 100);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });

    for (let v = 0; v <= 100; v += gridStep) {
      svg.appendChild(svgEl("line", { class: "grid-line", x1: L, x2: W - RR, y1: y(v), y2: y(v) }));
      if (v % labelStep === 0) {
        const lbl = svgEl("text", { class: "axis-label", x: L - 8, y: y(v) + 4, "text-anchor": "end" });
        lbl.textContent = v;
        svg.appendChild(lbl);
      }
    }
    for (let i = 0; i < 5; i++) {
      const lbl = svgEl("text", { class: "axis-label", x: x(i), y: H - B + 22, "text-anchor": "middle" });
      lbl.textContent = "L" + (i + 1);
      svg.appendChild(lbl);
    }
    if (xTitle) {
      const xt = svgEl("text", { class: "axis-title", x: L + pw / 2, y: H - 5, "text-anchor": "middle" });
      xt.textContent = xTitle;
      svg.appendChild(xt);
    }
    if (yTitle) {
      const yt = svgEl("text", { class: "axis-title", "text-anchor": "middle", transform: `translate(14 ${T + ph / 2}) rotate(-90)` });
      yt.textContent = yTitle;
      svg.appendChild(yt);
    }

    const lines = [];
    series.forEach((s) => {
      const g = svgEl("g", { class: `series series--${s.key}` });
      const pts = s.values.map((v, i) => [x(i), y(v)]);
      const path = svgEl("path", { class: "series-line", d: "M" + pts.map((p) => p.join(",")).join(" L") });
      g.appendChild(path);
      lines.push(path);
      pts.forEach((p) => g.appendChild(svgEl("circle", { class: "series-dot", cx: p[0], cy: p[1], r: dotR })));
      svg.appendChild(g);
    });
    root.appendChild(svg);
    animateLines(root, lines);
  }

  /* ---- difficulty chart (Results section) ---- */
  const chartRoot = document.getElementById("difficulty-chart");
  if (chartRoot) {
    const SERIES = [
      { key: "rocket", name: "ROCKET", values: [63.5, 46.5, 28, 26, 9.5] },
      { key: "cosmos", name: "Cosmos Policy", values: [86.5, 81, 83, 73.5, 55] },
      { key: "pi", name: "π0.5", values: [97, 86.5, 78.5, 70.5, 45.5] },
      { key: "gam", name: "Ours (GAM)", values: [98, 96.5, 88.5, 84, 65.5] },
    ];
    renderLineChart(chartRoot, {
      series: SERIES, W: 560, H: 380, L: 52, R: 18, T: 16, B: 52,
      xTitle: "Difficulty Level", yTitle: "Success Rate (%)",
    });
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    legend.innerHTML = SERIES.slice().reverse().map((s) =>
      `<span class="leg ${s.key}"><i></i>${s.name}</span>`).join("");
    chartRoot.appendChild(legend);
  }

  /* ---- robustness mini-chart grid (values read from Fig. 9) ---- */
  const panelsRoot = document.getElementById("robustness-panels");
  if (panelsRoot) {
    const PANELS = [
      { name: "Average",    spatial: [100, 97, 95, 92, 75],   object: [100, 100, 96, 92, 75],  goal: [89, 91, 86, 79, 61],  long: [84, 87, 82, 85, 65] },
      { name: "Camera",     spatial: [99, 96, 90, 73, 92],    object: [100, 100, 100, 99, 78], goal: [97, 100, 94, 87, 92], long: [100, 96, 68, 77, 43] },
      { name: "Robot",      spatial: [100, 95, 84, 68, 50],   object: [100, 95, 84, 68, 38],   goal: [99, 90, 73, 65, 41],  long: [82, 87, 70, 49, 30] },
      { name: "Light",      spatial: [100, 100, 100, 100, 100], object: [100, 100, 100, 100, 100], goal: [95, 99, 100, 100, 100], long: [92, 90, 84, 94, 97] },
      { name: "Background", spatial: [98, 100, 100, 100, 100], object: [100, 100, 100, 100, 100], goal: [80, 87, 94, 99, 100], long: [100, 89, 86, 93, 86] },
      { name: "Noise",      spatial: [100, 98, 100, 98, 59],  object: [100, 100, 100, 100, 100], goal: [100, 98, 100, 98, 94], long: [94, 98, 94, 91, 82] },
      { name: "Layout",     spatial: [84, 97, 95, 95, 97],    object: [100, 97, 88, 65, 62],   goal: [90, 86, 69, 32, 26],  long: [78, 81, 86, 91, 34] },
      { name: "Language",   spatial: [100, 93, 98, 99, 90],   object: [100, 100, 100, 100, 100], goal: [57, 77, 84, 71, 53], long: [86, 65, 88, 99, 75] },
    ];
    const SUITES = ["spatial", "object", "goal", "long"];
    PANELS.forEach((p) => {
      const cell = document.createElement("div");
      cell.className = "mini-chart line-chart" + (p.name === "Average" ? " mini-chart--avg" : "");
      const h = document.createElement("h4");
      h.textContent = p.name;
      cell.appendChild(h);
      panelsRoot.appendChild(cell);
      renderLineChart(cell, {
        series: SUITES.map((s) => ({ key: `suite-${s}`, values: p[s] })),
        W: 230, H: 180, L: 30, R: 8, T: 8, B: 26,
        gridStep: 25, labelStep: 50, dotR: 3.5,
      });
    });
  }

  /* ---- pretraining mixture: donut + dataset bars (Fig. 5) ---- */
  const mixRoot = document.getElementById("mixture-chart");
  if (mixRoot) {
    const SOURCES = [
      { key: "oxe", name: "Open X-Embodiment", pct: 72 },
      { key: "mimicgen", name: "MimicGen", pct: 18 },
      { key: "robocasa", name: "RoboCasa365", pct: 10 },
    ];
    const DATASETS = [
      ["MimicGen", 18.0, "mimicgen"], ["DROID", 13.8, "oxe"], ["Fractal", 12.4, "oxe"],
      ["Bridge", 10.2, "oxe"], ["RoboCasa365", 10.0, "robocasa"], ["BC-Z", 9.5, "oxe"],
      ["Language Table", 4.6, "oxe"], ["Furniture Bench", 3.2, "oxe"], ["TACO Play", 2.7, "oxe"],
      ["KUKA", 2.3, "oxe"], ["FMB", 1.9, "oxe"], ["UT Austin Mutex", 1.8, "oxe"],
      ["Jaco Play", 1.5, "oxe"], ["Berkeley UR5", 1.5, "oxe"], ["Stanford Hydra", 1.1, "oxe"],
      ["Austin Sirius", 1.1, "oxe"], ["RoboTurk", 0.9, "oxe"], ["Berkeley Fanuc", 0.9, "oxe"],
      ["Austin Sailor", 0.7, "oxe"], ["NYU Door Opening", 0.5, "oxe"], ["NYU Franka Play", 0.5, "oxe"],
      ["Berkeley Cable Routing", 0.4, "oxe"], ["Austin BUDS", 0.3, "oxe"], ["CMU Stretch", 0.2, "oxe"],
      ["DLR EDAN", 0.05, "oxe"],
    ];

    // donut
    const side = document.createElement("div");
    side.className = "mix-donut";
    const svg = svgEl("svg", { viewBox: "0 0 200 200", "aria-hidden": "true" });
    const CX = 100, CY = 100, RAD = 74, SW = 30;
    const C = 2 * Math.PI * RAD;
    const GAP = 5;
    let offset = 0;
    SOURCES.forEach((s) => {
      const seg = C * s.pct / 100;
      const arc = svgEl("circle", {
        class: `mix-arc mix--${s.key}`, cx: CX, cy: CY, r: RAD, fill: "none",
        "stroke-width": SW, "stroke-linecap": "round",
        "stroke-dasharray": `${Math.max(seg - GAP, 2)} ${C - seg + GAP}`,
        "stroke-dashoffset": -offset,
        transform: `rotate(-90 ${CX} ${CY})`,
      });
      svg.appendChild(arc);
      offset += seg;
    });
    const t1 = svgEl("text", { class: "mix-center-num", x: CX, y: CY - 2, "text-anchor": "middle" });
    t1.textContent = "784K";
    const t2 = svgEl("text", { class: "mix-center-sub", x: CX, y: CY + 20, "text-anchor": "middle" });
    t2.textContent = "trajectories";
    svg.appendChild(t1); svg.appendChild(t2);
    side.appendChild(svg);
    const slegend = document.createElement("div");
    slegend.className = "mix-legend";
    slegend.innerHTML = SOURCES.map((s) =>
      `<span class="leg"><i class="mix--${s.key}"></i>${s.name}<b>${s.pct}%</b></span>`).join("");
    side.appendChild(slegend);
    mixRoot.appendChild(side);

    // bars
    const bars = document.createElement("div");
    bars.className = "mix-bars";
    const maxV = DATASETS[0][1];
    bars.innerHTML = DATASETS.map(([name, v, grp]) => `
      <div class="mix-row">
        <span class="mix-name">${name}</span>
        <span class="mix-track"><i class="mix--${grp}" style="--bw:${(v / maxV * 100).toFixed(1)}%"></i></span>
        <span class="mix-val">${v >= 0.1 ? v.toFixed(1) : v}</span>
      </div>`).join("");
    mixRoot.appendChild(bars);

    if (!reduceMotion) {
      new IntersectionObserver((entries, io) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        bars.classList.add("is-in");
        io.disconnect();
      }, { threshold: 0.15 }).observe(bars);
    } else {
      bars.classList.add("is-in");
    }
  }

  /* ------------------------------------------------------ video chapters */
  const video = document.getElementById("rw-video");
  const chipWrap = document.getElementById("chapters");
  if (video && chipWrap) {
    const chapters = [
      { t: 0, label: "Intro" },
      { t: 6, label: "T1 · Pick & place" },
      { t: 24, label: "T1 · OOD" },
      { t: 42, label: "T2 · Stack milk & cube" },
      { t: 72, label: "T2 · OOD" },
      { t: 102, label: "T3 · Pot & pan" },
      { t: 126, label: "T3 · OOD" },
      { t: 150, label: "T4 · Insert cube" },
      { t: 168, label: "T4 · OOD" },
    ];
    const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    const chips = chapters.map(({ t, label }) => {
      const b = document.createElement("button");
      b.className = "chapter-chip";
      b.innerHTML = `${label}<span class="t">${fmt(t)}</span>`;
      b.addEventListener("click", () => {
        video.currentTime = t;
        video.play();
        video.scrollIntoView({ behavior: "smooth", block: "center" });
      });
      chipWrap.appendChild(b);
      return { t, el: b };
    });
    video.addEventListener("timeupdate", () => {
      const cur = video.currentTime;
      let active = 0;
      chips.forEach((c, i) => { if (cur >= c.t - 0.25) active = i; });
      chips.forEach((c, i) => c.el.classList.toggle("is-active", i === active));
    });
  }

  /* --------------------------------------------------------- carousels */
  function initCarousel(root) {
    const track = root.querySelector(".car-track");
    const prev = root.querySelector("[data-car-prev]");
    const next = root.querySelector("[data-car-next]");
    const dotWrap = root.querySelector(".car-dots");
    const slides = Array.from(track.children);
    if (!slides.length) return;

    let dots = [];
    if (dotWrap) {
      dots = slides.map((_, i) => {
        const d = document.createElement("button");
        d.className = "car-dot";
        d.setAttribute("aria-label", `Go to slide ${i + 1}`);
        d.addEventListener("click", () => {
          track.scrollTo({ left: slides[i].offsetLeft - track.offsetLeft, behavior: "smooth" });
        });
        dotWrap.appendChild(d);
        return d;
      });
    }

    const current = () => {
      const x = track.scrollLeft + track.clientWidth / 2;
      let best = 0, bestDist = Infinity;
      slides.forEach((s, i) => {
        const c = s.offsetLeft - track.offsetLeft + s.offsetWidth / 2;
        const dist = Math.abs(c - x);
        if (dist < bestDist) { bestDist = dist; best = i; }
      });
      return best;
    };

    const update = () => {
      const i = current();
      dots.forEach((d, k) => d.classList.toggle("is-active", k === i));
      const max = track.scrollWidth - track.clientWidth - 4;
      if (prev) prev.disabled = track.scrollLeft <= 4;
      if (next) next.disabled = track.scrollLeft >= max;
    };

    const step = () => Math.max(slides[0].offsetWidth * 0.9, track.clientWidth * 0.7);
    if (prev) prev.addEventListener("click", () => track.scrollBy({ left: -step(), behavior: "smooth" }));
    if (next) next.addEventListener("click", () => track.scrollBy({ left: step(), behavior: "smooth" }));
    track.addEventListener("scroll", () => requestAnimationFrame(update), { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  /* ------------------------------------------------- rollout explorer */
  const rolloutGroups = document.getElementById("rollout-groups");
  const suiteTabs = document.getElementById("suite-tabs");
  if (rolloutGroups && suiteTabs) {
    const SUITES = [
      { key: "spatial", label: "LIBERO-Spatial" },
      { key: "object", label: "LIBERO-Object" },
      { key: "goal", label: "LIBERO-Goal" },
      { key: "long", label: "LIBERO-Long" },
    ];
    const CAMERA_SUCCESS = {
      spatial: [99, 96, 90, 73, 92],
      object: [100, 100, 100, 99, 78],
      goal: [97, 100, 94, 87, 92],
      long: [100, 96, 68, 77, 43],
    };
    const cap = (s) => s[0].toUpperCase() + s.slice(1);
    const GROUPS = [
      {
        title: "LIBERO Original",
        desc: "Original benchmark episodes with the full external and wrist camera views.",
        items: ["task0", "task4", "task8"].map((k) => ({ key: k, label: "Task " + k.slice(4) })),
      },
      {
        title: "LIBERO-Plus Perturbations",
        desc: "Zero-shot rollouts under each perturbation factor, shown with the original two-view layout.",
        items: ["camera", "robot", "language", "light", "background", "noise", "layout"]
          .map((k) => ({ key: k, label: cap(k) })),
      },
      {
        title: "LIBERO-Plus Camera Difficulty",
        desc: "External-camera perturbation only. The wrist view is cropped out so viewpoint changes are easier to compare.",
        mode: "difficulty",
        items: [1, 2, 3, 4, 5].map((d) => ({
          key: `camera_d${d}`,
          label: `Level ${d}`,
          level: d,
          sub: d === 1 ? "easiest" : d === 5 ? "hardest" : "",
        })),
      },
    ];
    const arrow = (d) =>
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;

    GROUPS.forEach((g) => {
      const wrap = document.createElement("div");
      wrap.className = `rollout-group${g.mode ? ` rollout-group--${g.mode}` : ""}`;
      wrap.innerHTML = `
        <div class="rg-head"><h3>${g.title}</h3><p>${g.desc}</p></div>
        ${g.mode === "difficulty" ? `
          <div class="difficulty-scale" aria-hidden="true">
            <span>easier</span><i></i><span>harder</span>
          </div>` : ""}
        <div class="carousel" data-carousel>
          <div class="car-track">
            ${g.items.map((it) => `
              <div class="car-slide car-slide--vid">
                <article class="rollout-card${g.mode === "difficulty" ? " rollout-card--difficulty" : ""}">
                  <div class="rollout-thumb rollout-thumb--wide${g.mode === "difficulty" ? " rollout-thumb--external" : ""}">
                    <video data-key="${it.key}" muted loop playsinline preload="auto"
                           aria-label="${g.title} - ${it.label}"></video>
                  </div>
                  <div class="rollout-vbody">
                    <span class="vlabel">${it.label}</span>
                    <span class="vmeta">
                      ${g.mode === "difficulty" ? `<span class="success-pill" data-success-level="${it.level}"></span>` : ""}
                      ${it.sub ? `<span class="vsub">${it.sub}</span>` : ""}
                    </span>
                  </div>
                </article>
              </div>`).join("")}
          </div>
          <div class="car-nav">
            <button class="car-btn" data-car-prev aria-label="Previous">${arrow("M15 18l-6-6 6-6")}</button>
            <div class="car-dots"></div>
            <button class="car-btn" data-car-next aria-label="Next">${arrow("M9 6l6 6-6 6")}</button>
          </div>
        </div>`;
      rolloutGroups.appendChild(wrap);
    });

    const vids = Array.from(rolloutGroups.querySelectorAll("video[data-key]"));

    // play only while on screen
    const visible = new Set();
    const vio = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        const v = e.target;
        if (e.isIntersecting) {
          visible.add(v);
          if (v.src) v.play().catch(() => {});
        } else {
          visible.delete(v);
          v.pause();
        }
      });
    }, { threshold: 0.2 });
    vids.forEach((v) => vio.observe(v));

    let activeSuite = null;
    const setSuite = (key) => {
      if (key === activeSuite) return;
      activeSuite = key;
      suiteTabs.querySelectorAll("button").forEach((b) =>
        b.classList.toggle("is-active", b.dataset.suite === key));
      vids.forEach((v) => {
        v.src = `static/videos/rollouts/${key}_${v.dataset.key}.mp4`;
        if (visible.has(v)) v.play().catch(() => {});
      });
      rolloutGroups.querySelectorAll("[data-success-level]").forEach((el) => {
        const level = Number(el.dataset.successLevel);
        const value = CAMERA_SUCCESS[key] ? CAMERA_SUCCESS[key][level - 1] : null;
        el.textContent = value == null ? "" : `${value}%`;
      });
    };

    SUITES.forEach((s) => {
      const b = document.createElement("button");
      b.className = "suite-tab";
      b.dataset.suite = s.key;
      b.textContent = s.label;
      b.addEventListener("click", () => setSuite(s.key));
      suiteTabs.appendChild(b);
    });
    setSuite("spatial");
  }

  document.querySelectorAll("[data-carousel]").forEach(initCarousel);

  /* ----------------------------------------------------------- lightbox */
  const lb = document.querySelector(".lightbox");
  if (lb) {
    const lbImg = lb.querySelector("img");
    const lbCap = lb.querySelector(".lb-cap");
    const close = () => {
      lb.classList.remove("is-open");
      document.body.style.overflow = "";
    };
    document.querySelectorAll("[data-zoom]").forEach((img) => {
      img.addEventListener("click", () => {
        lbImg.src = img.src;
        lbImg.alt = img.alt || "";
        lbCap.textContent = img.alt || "";
        lb.classList.add("is-open");
        document.body.style.overflow = "hidden";
      });
    });
    lb.addEventListener("click", (e) => { if (e.target !== lbImg) close(); });
    lb.querySelector(".lb-close").addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  }

  /* -------------------------------------------------------- bibtex copy */
  const copyBtn = document.querySelector(".copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = document.getElementById("bibtex-text").innerText;
      try {
        await navigator.clipboard.writeText(text);
        const orig = copyBtn.innerHTML;
        copyBtn.innerHTML = "Copied!";
        setTimeout(() => (copyBtn.innerHTML = orig), 1600);
      } catch (_) { /* clipboard unavailable */ }
    });
  }
})();
