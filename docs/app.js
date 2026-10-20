(() => {
  const DATA = window.DEMO_DATA;
  if (!DATA) {
    console.error("DEMO_DATA missing — load data.js first");
    return;
  }

  const order = ["clean", "noise", "jpeg", "rotate", "blur", "occlude"];
  let scenarioId = DATA.scenarios[0].id;
  let perturb = "clean";

  const els = {
    scenario: document.getElementById("scenario"),
    perturbBtns: document.getElementById("perturb-btns"),
    scenarioNote: document.getElementById("scenario-note"),
    stageImg: document.getElementById("stage-img"),
    heroImg: document.getElementById("hero-img"),
    carrier: document.getElementById("carrier"),
    probeVal: document.getElementById("probe-val"),
    ocrVal: document.getElementById("ocr-val"),
    probeBar: document.getElementById("probe-bar"),
    ocrBar: document.getElementById("ocr-bar"),
    blurb: document.getElementById("blurb"),
    statRow: document.getElementById("stat-row"),
    cbBody: document.getElementById("cb-tbody"),
    takeaways: document.getElementById("takeaways"),
    cites: document.getElementById("cites"),
  };

  function scenario() {
    return DATA.scenarios.find((s) => s.id === scenarioId);
  }

  function pct(x) {
    return `${Math.round(x * 100)}%`;
  }

  function renderControls() {
    els.scenario.innerHTML = DATA.scenarios
      .map(
        (s) =>
          `<option value="${s.id}">${s.title} · ${s.label}</option>`
      )
      .join("");
    els.scenario.value = scenarioId;

    els.perturbBtns.innerHTML = order
      .map((key) => {
        const meta = DATA.perturb[key];
        return `<button type="button" data-key="${key}" class="${
          key === perturb ? "active" : ""
        }">${meta.label}</button>`;
      })
      .join("");

    els.perturbBtns.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        perturb = btn.dataset.key;
        renderControls();
        updateStage();
      });
    });
  }

  function updateStage() {
    const sc = scenario();
    const meta = DATA.perturb[perturb];
    const src = sc.images[perturb];

    els.scenarioNote.textContent = sc.note;
    els.carrier.textContent = `Carrier: “${sc.carrier}”`;

    els.stageImg.classList.add("is-fading");
    window.setTimeout(() => {
      els.stageImg.src = src;
      els.stageImg.alt = `${sc.title} — ${meta.label}`;
      els.stageImg.classList.remove("is-fading");
    }, 160);

    // For harmful scenarios show eval rates; benign: qualitative
    if (sc.label === "harmful") {
      els.probeVal.textContent = pct(meta.probe);
      els.ocrVal.textContent = pct(meta.ocr);
      els.probeBar.style.width = pct(meta.probe);
      els.ocrBar.style.width = pct(meta.ocr);
      els.blurb.textContent = meta.blurb;
    } else {
      els.probeVal.textContent = "low";
      els.ocrVal.textContent = "n/a";
      els.probeBar.style.width = "12%";
      els.ocrBar.style.width = "0%";
      els.blurb.textContent =
        "Benign control — probe should stay quiet; used to check over-refusal / retain.";
    }
  }

  function renderStats() {
    const blur = DATA.chart.find(
      (r) => r.perturbation === "blur" && r.severity === "4x"
    );
    const occ = DATA.chart.find(
      (r) => r.perturbation === "occlude" && r.severity === "30%"
    );
    const clean = DATA.chart.find((r) => r.perturbation === "clean");

    const items = [
      {
        v: pct(blur.probe_rate),
        l: `Probe under heavy blur (OCR ${pct(blur.ocr_rate)})`,
      },
      {
        v: pct(occ.probe_rate),
        l: `Probe under 30% occlusion (OCR ${pct(occ.ocr_rate)})`,
      },
      {
        v: pct(clean.probe_rate),
        l: "Probe on clean harmful images",
      },
    ];

    els.statRow.innerHTML = items
      .map((it) => `<li><strong>${it.v}</strong><span>${it.l}</span></li>`)
      .join("");
  }

  function renderCb() {
    const c = DATA.cb;
    const rows = [
      ["Harmful · clean · comply", c.harmful_clean_base_comply, c.harmful_clean_cb_comply, "meh"],
      ["Harmful · occlude · comply", c.harmful_occlude_base_comply, c.harmful_occlude_cb_comply, "bad"],
      ["Benign · clean · comply", c.benign_base_comply, c.benign_cb_comply, "meh"],
    ];
    els.cbBody.innerHTML = rows
      .map(
        ([name, b, cb, cls]) =>
          `<tr><td>${name}</td><td>${pct(b)}</td><td class="${cls}">${pct(cb)}</td></tr>`
      )
      .join("");
  }

  function renderMethod() {
    els.takeaways.innerHTML = DATA.takeaways
      .map((t) => `<li><div>${t}</div></li>`)
      .join("");
    els.cites.innerHTML = DATA.citations
      .map(
        (c) =>
          `<div class="cite"><a href="${c.url}" target="_blank" rel="noopener">${c.title}</a><p>${c.venue}</p></div>`
      )
      .join("");
  }

  els.scenario.addEventListener("change", () => {
    scenarioId = els.scenario.value;
    updateStage();
  });

  // Cycle hero image subtly
  const heroKeys = ["fakenews_clean.png", "fakenews_blur.png", "fakenews_occlude.png"];
  let hi = 0;
  window.setInterval(() => {
    hi = (hi + 1) % heroKeys.length;
    if (els.heroImg) {
      els.heroImg.style.opacity = "0.55";
      window.setTimeout(() => {
        els.heroImg.src = `assets/${heroKeys[hi]}`;
        els.heroImg.style.opacity = "1";
      }, 280);
    }
  }, 4200);
  if (els.heroImg) {
    els.heroImg.style.transition = "opacity 0.35s ease";
  }

  renderControls();
  updateStage();
  renderStats();
  renderCb();
  renderMethod();
})();
