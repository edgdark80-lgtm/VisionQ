// VisionQ frontend: se comunica con el backend solo a través de la API REST /api/v1.
(() => {
  const API = (window.VISIONQ_API_BASE || "").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const DECISION_ES = { RECHAZO: "Rechazo", RETRABAJO: "Retrabajo", REVISION_MANUAL: "Revisión manual" };
  const SAMPLES = ["crazing_151.jpg", "inclusion_161.jpg", "patches_128.jpg",
    "pitted_surface_110.jpg", "rolled-in_scale_138.jpg", "scratches_195.jpg"];
  const pct = (x) => `${(x * 100).toFixed(1)} %`;

  async function api(path, options = {}) {
    const res = await fetch(`${API}/api/v1${path}`, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => `${d.campo}: ${d.mensaje}`).join(" · ")
        : body.detail || "";
      throw new Error(`${body.error || res.statusText}${detail ? " — " + detail : ""}`);
    }
    return body;
  }

  // ---------- Pestañas ----------
  const tabs = ["inspect", "failure", "system"];
  tabs.forEach((t) => $(`tab-${t}`).addEventListener("click", () => {
    tabs.forEach((o) => {
      $(`tab-${o}`).setAttribute("aria-selected", String(o === t));
      $(`panel-${o}`).hidden = o !== t;
    });
    if (t === "system") loadSystem();
  }));

  // ---------- Estado ----------
  async function checkHealth() {
    try {
      const h = await api("/health");
      $("status-dot").className = `dot ${h.status === "ok" ? "ok" : "warn"}`;
      $("status-text").textContent = h.status === "ok"
        ? `API en línea · v${h.version} · ${h.environment}`
        : "API en modo degradado: falta un modelo";
    } catch {
      $("status-dot").className = "dot down";
      $("status-text").textContent = "Sin conexión con la API";
    }
  }
  checkHealth();
  setInterval(checkHealth, 15000);

  // ---------- Inspección visual ----------
  const tag = $("tag");
  function setTag(el, cls, title, detail, decisionEl, detailEl) {
    el.className = `tag ${cls}`;
    void el.offsetWidth; el.classList.add("stamp");
    $(decisionEl).textContent = title;
    $(detailEl).textContent = detail;
  }

  async function inspect(blob, name) {
    $("preview").src = URL.createObjectURL(blob);
    $("preview").hidden = false;
    $("viewport-hint").hidden = true;
    $("dropzone").classList.add("busy");
    const form = new FormData();
    form.append("file", blob, name);
    try {
      const r = await api("/inspect", { method: "POST", body: form });
      setTag(tag, `tag-${r.decision}`, DECISION_ES[r.decision], r.defect_name_es, "tag-decision", "tag-defect");
      $("r-conf").textContent = pct(r.confidence);
      $("r-lat").textContent = `${r.latency_ms.toFixed(0)} ms`;
      $("r-ood").textContent = r.out_of_distribution ? "Fuera de rango" : "Dentro de rango";
      $("r-quality").textContent = r.quality.flags.length ? "Con alertas" : "Correcta";
      $("topk-list").innerHTML = r.top_k.map((k) =>
        `<li><span>${k.label}</span><span class="bar"><span style="width:${k.probability * 100}%"></span></span><span class="pct">${pct(k.probability)}</span></li>`).join("");
      $("reason").textContent = r.decision_reason;
      ["readout", "topk", "reason"].forEach((id) => ($(id).hidden = false));
    } catch (e) {
      setTag(tag, "tag-empty", "Sin resultado", e.message, "tag-decision", "tag-defect");
      ["readout", "topk", "reason"].forEach((id) => ($(id).hidden = true));
    } finally {
      $("dropzone").classList.remove("busy");
    }
  }

  $("file-input").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) inspect(f, f.name);
  });
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) inspect(f, f.name);
  });

  const row = $("sample-row");
  SAMPLES.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `Inspeccionar muestra ${s}`);
    b.innerHTML = `<img src="samples/${s}" alt="">`;
    b.addEventListener("click", async () => {
      row.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      const blob = await (await fetch(`samples/${s}`)).blob();
      inspect(blob, s);
    });
    row.appendChild(b);
  });

  // ---------- Predicción de fallas ----------
  const form = $("failure-form");
  $("preset-risk").addEventListener("click", () => {
    Object.entries({ type: "L", air_temperature_k: 302.0, process_temperature_k: 310.5,
      rotational_speed_rpm: 1300, torque_nm: 60, tool_wear_min: 210 })
      .forEach(([k, v]) => (form.elements[k].value = v));
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("form-error").hidden = true;
    const fd = new FormData(form);
    const payload = Object.fromEntries(fd.entries());
    for (const k of Object.keys(payload)) if (k !== "type") payload[k] = Number(payload[k]);
    try {
      const r = await api("/failure/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const title = r.prediction === "FALLA_PROBABLE" ? "Falla probable" : "Operación normal";
      setTag($("ftag"), `tag-${r.risk_level}`, title, `Riesgo ${r.risk_level.toLowerCase()}`, "ftag-decision", "ftag-detail");
      $("gauge-fill").style.width = `${r.failure_probability * 100}%`;
      $("gauge-thr").style.left = `${r.threshold * 100}%`;
      $("gauge-p").textContent = pct(r.failure_probability);
      $("gauge-t").textContent = pct(r.threshold);
      $("indicator-list").innerHTML = r.risk_indicators.length
        ? r.risk_indicators.map((i) => `<li><strong>${i.mode} · ${i.name}.</strong> ${i.detail}</li>`).join("")
        : "<li>Ninguna regla física de falla se activa con esta lectura.</li>";
      if (r.out_of_training_range.length) {
        $("indicator-list").innerHTML += `<li><strong>Extrapolación.</strong> Fuera del rango de entrenamiento: ${r.out_of_training_range.join(", ")}.</li>`;
      }
      $("faction").textContent = r.recommended_action;
      ["gauge", "indicators", "faction"].forEach((id) => ($(id).hidden = false));
    } catch (err) {
      $("form-error").textContent = err.message;
      $("form-error").hidden = false;
    }
  });

  // ---------- Sistema ----------
  async function loadSystem() {
    try {
      $("models-json").textContent = JSON.stringify(await api("/models"), null, 2);
      $("metrics-json").textContent = JSON.stringify(await api("/metrics"), null, 2);
    } catch (e) {
      $("models-json").textContent = e.message;
    }
  }
  $("refresh-metrics").addEventListener("click", loadSystem);
  ["link-docs", "link-redoc", "link-openapi"].forEach((id) => {
    const a = $(id); a.href = API + a.getAttribute("href");
  });
})();
