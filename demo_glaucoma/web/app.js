import { binaryMetrics, compare, confusionMatrix, labels } from "./comparisons.mjs";

const $ = (id) => document.getElementById(id);
const modeInputs = [...document.querySelectorAll("input[name=mode]")];
const modelSelect = $("model");
const retina = $("retina");
const state = {
  mode: "binary",
  chosen: {},
  models: [],
  examples: [],
  rows: [],
  active: null,
  nextId: 1,
  running: false,
  importing: false,
  starter: false,
  progress: null,
};
const MAX_ROWS = 100;
const percent = new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 1 });

function node(tag, content = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = content;
  if (className) element.className = className;
  return element;
}

function badge(content, tone = "neutral") {
  return node("span", content, `badge ${tone}`);
}

function buttonStatus(result) {
  switch (result?.status) {
    case "ok": return ["Procesado", "good"];
    case "error": return ["Error", "bad"];
    case "running": return ["Procesando…", "working"];
    default: return ["Pendiente", "neutral"];
  }
}

const LODO_FOLD_MAP = {
  "RIM-ONE DL": "rim",
  "ACRIMA": "acrima",
  "EyePACS REAL": "eyepacs",
  "Moondream Staged · test": "moondream",
};

function autoRouterSpecs() {
  const readySet = new Set(state.models.filter((m) => m.status === "ready").map((m) => m.id));
  const specs = [];
  const allBase = Object.values(LODO_FOLD_MAP).every((slug) => readySet.has(`lodo-sin-${slug}-base`));
  const allAug = Object.values(LODO_FOLD_MAP).every((slug) => readySet.has(`lodo-sin-${slug}-aumento`));
  if (allBase && allAug) {
    specs.push({
      id: "lodo-auto-compare",
      mode: "binary",
      name: "LODO · Comparar Sin aumento vs. Con aumento (auto)",
      status: "ready",
      message: "Ejecuta automáticamente ambos modelos LODO (Sin aumento y Con aumento) del fold correspondiente a cada dataset.",
      isCompare: true,
    });
  }
  for (const [variant, label] of [["aumento", "Aumento de escala"], ["base", "Sin aumento"]]) {
    const foldIds = Object.values(LODO_FOLD_MAP).map((slug) => `lodo-sin-${slug}-${variant}`);
    if (foldIds.every((id) => readySet.has(id))) {
      specs.push({
        id: `lodo-auto-${variant}`,
        mode: "binary",
        name: `LODO · Solo ${label} (auto por dataset)`,
        status: "ready",
        message: "Selecciona automáticamente el modelo LODO que excluyó el dataset de cada imagen.",
        isAuto: true,
        variant,
      });
    }
  }
  return specs;
}

function allModelsForMode(mode) {
  const list = state.models.filter((model) => model.mode === mode);
  return mode === "binary" ? [...autoRouterSpecs(), ...list] : list;
}

function currentModel() {
  return allModelsForMode(state.mode).find((model) => model.id === state.chosen[state.mode]);
}

function resolveModelForRow(model, row, variantOverride = null) {
  const slug = LODO_FOLD_MAP[row?.source];
  if (variantOverride) {
    return `lodo-sin-${slug || "eyepacs"}-${variantOverride}`;
  }
  if (!model?.isAuto) return model?.id;
  return slug ? `lodo-sin-${slug}-${model.variant}` : "exp04-binario-estandar";
}

function isCompareMode() {
  return Boolean(currentModel()?.isCompare) && state.mode === "binary";
}

function currentResult(row) {
  return row?.results.get(state.chosen[state.mode]);
}

function showModels() {
  const choices = allModelsForMode(state.mode);
  const model = currentModel();
  modelSelect.replaceChildren(...choices.map((choice) => {
    const suffix = choice.status === "ready" ? "" : " · No disponible";
    return new Option(choice.name + suffix, choice.id);
  }));
  if (model) {
    modelSelect.value = model.id;
    modelSelect.title = model.id;
  } else {
    modelSelect.add(new Option("Sin modelo configurado", ""));
    modelSelect.title = "";
  }
  $("model-status").textContent = model?.message || "No hay modelos para este modo.";
  $("model-status").classList.toggle("error", model?.status !== "ready");
}

function comparisonText(comparison) {
  if (!comparison) return null;
  const outcome = comparison.match ? "Coincide" : "Discrepa";
  return badge(outcome, comparison.match ? "good" : "bad");
}

const manualOptions = [
  ["binary:normal", "Sano"], ["binary:glaucoma", "Glaucoma"],
  ["ternary:normal", "Sano"], ["ternary:early", "Temprano"], ["ternary:advanced", "Avanzado"],
];

function referenceSection(row, result) {
  const section = node("div", "", "reference");
  section.append(node("p", "Referencia", "field-label"));
  const ref = row?.reference;
  const line = node("div", "", "reference-line");
  const comparison = compare(result, ref, state.mode);
  if (ref) {
    line.append(node("strong", labels[ref.label]), node("span", ref.level === "binary" ? "Binaria" : "Trinaria", "dim"));
    if (comparison) line.append(comparisonText(comparison));
  } else {
    line.append(node("span", "Sin etiqueta de referencia", "dim"));
  }
  section.append(line);

  if (ref?.level === "binary" && state.mode === "ternary") {
    const projection = comparison?.projected;
    const details = projection
      ? `Comparación binaria derivada · ${labels[projection.label]} · P(Temprano) + P(Avanzado) = ${percent.format(projection.score)} · umbral ${percent.format(projection.threshold)}.`
      : "Severidad de referencia no disponible.";
    section.append(node("p", details, "comparison-note"));
    if (projection) section.append(node("p", "Severidad de referencia no disponible.", "comparison-note"));
  } else if (comparison?.level === "grouped") {
    section.append(node("p", `Referencia agrupada a binario · ${labels[comparison.expected]}.`, "comparison-note"));
  }

  if (row?.file) {
    const select = node("select");
    select.id = "manual-label";
    const label = node("label", "Editar referencia", "field-label");
    label.htmlFor = select.id;
    label.classList.add("edit-label");
    select.add(new Option("Sin etiqueta de referencia", ""));
    for (const [level, name] of [["binary", "Binaria"], ["ternary", "Trinaria"]]) {
      const group = document.createElement("optgroup");
      group.label = name;
      manualOptions.filter(([value]) => value.startsWith(level + ":"))
        .forEach(([value, text]) => group.append(new Option(text, value)));
      select.append(group);
    }
    select.value = ref ? `${ref.level}:${ref.label}` : "";
    select.disabled = state.running || state.importing;
    select.addEventListener("change", () => {
      const [level, label] = select.value.split(":");
      row.reference = level ? { level, label } : null;
      render();
      $("manual-label").focus();
    });
    section.append(label, select);
  }
  return section;
}

function renderArmCard(title, res, ref, extraClass = "") {
  const card = node("div", "", `arm-card ${extraClass}`.trim());
  const head = node("div", "", "arm-head");
  head.append(node("span", title, "arm-title"));
  const comp = compare(res, ref, state.mode);
  if (comp) head.append(comparisonText(comp));
  const predRow = node("div", "", "arm-pred");
  predRow.append(node("strong", labels[res.prediction]));
  const tau = res.decision_summary?.threshold;
  if (tau != null) predRow.append(node("span", `τ = ${percent.format(tau)}`, "dim"));
  const scores = node("div", "", "scores");
  for (const item of res.scores) {
    const line = node("div", "", "score");
    const meter = node("meter");
    meter.min = 0;
    meter.max = 1;
    meter.value = item.score;
    meter.setAttribute("aria-label", `${title} ${labels[item.label]}`);
    line.append(node("span", labels[item.label]), node("span", percent.format(item.score)), meter);
    scores.append(line);
  }
  card.append(head, predRow, scores);
  return card;
}

function foldCaption(modelId) {
  const m = state.models.find((item) => item.id === modelId);
  if (!m) return modelId || "";
  return m.name.replace("ConvNeXt-Tiny · ", "");
}

function showActiveModelBanner() {
  const banner = $("active-model-banner");
  if (!banner) return;
  const model = currentModel();
  if (!model) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const uniqueSources = [...new Set(state.rows.map((r) => (r.file ? "Archivo local" : r.source)))];
  let detail = "";
  if (model.isCompare || model.isAuto) {
    if (uniqueSources.length === 1) {
      const only = uniqueSources[0];
      const baseId = resolveModelForRow(model, { source: only, file: only === "Archivo local" ? true : null }, model.isCompare ? "base" : null);
      const cleanFold = foldCaption(baseId).replace(" (Sin aumento)", "").replace(" (Aumento escala)", "");
      detail = `${cleanFold} · ${model.isCompare ? "compara Sin aumento vs. Con aumento de escala" : model.name}`;
    } else {
      detail = `ConvNeXt-Tiny · LODO automático por dataset (${model.isCompare ? "Sin aumento vs. Con aumento en cada cohorte ciega" : model.name})`;
    }
  } else {
    const dec = model.decision || {};
    const tau = dec.threshold ?? dec.binary_threshold;
    detail = `${model.name}${tau != null ? ` · umbral τ = ${percent.format(tau)}` : ""}`;
  }
  banner.replaceChildren(
    badge("Modelo clasificando", "working"),
    node("strong", detail),
  );
}

function showCase() {
  const index = state.rows.findIndex((row) => row.id === state.active);
  const row = state.rows[index];
  const result = currentResult(row);
  $("case-count").textContent = row ? `Imagen ${index + 1} de ${state.rows.length}` : "Sin imagen";
  $("image-name").textContent = row?.name || "Sin archivo seleccionado";
  $("image-source").textContent = row?.source || "";
  const empty = $("image-empty");
  if (!row) {
    retina.removeAttribute("src");
    retina.hidden = true;
    empty.hidden = false;
    empty.lastElementChild.textContent = "Selecciona una retinografía";
  } else if (retina.getAttribute("src") !== row.url) {
    retina.alt = `Retinografía ${row.name}`;
    retina.hidden = true;
    empty.hidden = false;
    empty.lastElementChild.textContent = "Cargando vista previa…";
    retina.src = row.url;
  }

  const [status, tone] = buttonStatus(result);
  $("result-status").textContent = status;
  $("result-status").className = `badge ${tone}`;
  const output = $("results-body");
  output.replaceChildren();
  const activeModel = currentModel();
  if (result?.status === "ok" && result.mode === "compare") {
    const cleanFold = foldCaption(result.aug.model_id).replace(" (Aumento escala)", "");
    output.append(node("p", `Clasificado con: ConvNeXt-Tiny · ${cleanFold}`, "active-model-pill"));
    const wrap = node("div", "", "compare-arms");
    wrap.append(
      renderArmCard("Sin aumento (Base)", result.base, row.reference, "arm-base"),
      renderArmCard("Con aumento de escala", result.aug, row.reference, "arm-aug"),
    );
    output.append(wrap);
    if (result.base.prediction !== result.aug.prediction) {
      output.append(node("p", `El aumento cambió la predicción: ${labels[result.base.prediction]} → ${labels[result.aug.prediction]}`, "flip-note"));
    }
  } else if (result?.status === "ok") {
    const usedId = result.routed_model_id || result.model_id;
    output.append(node("p", `Clasificado con: ${foldCaption(usedId)}`, "active-model-pill"));
    const prediction = node("div", "", "prediction");
    prediction.append(node("p", "Clase predicha", "field-label"), node("strong", labels[result.prediction]));
    const scores = node("div", "", "scores");
    for (const item of result.scores) {
      const line = node("div", "", "score");
      const meter = node("meter");
      meter.min = 0;
      meter.max = 1;
      meter.value = item.score;
      meter.setAttribute("aria-label", labels[item.label]);
      line.append(node("span", labels[item.label]), node("span", percent.format(item.score)), meter);
      scores.append(line);
    }
    output.append(prediction, scores);
  } else {
    if (activeModel && row) {
      const targetId = resolveModelForRow(activeModel, row, activeModel.isCompare ? "base" : null);
      const previewLabel = activeModel.isCompare
        ? foldCaption(targetId).replace(" (Sin aumento)", " (Base vs. Con aumento)")
        : activeModel.isAuto ? foldCaption(targetId) : activeModel.name;
      output.append(node("p", `Modelo asignado: ${previewLabel}`, "active-model-pill"));
    }
    const message = result?.message || (result?.status === "running" ? "Procesando esta imagen…" :
      !row ? "Agrega una imagen para comenzar." :
      activeModel?.status === "ready" ? "Ejecuta la predicción para ver las puntuaciones." : "Modelo no disponible.");
    output.append(node("p", message, result?.status === "error" ? "waiting row-error" : "waiting"));
  }
  output.append(referenceSection(row, result?.mode === "compare" ? result.aug : result));
}

function openCase(id) {
  state.active = id;
  showCase();
  for (const row of $("rows").rows) {
    const selected = row.dataset.id === id;
    row.classList.toggle("active", selected);
    row.querySelector("button")?.setAttribute("aria-pressed", String(selected));
  }
  const viewer = $("viewer-heading");
  const top = viewer.getBoundingClientRect().top;
  if (top < 0 || top >= window.innerHeight) viewer.scrollIntoView({ block: "start" });
}

function tableRow(row, compareMode) {
  const tr = node("tr");
  tr.dataset.id = row.id;
  tr.classList.toggle("active", row.id === state.active);
  const result = currentResult(row);
  const ref = row.reference;
  const imageCell = node("td");
  const button = node("button", "", "row-image");
  button.type = "button";
  button.setAttribute("aria-pressed", String(row.id === state.active));
  button.setAttribute("aria-label", `Inspeccionar ${row.name} · ${row.id}`);
  const thumb = node("img", "", "thumb");
  thumb.alt = "";
  thumb.loading = "lazy";
  thumb.addEventListener("error", () => thumb.replaceWith(node("span", "", "thumb")), { once: true });
  thumb.src = row.url;
  const filename = node("span");
  filename.append(node("span", row.name, "row-name"), node("span", row.file ? row.id : row.source, "row-id"));
  button.append(thumb, filename);
  button.addEventListener("click", () => openCase(row.id));
  imageCell.append(button);

  const refCell = node("td");
  refCell.append(node("span", ref ? labels[ref.label] : "Sin etiqueta de referencia", ref ? "" : "dim"));
  if (ref) {
    const refLevel = ref.level === "ternary" ? "Trinaria" :
      state.mode === "ternary" ? "Binaria · severidad no disponible" : "Binaria";
    refCell.append(node("span", refLevel, "row-id"));
  }

  const statusCell = node("td");
  statusCell.append(badge(...buttonStatus(result)));
  const activeModel = currentModel();
  if (result?.status === "ok" && (activeModel?.isCompare || activeModel?.isAuto)) {
    const usedId = result.mode === "compare" ? result.aug.model_id : (result.routed_model_id || result.model_id);
    const foldTag = foldCaption(usedId).replace(" (Aumento escala)", "").replace(" (Sin aumento)", "");
    statusCell.append(node("span", foldTag, "row-id"));
  }
  if (result?.message) statusCell.append(node("span", result.message, "row-error"));

  if (compareMode) {
    const baseRes = result?.status === "ok" ? result.base : null;
    const augRes = result?.status === "ok" ? result.aug : null;
    const baseCell = node("td");
    if (baseRes) {
      const box = node("div", "", "cell-stack");
      box.append(node("span", labels[baseRes.prediction]));
      const cBase = compare(baseRes, ref, state.mode);
      if (cBase) box.append(comparisonText(cBase));
      baseCell.append(box);
    } else {
      baseCell.textContent = "—";
    }
    const augCell = node("td");
    if (augRes) {
      const box = node("div", "", "cell-stack");
      box.append(node("span", labels[augRes.prediction]));
      const cAug = compare(augRes, ref, state.mode);
      if (cAug) box.append(comparisonText(cAug));
      augCell.append(box);
    } else {
      augCell.textContent = "—";
    }
    tr.append(imageCell, baseCell, augCell, refCell, statusCell);
  } else {
    const comparison = compare(result, ref, state.mode);
    const predCell = node("td", result?.prediction ? labels[result.prediction] : "—");
    const compCell = node("td", comparison ? "" : "—");
    if (comparison) {
      compCell.append(comparisonText(comparison));
      if (comparison.level === "derived") compCell.append(node("span", "Binaria derivada", "row-id"));
      if (comparison.level === "grouped") compCell.append(node("span", "Referencia agrupada", "row-id"));
    }
    tr.append(imageCell, predCell, refCell, compCell, statusCell);
  }
  return tr;
}

function showTable() {
  const compareMode = isCompareMode();
  const headers = compareMode
    ? ["Imagen", "Sin aumento (Base)", "Con aumento", "Referencia", "Estado"]
    : ["Imagen", "Predicción", "Referencia", "Comparación", "Estado"];
  const headerRow = $("batch-header-row");
  if (headerRow) {
    headerRow.replaceChildren(...headers.map((text) => {
      const th = node("th", text);
      th.scope = "col";
      return th;
    }));
  }
  const focused = document.activeElement?.closest("tr")?.dataset.id;
  const rows = $("rows");
  if (state.rows.length) {
    rows.replaceChildren(...state.rows.map((row) => tableRow(row, compareMode)));
    if (focused) [...rows.rows].find((row) => row.dataset.id === focused)?.querySelector("button").focus({ preventScroll: true });
  } else {
    const row = node("tr");
    const cell = node("td", "El lote está vacío.", "table-empty");
    cell.colSpan = 5;
    row.append(cell);
    rows.replaceChildren(row);
  }
}

function showSummary() {
  const total = state.rows.length;
  const compareMode = isCompareMode();
  let done = 0;
  let failures = 0;
  let comparable = 0;
  let matches = 0;
  let baseMatches = 0;
  let augMatches = 0;
  let derived = 0;
  let derivedMatches = 0;
  for (const row of state.rows) {
    const result = currentResult(row);
    if (result?.status === "error") failures++;
    if (result?.status === "ok") done++;
    if (compareMode && result?.status === "ok" && result.mode === "compare") {
      const cBase = compare(result.base, row.reference, state.mode);
      const cAug = compare(result.aug, row.reference, state.mode);
      if (cBase && cAug) {
        comparable++;
        if (cBase.match) baseMatches++;
        if (cAug.match) augMatches++;
      }
      continue;
    }
    const comparison = compare(result, row.reference, state.mode);
    if (!comparison) continue;
    if (comparison.level === "derived") {
      derived++;
      if (comparison.match) derivedMatches++;
    } else {
      comparable++;
      if (comparison.match) matches++;
    }
  }
  $("batch-count").textContent = `${total} ${total === 1 ? "archivo" : "archivos"}`;
  const parts = [`${done + failures}/${total} revisados`, `${done} con predicción`, `${failures} con error`];
  if (compareMode && comparable) {
    parts.push(`Sin aumento: ${baseMatches}/${comparable}`, `Con aumento: ${augMatches}/${comparable}`);
  } else {
    if (comparable) parts.push(`${matches}/${comparable} coincidencias`);
    if (derived) parts.push(`${derivedMatches}/${derived} coincidencias binarias derivadas`);
    if (done && done - comparable - derived) parts.push(`${done - comparable - derived} sin comparación`);
  }
  $("batch-summary").textContent = total ? parts.join(" · ") : "Agrega una imagen o carga el lote de ejemplo.";
  const bar = $("progress");
  bar.hidden = !state.running || !state.progress;
  if (state.progress) {
    bar.max = state.progress.total;
    bar.value = state.progress.done;
  }
}

const dec3 = new Intl.NumberFormat("es-CO", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const pct1 = new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 1 });
const fmtDec = (v) => (v == null || Number.isNaN(v) ? "—" : dec3.format(v));
const fmtPct = (v) => (v == null || Number.isNaN(v) ? "—" : pct1.format(v));

function buildMetricsPanel(title, badgeLabel, groups, baselineBySource = null) {
  const panel = node("div", "", "metrics-panel");
  const heading = node("h3", "", "metrics-panel-title");
  heading.append(node("span", title), badge(badgeLabel, "working"));
  panel.append(heading);

  const wrap = node("div", "", "table-wrap");
  const table = node("table", "", "metrics-table");
  const thead = table.createTHead().insertRow();
  for (const col of ["Dataset", "n (+/−)", "Exact.", "Prec.", "Sens.", "Espec.", "TFP", "TFN", "F1", "AUROC"]) {
    const th = node("th", col);
    th.scope = "col";
    thead.append(th);
  }
  const tbody = table.createTBody();
  for (const group of groups) {
    const m = group.metrics;
    const base = baselineBySource?.get(group.name);
    const tr = tbody.insertRow();
    if (group.isGlobal) tr.className = "global-row";
    tr.append(
      node("td", group.name),
      node("td", m.n ? `${m.n} (${m.pos}/${m.neg})` : "0"),
      node("td", fmtPct(m.accuracy)),
      node("td", fmtPct(m.precision)),
      node("td", fmtPct(m.sensitivity)),
      node("td", fmtPct(m.specificity)),
      node("td", fmtPct(m.fpr)),
      node("td", fmtPct(m.fnr)),
    );
    const f1Cell = node("td");
    f1Cell.append(document.createTextNode(fmtDec(m.f1Macro)));
    if (base && m.f1Macro != null && base.f1Macro != null) {
      const df1 = 100 * (m.f1Macro - base.f1Macro);
      if (Math.abs(df1) >= 0.05) {
        f1Cell.append(node("span", `${df1 > 0 ? "+" : ""}${df1.toFixed(1)}`, `delta-chip ${df1 > 0 ? "delta-up" : "delta-down"}`));
      }
    }
    const aucCell = node("td");
    aucCell.append(document.createTextNode(fmtDec(m.auroc)));
    if (base && m.auroc != null && base.auroc != null) {
      const dauc = 100 * (m.auroc - base.auroc);
      if (Math.abs(dauc) >= 0.05) {
        aucCell.append(node("span", `${dauc > 0 ? "+" : ""}${dauc.toFixed(1)}`, `delta-chip ${dauc > 0 ? "delta-up" : "delta-down"}`));
      }
    }
    tr.append(f1Cell, aucCell);
  }
  wrap.append(table);
  panel.append(wrap);
  return panel;
}

function showDatasetMetrics() {
  const container = $("dataset-metrics");
  const subtitle = $("metrics-subtitle");
  if (!container) return;
  const compareMode = isCompareMode();
  const sources = [];
  const seen = new Set();
  for (const row of state.rows) {
    const key = row.file ? "Archivo local" : row.source;
    if (!seen.has(key)) {
      seen.add(key);
      sources.push(key);
    }
  }

  const collect = (picker) => {
    const groups = sources.map((source) => {
      const cases = state.rows
        .filter((row) => (row.file ? "Archivo local" : row.source) === source)
        .map((row) => ({ result: picker(row), reference: row.reference }));
      return { name: source, isGlobal: false, metrics: binaryMetrics(cases) };
    });
    const allCases = state.rows.map((row) => ({ result: picker(row), reference: row.reference }));
    groups.push({ name: "Global (lote)", isGlobal: true, metrics: binaryMetrics(allCases) });
    return groups;
  };

  if (compareMode) {
    container.className = "metrics-panels";
    subtitle.textContent = "Tablas pareadas por dataset en el lote actual: Sin aumento (Base) frente a Con aumento de escala (δ en puntos porcentuales junto a F1 y AUROC).";
    const baseGroups = collect((row) => {
      const r = currentResult(row);
      return r?.status === "ok" && r.mode === "compare" ? r.base : null;
    });
    const augGroups = collect((row) => {
      const r = currentResult(row);
      return r?.status === "ok" && r.mode === "compare" ? r.aug : null;
    });
    const baseMap = new Map(baseGroups.map((g) => [g.name, g.metrics]));
    container.replaceChildren(
      buildMetricsPanel("Sin aumento (Base)", "Control LODO", baseGroups),
      buildMetricsPanel("Con aumento (Aumento de escala)", "Δ vs. Base (pp)", augGroups, baseMap),
    );
  } else {
    container.className = "metrics-panels single";
    subtitle.textContent = state.mode === "ternary"
      ? "Desglose binario derivado por dataset (proyección P(Temprano)+P(Avanzado) frente a referencia binaria o agrupada)."
      : "Desglose en vivo por dataset del modelo seleccionado (Exactitud, Precisión, Sensibilidad, Especificidad, TFP, TFN, F1 macro y AUROC).";
    const singleGroups = collect((row) => currentResult(row));
    const modelName = currentModel()?.name || "Modelo activo";
    container.replaceChildren(buildMetricsPanel(modelName, state.mode === "ternary" ? "Proyección binaria" : "Lote actual", singleGroups));
  }
}

function buildMatrixNode(titleText, cases) {
  const { classes, counts, total, binaryHealthy, withoutStage } = confusionMatrix(cases, state.mode);
  const wrapper = node("div", "", "matrix");
  const table = node("table");
  table.append(node("caption", titleText + " · " + total + (total === 1 ? " caso" : " casos")));
  const header = table.createTHead().insertRow();
  for (const label of ["Referencia", ...classes.map((key) => labels[key])]) {
    const cell = node("th", label);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  classes.forEach((label, rowIndex) => {
    const row = body.insertRow();
    const labelCell = node("th", labels[label]);
    labelCell.scope = "row";
    row.append(labelCell);
    counts[rowIndex].forEach((count, columnIndex) => {
      const tone = count ? (rowIndex === columnIndex ? "hit" : "miss") : "";
      row.append(node("td", total ? String(count) : "—", tone));
    });
  });
  wrapper.append(table);
  if (!total) {
    const description = classes.length === 3
      ? "La matriz requiere predicciones con una referencia de clase verificable."
      : "La matriz requiere predicciones con referencias comparables.";
    wrapper.append(node("p", description, "fine"));
  }
  return { wrapper, binaryHealthy, withoutStage };
}

function showConfusion() {
  const container = $("confusion-matrices");
  if (isCompareMode()) {
    const baseCases = state.rows.map((row) => {
      const r = currentResult(row);
      return { result: r?.status === "ok" && r.mode === "compare" ? r.base : null, reference: row.reference };
    });
    const augCases = state.rows.map((row) => {
      const r = currentResult(row);
      return { result: r?.status === "ok" && r.mode === "compare" ? r.aug : null, reference: row.reference };
    });
    const mBase = buildMatrixNode("Sin aumento (Base)", baseCases);
    const mAug = buildMatrixNode("Con aumento (Aumento de escala)", augCases);
    container.replaceChildren(mBase.wrapper, mAug.wrapper);
    return;
  }
  const cases = state.rows.map((row) => ({ result: currentResult(row), reference: row.reference }));
  const title = state.mode === "binary" ? "Clasificación binaria" : "Clasificación trinaria";
  const { wrapper, binaryHealthy, withoutStage } = buildMatrixNode(title, cases);
  container.replaceChildren(wrapper);
  if (state.mode === "ternary") {
    const notice = node("aside", "", "matrix-notice");
    notice.setAttribute("aria-label", "Alcance de las referencias");
    notice.append(
      node("h3", "Qué permite verificar la referencia"),
      node("p", "Una etiqueta binaria Sano corresponde a la fila Sano. Una etiqueta Glaucoma solo confirma presencia y deja el estadio sin verificar. Esas imágenes quedan fuera de la matriz y sirven para la comparación binaria derivada."),
    );
    if (binaryHealthy || withoutStage) {
      notice.append(node("p", "En este lote: " + binaryHealthy + " Sano con referencia binaria incluidos · "
        + withoutStage + " Glaucoma sin estadio excluidos.", "matrix-tally"));
    }
    container.append(notice);
  }
}

function showControls() {
  const busy = state.running || state.importing;
  const canRun = currentModel()?.status === "ready" && state.rows.length > 0;
  for (const input of modeInputs) input.disabled = busy;
  modelSelect.disabled = busy || !currentModel();
  $("dropzone").disabled = busy;
  $("example-dataset").disabled = busy || state.examples.length === 0;
  $("load-examples").disabled = busy || state.examples.length === 0;
  $("run").disabled = busy || !canRun;
  $("run").textContent = state.running ? "Procesando…" : "Ejecutar predicción";
  $("clear").disabled = busy || state.rows.length === 0;
  $("import-csv").disabled = busy || !state.rows.some((row) => row.file);
}

function render() {
  showCase();
  showActiveModelBanner();
  showTable();
  showSummary();
  showDatasetMetrics();
  showConfusion();
  showControls();
}

function showFeedback(message = "", issues = [], ok = false) {
  const container = $("csv-feedback");
  container.hidden = !message && !issues.length;
  container.classList.toggle("ok", ok && issues.length === 0);
  container.replaceChildren(node("p", message));
  if (issues.length) {
    const list = node("ul");
    for (const issue of issues) list.append(node("li", issue));
    container.append(list);
  }
}

function resetBatch() {
  for (const row of state.rows) if (row.file) URL.revokeObjectURL(row.url);
  state.rows = [];
  state.active = null;
  state.nextId = 1;
  state.starter = false;
  state.progress = null;
  showFeedback();
}

function loadExamples(entries, starter = false) {
  resetBatch();
  state.rows = entries.map((entry) => ({
    ...entry,
    url: `/api/examples/${encodeURIComponent(entry.id)}/image`,
    results: new Map(),
  }));
  state.active = state.rows[0]?.id;
  state.starter = starter;
  render();
}

function upload(files) {
  if (!files.length || state.running || state.importing) return;
  if (state.starter) resetBatch();
  const available = MAX_ROWS - state.rows.length;
  const added = Array.from(files).slice(0, available).map((file) => ({
    id: `U${String(state.nextId++).padStart(3, "0")}`,
    file,
    name: file.name,
    source: "Archivo local",
    url: URL.createObjectURL(file),
    reference: null,
    results: new Map(),
  }));
  state.rows.push(...added);
  state.active ??= added[0]?.id;
  state.progress = null;
  if (files.length > available) showFeedback(`El lote admite ${MAX_ROWS} archivos. Se omitieron ${files.length - available}.`);
  render();
}

async function runBatch() {
  const model = currentModel();
  if (!model || model.status !== "ready" || state.running || state.importing || !state.rows.length) return;
  state.running = true;
  state.starter = false;
  state.progress = { done: 0, total: state.rows.length };
  for (const row of state.rows) row.results.delete(model.id);
  for (const row of state.rows) {
    row.results.set(model.id, { status: "running" });
    render();
    try {
      if (model.isCompare) {
        const baseId = resolveModelForRow(model, row, "base");
        const augId = resolveModelForRow(model, row, "aumento");
        const [rBase, rAug] = await Promise.all([baseId, augId].map(async (mid) => {
          const q = new URLSearchParams({ model_id: mid });
          if (!row.file) q.set("example_id", row.id);
          const resp = await fetch(`/api/predict?${q}`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: row.file,
          });
          return resp.json();
        }));
        if (rBase.status === "ok" && rAug.status === "ok") {
          row.results.set(model.id, { status: "ok", mode: "compare", base: rBase, aug: rAug });
        } else {
          row.results.set(model.id, {
            status: "error",
            model_id: model.id,
            message: rBase.message || rAug.message || "Error al evaluar el par LODO.",
          });
        }
      } else {
        const targetModelId = resolveModelForRow(model, row);
        const query = new URLSearchParams({ model_id: targetModelId });
        if (!row.file) query.set("example_id", row.id);
        const response = await fetch(`/api/predict?${query}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: row.file,
        });
        const result = await response.json();
        if (targetModelId !== model.id) result.routed_model_id = targetModelId;
        row.results.set(model.id, result);
      }
    } catch {
      row.results.set(model.id, {
        status: "error", model_id: model.id, message: "No se pudo comunicar con el motor local.",
      });
    }
    state.progress.done++;
    render();
  }
  state.running = false;
  render();
}

async function attachCsv(file) {
  if (!file) return;
  state.importing = true;
  render();
  try {
    if (file.size > 512_000) throw new Error("El CSV supera los 512 kB.");
    const files = state.rows.filter((row) => row.file).map(({ id, name }) => ({ id, name }));
    const csv = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    const response = await fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv, files }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message);
    for (const [id, ref] of Object.entries(result.references)) {
      state.rows.find((row) => row.id === id).reference = ref;
    }
    const count = Object.keys(result.references).length;
    showFeedback(`${count} ${count === 1 ? "etiqueta aplicada" : "etiquetas aplicadas"}.`, result.issues, true);
  } catch (error) {
    showFeedback(error instanceof TypeError ? "Lee el CSV como UTF-8 y verifica que el motor local esté activo." : error.message);
  } finally {
    state.importing = false;
    render();
  }
}

retina.addEventListener("load", () => {
  retina.hidden = false;
  $("image-empty").hidden = true;
});
retina.addEventListener("error", () => {
  retina.hidden = true;
  $("image-empty").hidden = false;
  $("image-empty").lastElementChild.textContent = "Vista previa no disponible";
});
for (const input of modeInputs) {
  input.addEventListener("change", () => {
    state.mode = input.value;
    state.progress = null;
    showModels();
    render();
  });
}
modelSelect.addEventListener("change", () => {
  state.chosen[state.mode] = modelSelect.value;
  state.progress = null;
  showModels();
  render();
});
function selectedExamples() {
  const chosen = $("example-dataset").value;
  return chosen === "all" ? state.examples : state.examples.filter((entry) => entry.source === chosen);
}

function populateExampleDatasets() {
  const select = $("example-dataset");
  const counts = new Map();
  for (const entry of state.examples) {
    counts.set(entry.source, (counts.get(entry.source) || 0) + 1);
  }
  const options = [new Option(`Todos los datasets (${state.examples.length} · 6 por fuente)`, "all")];
  for (const [source, count] of counts) {
    options.push(new Option(`${source} (${count})`, source));
  }
  select.replaceChildren(...options);
}

$("example-dataset").addEventListener("change", () => loadExamples(selectedExamples()));
$("load-examples").addEventListener("click", () => loadExamples(selectedExamples()));
$("dropzone").addEventListener("click", () => $("images").click());
$("images").addEventListener("change", (event) => {
  upload(event.target.files);
  event.target.value = "";
});
for (const type of ["dragover", "drop"]) {
  window.addEventListener(type, (event) => {
    if (event.dataTransfer.types.includes("Files")) event.preventDefault();
  });
}
$("dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("dropzone").classList.add("dragging");
});
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
$("dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("dragging");
  upload(event.dataTransfer.files);
});
$("run").addEventListener("click", runBatch);
$("clear").addEventListener("click", () => {
  resetBatch();
  render();
  $("dropzone").focus();
});
$("import-csv").addEventListener("click", () => $("csv-file").click());
$("csv-file").addEventListener("change", (event) => {
  attachCsv(event.target.files[0]);
  event.target.value = "";
});

async function init() {
  render();
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error("Catálogo no disponible");
    const catalog = await response.json();
    state.models = catalog.models;
    state.examples = catalog.examples;
    const issues = [...catalog.example_issues];
    for (const mode of ["binary", "ternary"]) {
      const choices = allModelsForMode(mode);
      const autoPreferred = mode === "binary" ? choices.find((m) => m.id === "lodo-auto-compare") : null;
      const preferred = autoPreferred || choices.find((model) => model.id === catalog.defaults[mode]);
      state.chosen[mode] = (preferred?.status === "ready" ? preferred :
        choices.find((model) => model.status === "ready") || preferred || choices[0])?.id;
      if (preferred?.status !== "ready" && preferred) {
        issues.push(`${mode === "binary" ? "Binario" : "Trinario"} · ${preferred.message}`);
      }
    }
    if (issues.length) {
      $("catalog-alert").textContent = issues.join(" ");
      $("catalog-alert").hidden = false;
    }
    showModels();
    populateExampleDatasets();
    if (state.examples.length && !state.rows.length) loadExamples(selectedExamples(), true);
  } catch {
    $("model-status").textContent = "No se pudo consultar el catálogo de modelos.";
    modelSelect.replaceChildren(new Option("Motor no disponible", ""));
  }
  render();
}

init();
