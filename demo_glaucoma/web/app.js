import { compare, confusionMatrix, labels } from "./comparisons.mjs";

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

function currentModel() {
  return state.models.find((model) => model.id === state.chosen[state.mode]);
}

function currentResult(row) {
  return row?.results.get(state.chosen[state.mode]);
}

function showModels() {
  const choices = state.models.filter((model) => model.mode === state.mode);
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
  if (result?.status === "ok") {
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
    output.append(prediction, scores, node("p", "Puntuaciones del modelo · uso académico", "fine"));
  } else {
    const model = currentModel();
    const message = result?.message || (result?.status === "running" ? "Procesando esta imagen…" :
      !row ? "Agrega una imagen para comenzar." :
      model?.status === "ready" ? "Ejecuta la predicción para ver las puntuaciones." : "Modelo no disponible.");
    output.append(node("p", message, result?.status === "error" ? "waiting row-error" : "waiting"));
  }
  output.append(referenceSection(row, result));
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

function tableRow(row) {
  const tr = node("tr");
  tr.dataset.id = row.id;
  tr.classList.toggle("active", row.id === state.active);
  const result = currentResult(row);
  const ref = row.reference;
  const comparison = compare(result, ref, state.mode);
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

  const predCell = node("td", result?.prediction ? labels[result.prediction] : "—");
  const refCell = node("td");
  refCell.append(node("span", ref ? labels[ref.label] : "Sin etiqueta de referencia", ref ? "" : "dim"));
  if (ref) {
    const refLevel = ref.level === "ternary" ? "Trinaria" :
      state.mode === "ternary" ? "Binaria · severidad no disponible" : "Binaria";
    refCell.append(node("span", refLevel, "row-id"));
  }
  const compCell = node("td", comparison ? "" : "—");
  if (comparison) {
    compCell.append(comparisonText(comparison));
    if (comparison.level === "derived") compCell.append(node("span", "Binaria derivada", "row-id"));
    if (comparison.level === "grouped") compCell.append(node("span", "Referencia agrupada", "row-id"));
  }
  const statusCell = node("td");
  statusCell.append(badge(...buttonStatus(result)));
  if (result?.message) statusCell.append(node("span", result.message, "row-error"));
  tr.append(imageCell, predCell, refCell, compCell, statusCell);
  return tr;
}

function showTable() {
  const focused = document.activeElement?.closest("tr")?.dataset.id;
  const rows = $("rows");
  if (state.rows.length) {
    rows.replaceChildren(...state.rows.map(tableRow));
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
  let done = 0;
  let failures = 0;
  let comparable = 0;
  let matches = 0;
  let derived = 0;
  let derivedMatches = 0;
  for (const row of state.rows) {
    const result = currentResult(row);
    if (result?.status === "error") failures++;
    if (result?.status === "ok") done++;
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
  if (comparable) parts.push(`${matches}/${comparable} coincidencias`);
  if (derived) parts.push(`${derivedMatches}/${derived} coincidencias binarias derivadas`);
  if (done && done - comparable - derived) parts.push(`${done - comparable - derived} sin comparación`);
  $("batch-summary").textContent = total ? parts.join(" · ") : "Agrega una imagen o carga el lote de ejemplo.";
  const bar = $("progress");
  bar.hidden = !state.running || !state.progress;
  if (state.progress) {
    bar.max = state.progress.total;
    bar.value = state.progress.done;
  }
}

function showConfusion() {
  const container = $("confusion-matrices");
  const cases = state.rows.map((row) => ({ result: currentResult(row), reference: row.reference }));
  const { title, classes, counts, total, binaryHealthy, withoutStage } = confusionMatrix(cases, state.mode);
  const wrapper = node("div", "", "matrix");
  const table = node("table");
  table.append(node("caption", title + " · " + total + (total === 1 ? " caso" : " casos")));
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
  $("load-examples").disabled = busy || state.examples.length === 0;
  $("run").disabled = busy || !canRun;
  $("run").textContent = state.running ? "Procesando…" : "Ejecutar predicción";
  $("clear").disabled = busy || state.rows.length === 0;
  $("import-csv").disabled = busy || !state.rows.some((row) => row.file);
}

function render() {
  showCase();
  showTable();
  showSummary();
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
    const query = new URLSearchParams({ model_id: model.id });
    if (!row.file) query.set("example_id", row.id);
    try {
      const response = await fetch(`/api/predict?${query}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: row.file,
      });
      const result = await response.json();
      row.results.set(model.id, result);
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
$("load-examples").addEventListener("click", () => loadExamples(state.examples));
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
      const choices = catalog.models.filter((model) => model.mode === mode);
      const preferred = choices.find((model) => model.id === catalog.defaults[mode]);
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
    if (state.examples.length && !state.rows.length) loadExamples(state.examples, true);
  } catch {
    $("model-status").textContent = "No se pudo consultar el catálogo de modelos.";
    modelSelect.replaceChildren(new Option("Motor no disponible", ""));
  }
  render();
}

init();
