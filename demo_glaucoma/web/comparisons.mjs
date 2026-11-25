export const labels = {
  normal: "Sano",
  glaucoma: "Glaucoma",
  early: "Temprano",
  advanced: "Avanzado",
};

const binary = (label) => (label === "normal" ? "normal" : "glaucoma");

export function compare(result, reference, mode) {
  if (result?.status !== "ok" || !reference) return null;
  if (mode === "ternary" && reference.level === "binary") {
    const projected = result.binary_projection;
    return projected && {
      level: "derived",
      match: projected.label === reference.label,
      projected,
    };
  }
  const expected = mode === "binary" ? binary(reference.label) : reference.label;
  return {
    level: mode === "binary" && reference.level === "ternary" ? "grouped" : "direct",
    match: result.prediction === expected,
    expected,
  };
}

export function confusionMatrix(cases, mode) {
  const classes = mode === "binary" ? ["normal", "glaucoma"] : ["normal", "early", "advanced"];
  const counts = classes.map(() => classes.map(() => 0));
  let total = 0;
  let binaryHealthy = 0;
  let withoutStage = 0;
  for (const { result, reference } of cases) {
    if (result?.status !== "ok" || !reference) continue;
    if (mode === "ternary" && reference.level === "binary") {
      if (reference.label !== "normal") {
        withoutStage++;
        continue;
      }
      binaryHealthy++;
    }
    const expected = mode === "binary" ? binary(reference.label) : reference.label;
    counts[classes.indexOf(expected)][classes.indexOf(result.prediction)]++;
    total++;
  }
  const title = mode === "binary" ? "Clasificación binaria" : "Clasificación trinaria";
  return { title, classes, counts, total, binaryHealthy, withoutStage };
}

function positiveScore(result) {
  if (result?.binary_projection?.score != null) return result.binary_projection.score;
  if (result?.decision_summary?.positive_score != null) return result.decision_summary.positive_score;
  const item = result?.scores?.find((entry) => entry.label === "glaucoma");
  return item ? item.score : null;
}

function computeAuroc(pairs) {
  const positives = pairs.filter((item) => item.y === 1 && item.score != null);
  const negatives = pairs.filter((item) => item.y === 0 && item.score != null);
  if (!positives.length || !negatives.length) return null;
  let wins = 0;
  for (const pos of positives) {
    for (const neg of negatives) {
      if (pos.score > neg.score) wins += 1;
      else if (pos.score === neg.score) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
}

export function binaryMetrics(cases) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  const rocPairs = [];

  for (const { result, reference } of cases) {
    if (result?.status !== "ok" || !reference) continue;
    const y = binary(reference.label) === "glaucoma" ? 1 : 0;
    const predLabel = result.binary_projection?.label ?? result.prediction;
    const pred = predLabel === "glaucoma" ? 1 : 0;
    if (y === 1 && pred === 1) tp++;
    else if (y === 0 && pred === 1) fp++;
    else if (y === 0 && pred === 0) tn++;
    else fn++;
    rocPairs.push({ y, score: positiveScore(result) });
  }

  const pos = tp + fn;
  const neg = tn + fp;
  const n = pos + neg;
  if (!n) {
    return {
      n: 0, pos: 0, neg: 0, tp: 0, fp: 0, tn: 0, fn: 0,
      accuracy: null, precision: null, sensitivity: null, specificity: null,
      fpr: null, fnr: null, f1Macro: null, auroc: null,
    };
  }
  const f1Pos = (2 * tp + fp + fn) ? (2 * tp) / (2 * tp + fp + fn) : 0;
  const f1Neg = (2 * tn + fp + fn) ? (2 * tn) / (2 * tn + fp + fn) : 0;
  return {
    n,
    pos,
    neg,
    tp,
    fp,
    tn,
    fn,
    accuracy: (tp + tn) / n,
    precision: (tp + fp) ? tp / (tp + fp) : null,
    sensitivity: pos ? tp / pos : null,
    specificity: neg ? tn / neg : null,
    fpr: neg ? fp / neg : null,
    fnr: pos ? fn / pos : null,
    f1Macro: (f1Pos + f1Neg) / 2,
    auroc: computeAuroc(rocPairs),
  };
}
