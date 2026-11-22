export const labels = {
  normal: "Sano",
  glaucoma: "Glaucoma",
  early: "Temprano",
  advanced: "Avanzado",
};

const binary = (label) => label === "normal" ? "normal" : "glaucoma";

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
