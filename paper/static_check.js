const fs = require("fs");

const sourcePath = process.argv[2] || "paper/main_final_submission.tex";
const source = fs.readFileSync(sourcePath, "utf8");
const bibliography = fs.readFileSync("paper/references_IECON.bib", "utf8");
const collect = (expression, text) => new Set([...text.matchAll(expression)].map((match) => match[1]));
const labels = collect(/\\label\{([^}]+)\}/g, source);
const references = collect(/\\(?:ref|eqref)\{([^}]+)\}/g, source);
const citations = new Set(
  [...source.matchAll(/\\cite\{([^}]+)\}/g)].flatMap((match) => match[1].split(",").map((key) => key.trim())),
);
const keys = collect(/^@\w+\{([^,]+),/gm, bibliography);
const missing = (left, right) => [...left].filter((item) => !right.has(item));
const result = {
  undefinedReferences: missing(references, labels),
  missingCitations: missing(citations, keys),
  braceBalance: [...source].filter((character) => character === "{").length
    - [...source].filter((character) => character === "}").length,
  labels: labels.size,
  citations: citations.size,
  bibliographyEntries: keys.size,
};
console.log(result);
if (result.undefinedReferences.length || result.missingCitations.length || result.braceBalance !== 0) {
  process.exit(1);
}
