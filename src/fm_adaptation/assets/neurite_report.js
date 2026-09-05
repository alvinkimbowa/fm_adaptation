/* Standalone report: pure calculations are also exercised from Node against Python fixtures. */
(() => {
  "use strict";
  const finite = values => values.filter(Number.isFinite);
  const mean = values => values.reduce((sum, x) => sum + x, 0) / values.length;
  function fixed2(value) {
    // Python rounds exact halfway cases to even; toFixed rounds those away from zero.
    // Binary-representable hundredth midpoints are odd multiples of 1/8.
    const scaled = Math.abs(value) * 100;
    if (Number.isInteger(value * 8) && scaled % 1 === 0.5) {
      const lower = Math.floor(scaled);
      return (Math.sign(value) * (lower + lower % 2) / 100).toFixed(2);
    }
    return value.toFixed(2);
  }
  function percentile(values, q) {
    const sorted = [...values].sort((a, b) => a - b);
    if (!sorted.length) return NaN;
    const index = (sorted.length - 1) * q;
    const lo = Math.floor(index), hi = Math.ceil(index);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
  }
  function reduce(values, statistic) {
    const valid = finite(values);
    return !valid.length ? NaN : statistic === "Mean ± SD" ? mean(valid) : percentile(valid, 0.5);
  }
  function format(values, scale, statistic) {
    const valid = finite(values).map(x => x * scale);
    const missing = values.length - valid.length;
    let text = "—";
    if (valid.length && statistic === "Mean ± SD") {
      const average = mean(valid);
      const variance = valid.reduce((sum, x) => sum + (x - average) ** 2, 0) / Math.max(1, valid.length - 1);
      text = `${fixed2(average)} ± ${fixed2(Math.sqrt(variance))}`;
    } else if (valid.length) {
      text = `${fixed2(percentile(valid, 0.5))} (${fixed2(percentile(valid, 0.25))}–${fixed2(percentile(valid, 0.75))})`;
    }
    return text + (missing ? `<div class="undef">(${missing} undef.)</div>` : "");
  }
  const close = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b);

  function calculate(data, selected, showAverage) {
    const rankings = new Map();
    const rankKey = (group, column, metric) => JSON.stringify([group, column, metric]);
    const rows = data.rows.map(row => {
      const group = data.grouped ? row.trainedOn : null;
      const cross = data.metrics.map(() => []);
      const cells = [];
      const addCell = (column, metricIndex, values, reference, count) => {
        const metric = data.metrics[metricIndex];
        const value = values === null ? NaN : reduce(values, data.statistic);
        const key = rankKey(group, column, metric.name);
        if (!reference && Number.isFinite(value) && data.rows.length > 1) {
          if (!rankings.has(key)) rankings.set(key, []);
          rankings.get(key).push(value);
        }
        cells.push({values, value, reference, count, key, metric});
        return value;
      };
      data.datasets.forEach(column => {
        const dataset = column === data.ownTest ? row.trainedOn : column;
        const original = row.results[dataset];
        const reference = data.inDomain.some(([train, test]) => train === row.trainedOn && test === column);
        const indices = original ? original.identities.flatMap((who, i) =>
          !who || !selected[who[0]] || selected[who[0]].includes(who[1]) ? [i] : []) : [];
        data.metrics.forEach((metric, i) => {
          const available = original && Object.hasOwn(original.metrics, metric.name);
          const values = available ? indices.map(index => original.metrics[metric.name][index]) : null;
          const value = addCell(column, i, values, reference, original ? indices.length : null);
          if (column !== data.ownTest && !reference && available) cross[i].push(value);
        });
      });
      if (showAverage) data.metrics.forEach((_, i) => addCell("cross", i, cross[i], false, null));
      return cells;
    });
    for (const cells of rows) for (const cell of cells) {
      const ranking = [...new Set(rankings.get(cell.key) || [])].sort((a, b) => cell.metric.higher ? b - a : a - b);
      cell.rank = cell.reference ? 0 : close(cell.value, ranking[0]) ? 1 : close(cell.value, ranking[1]) ? 2 : 0;
    }
    return rows;
  }

  function cellHTML(cell, statistic) {
    let text = cell.values === null ? "—" : format(cell.values, cell.metric.scale, statistic);
    if (cell.rank) {
      const tag = cell.rank === 1 ? "strong" : "u";
      const start = text.indexOf("<div");
      const split = start === -1 ? text.length : start;
      text = `<${tag}>${text.slice(0, split)}</${tag}>${text.slice(split)}`;
    }
    if (cell.count !== null) text += `<div class="annotation-count">n=${cell.count}</div>`;
    return text;
  }

  function mount(root) {
    const data = JSON.parse(root.querySelector(".neurite-data").textContent);
    const table = root.querySelector("table");
    const showAverage = table.tHead.rows[0].lastElementChild.textContent === "Cross-dataset average";
    const boxes = [...root.querySelectorAll('input[type="checkbox"]')];
    function update() {
      const selected = {};
      boxes.forEach(box => {
        const source = box.dataset.source;
        if (!selected[source]) selected[source] = [];
        if (box.checked) selected[source].push(box.value);
      });
      root.querySelector(".selection-status").textContent = Object.entries(selected)
        .map(([source, names]) => `${source}: ${names.length ? names.join(", ") : "none"}`).join(" · ");
      calculate(data, selected, showAverage).forEach((cells, rowIndex) => {
        cells.forEach((cell, index) => {
          table.tBodies[0].rows[rowIndex].cells[index + 5].innerHTML = cellHTML(cell, data.statistic);
        });
      });
    }
    boxes.forEach(box => box.addEventListener("change", update));
    update();
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, format, reduce, percentile, close, cellHTML};
  if (typeof document !== "undefined") document.querySelectorAll(".neurite-model-report").forEach(mount);
})();
