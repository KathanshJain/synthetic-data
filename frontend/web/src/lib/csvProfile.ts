import Papa from "papaparse";
import type { Upload } from "../api/types";
export function readLocalProfile(input: string): {
  profile: Upload["profile"];
  balanceFor: (column: string) => Record<string, number>;
} {
  const parsed = Papa.parse<string[]>(input, { skipEmptyLines: "greedy" });
  const [header, ...records] = parsed.data;
  const columns = header?.map((c) => c.trim()) ?? [];
  if (
    parsed.errors.length ||
    !columns.length ||
    columns.some((c) => !c) ||
    new Set(columns).size !== columns.length ||
    !records.length ||
    records.some((row) => row.length !== columns.length)
  )
    throw new Error(
      "Use a CSV with a unique header for each column and the same number of fields in every row.",
    );
  const entries = columns.map((column, i) => {
    const values = records.map((row) => row[i].trim()).filter(Boolean);
    const freq = new Map<string, number>();
    for (const value of values) freq.set(value, (freq.get(value) ?? 0) + 1);
    const idLike =
      /(^id$|[_\s-]id$|^id[_\s-]|email|phone|address|name)/i.test(column) ||
      (values.length > 4 && freq.size === values.length);
    return {
      column,
      missing: (100 * (records.length - values.length)) / records.length,
      type:
        values.length > 0 && values.every((v) => Number.isFinite(Number(v)))
          ? "numerical"
          : "categorical",
      freq,
      idLike,
    };
  });
  function balanceFor(column: string) {
    const match = entries.find((e) => e.column === column);
    return match && !match.idLike && match.freq.size <= 20 ? Object.fromEntries(match.freq) : {};
  }
  return {
    profile: {
      rows: records.length,
      columns,
      types: Object.fromEntries(entries.map((e) => [e.column, e.type])),
      missing: Object.fromEntries(entries.map((e) => [e.column, e.missing])),
      warnings: entries
        .filter((e) => e.idLike)
        .map(
          (e) =>
            `“${e.column}” looks like an identifier. Remove identifiers before generating records.`,
        ),
      class_balance: balanceFor(columns[0]),
    },
    balanceFor,
  };
}
