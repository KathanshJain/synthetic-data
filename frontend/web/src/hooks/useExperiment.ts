import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { generators, seedSizes } from "../lib/study";
export function useExperiment(rows: { generator: string; seed_size: number }[] = []) {
  const { data } = useQuery({ queryKey: ["seeds"], queryFn: api.seeds });
  const [g, setGenerator] = useState(""),
    [s, setSize] = useState<number>(),
    [r, setRun] = useState<number>();
  const generatorOptions = rows.length ? [...new Set(rows.map((r) => r.generator))] : generators;
  const generator = generatorOptions.includes(g) ? g : generatorOptions[0];
  const sizes = rows.length
    ? [...new Set(rows.filter((r) => r.generator === generator).map((r) => r.seed_size))].sort(
        (a, b) => a - b,
      )
    : (data?.sizes ?? seedSizes);
  const size = sizes.includes(s ?? -1) ? s! : (sizes.at(-1) ?? seedSizes.at(-1)!);
  const runs = data?.random_seeds ?? [];
  const run = runs.includes(r ?? -1) ? r : runs[0];
  return { generator, generatorOptions, setGenerator, sizes, size, setSize, runs, run, setRun };
}
