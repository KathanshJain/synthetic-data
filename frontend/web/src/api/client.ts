import { fixtures } from "../mocks/fixtures";
import type {
  Health,
  Overview,
  Distribution,
  Seeds,
  Summary,
  Reference,
  Fidelity,
  Privacy,
  Histogram,
  Runtime,
  LossCurve,
  Criteria,
  Comparison,
  Correlation,
  Projection,
  Upload,
  Job,
  Model,
} from "./types";
const base = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");
export const useMocks = import.meta.env.VITE_USE_MOCKS !== "false";
export class DataUnavailable extends Error {
  constructor() {
    super("This data was not supplied with the research brief.");
  }
}
async function get<T>(path: string): Promise<T> {
  if (useMocks) {
    const key = path.split("?")[0] as keyof typeof fixtures;
    if (Object.hasOwn(fixtures, key)) return fixtures[key] as T;
    throw new DataUnavailable();
  }
  const response = await fetch(`${base}${path}`);
  if (!response.ok) throw new Error(`Could not load data (${response.status}).`);
  return response.json() as Promise<T>;
}
async function post<T>(path: string, body: BodyInit, headers?: HeadersInit): Promise<T> {
  if (useMocks)
    throw new Error(
      "The interactive lab needs the backend. Switch mock mode off when it is available.",
    );
  const response = await fetch(`${base}${path}`, { method: "POST", body, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.message ?? body?.detail ?? body?.error;
    throw new Error(
      typeof message === "string"
        ? message
        : `The request could not be completed (${response.status}).`,
    );
  }
  return response.json() as Promise<T>;
}
const params = (p: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)])).toString();
export const api = {
  health: () => get<Health>("/health"),
  overview: () => get<Overview>("/overview"),
  seeds: () => get<Seeds>("/seeds"),
  distribution: (column: string) =>
    get<Distribution>(`/dataset/distribution?${params({ column })}`),
  summary: (model: Model) => get<Summary[]>(`/results/summary?${params({ model })}`),
  reference: () => get<Reference>("/results/reference"),
  fidelity: () => get<Fidelity>("/results/fidelity"),
  privacy: () => get<Privacy[]>("/results/privacy"),
  privacyHistogram: (generator: string, seed_size: number, random_seed: number) =>
    get<Histogram>(`/results/privacy/histogram?${params({ generator, seed_size, random_seed })}`),
  runtime: () => get<Runtime[]>("/results/runtime"),
  loss: (seed_size: number, random_seed: number) =>
    get<LossCurve>(`/results/loss-curve?${params({ seed_size, random_seed })}`),
  criteria: () => get<Criteria>("/results/criteria"),
  compare: (column: string, generator: string, seed_size: number, random_seed: number) =>
    get<Comparison>(
      `/compare/distribution?${params({ column, generator, seed_size, random_seed })}`,
    ),
  correlation: (generator: string, seed_size: number, random_seed: number) =>
    get<Correlation>(`/compare/correlation?${params({ generator, seed_size, random_seed })}`),
  projection: (generator: string, seed_size: number, random_seed: number) =>
    get<Projection>(`/compare/projection?${params({ generator, seed_size, random_seed })}`),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return post<Upload>("/lab/upload", body);
  },
  run: (body: {
    dataset_id: string;
    target: string;
    seed_size: number;
    generator: string;
    synthetic_rows: number;
    ctgan_steps: number;
  }) =>
    post<{ job_id: string }>("/lab/run", JSON.stringify(body), {
      "Content-Type": "application/json",
    }),
  job: (id: string) => get<Job>(`/lab/jobs/${encodeURIComponent(id)}`),
  jobEvents: (id: string) => new EventSource(`${base}/lab/jobs/${encodeURIComponent(id)}/events`),
  downloadUrl: (id: string) => `${base}/lab/jobs/${encodeURIComponent(id)}/download`,
};
