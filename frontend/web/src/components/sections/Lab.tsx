import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Download, FlaskConical, LockKeyhole, UploadCloud } from "lucide-react";
import { api, useMocks } from "../../api/client";
import type { Upload } from "../../api/types";
import { readLocalProfile } from "../../lib/csvProfile";
import { terminal, useLabJob } from "../../hooks/useLabJob";
import { dcrCaveat, generators, generatorColor, generatorLabel } from "../../lib/study";
import { colors } from "../../lib/theme";
import { formatNumber } from "../../lib/transforms";
import { Reveal, SectionHeading } from "../ui/Primitives";
import { ContextNote, Control, QueryState } from "../ui/QueryState";
import { GaugeCard } from "../cards/GaugeCard";
import { ResearchChart } from "../charts/ResearchChart";
import { comparisonOption } from "../charts/options";
const isRecord = (raw: unknown): raw is Record<string, unknown> =>
  !!raw && typeof raw === "object" && !Array.isArray(raw);
const numericMap = (raw: unknown) =>
  isRecord(raw)
    ? Object.fromEntries(
        Object.entries(raw).filter(
          (e): e is [string, number] => typeof e[1] === "number" && Number.isFinite(e[1]),
        ),
      )
    : {};
function LabResult({ result, id, generator }: { result: unknown; id: string; generator: string }) {
  const data = isRecord(result) ? result : {};
  const balance = isRecord(data.class_balance) ? data.class_balance : {};
  const real = numericMap(balance.real),
    synth = numericMap(balance.synthetic);
  const labels = [...new Set([...Object.keys(real), ...Object.keys(synth)])];
  const fidelity = numericMap(data.fidelity),
    privacy = numericMap(data.privacy ?? data.dcr);
  const plots = isRecord(data.per_column) ? data.per_column : {};
  const choices = Object.keys(plots);
  const [picked, setPicked] = useState("");
  const column = choices.includes(picked) ? picked : choices[0];
  const plot = plots[column];
  const distribution = isRecord(plot) ? plot : {};
  const bins = distribution.bins ?? distribution.categories;
  const color = generatorColor(generator);
  return (
    <div className="section-stack lab-result">
      <div className="subhead">
        <div>
          <span className="section-kicker">YOUR EXPERIMENT</span>
          <h3>Explore your results</h3>
        </div>
        <a className="button secondary" href={api.downloadUrl(id)}>
          <Download size={16} /> Download CSV
        </a>
      </div>
      {labels.length > 0 && (
        <ResearchChart
          title="Class balance comparison"
          subtitle="Real and synthetic class counts."
          summary="Counts by reported target label."
          option={comparisonOption(
            labels,
            labels.map((l) => real[l] ?? 0),
            labels.map((l) => synth[l] ?? 0),
            color,
            "grouped",
          )}
        />
      )}
      {Object.keys(fidelity).length > 0 && (
        <div className="section-stack metric-grid">
          {Object.entries(fidelity).map(([label, value]) => (
            <GaugeCard key={label} title={label.replaceAll("_", " ")} value={value} color={color} />
          ))}
        </div>
      )}
      {Array.isArray(bins) &&
        Array.isArray(distribution.real) &&
        Array.isArray(distribution.synthetic) && (
          <div className="section-stack">
            <ResearchChart
              title="Feature comparison"
              subtitle="Inspect the selected column in both datasets."
              summary={`Real and synthetic distribution for ${column}.`}
              option={comparisonOption(
                bins,
                distribution.real,
                distribution.synthetic,
                color,
                "overlay",
              )}
            >
              <div className="filter-strip inline-controls">
                <Control
                  label="Result column"
                  value={column}
                  options={choices}
                  onChange={setPicked}
                />
              </div>
            </ResearchChart>
          </div>
        )}
      {Object.keys(privacy).length > 0 && (
        <>
          <div className="caveat">{dcrCaveat}</div>
          <div className="section-stack metric-grid">
            {Object.entries(privacy).map(([label, value]) => (
              <GaugeCard
                key={label}
                title={label.replaceAll("_", " ")}
                value={value}
                color={color}
              />
            ))}
          </div>
        </>
      )}
      {!labels.length &&
        !Object.keys(fidelity).length &&
        !choices.length &&
        !Object.keys(privacy).length && (
          <ContextNote>
            The job finished, but the response didn’t include a recognized chart payload. Your CSV
            is ready to download.
          </ContextNote>
        )}
    </div>
  );
}
export function Lab() {
  const [file, setFile] = useState(""),
    [upload, setUpload] = useState<Upload | null>(null),
    [target, setTarget] = useState(""),
    [generator, setGenerator] = useState(generators[0]),
    [seed, setSeed] = useState(100),
    [rows, setRows] = useState(100),
    [steps, setSteps] = useState(100),
    [jobId, setJobId] = useState(""),
    [error, setError] = useState("");
  const local = useRef<ReturnType<typeof readLocalProfile>>();
  const input = useRef<HTMLInputElement>(null);
  const job = useLabJob(jobId);
  const uploadMutation = useMutation({
    mutationFn: async (f: File) => {
      if (!/\.csv$/i.test(f.name)) throw new Error("Choose a file with a .csv extension.");
      if (!useMocks) return api.upload(f);
      if (f.size > 5 * 1024 * 1024)
        throw new Error("The browser-only preview accepts CSV files up to 5 MB.");
      local.current = readLocalProfile(await f.text());
      return { dataset_id: "local-preview", profile: local.current.profile };
    },
    onSuccess: (u) => {
      setUpload(u);
      setTarget(u.profile.columns[0] ?? "");
      setSeed(Math.min(100, u.profile.rows));
    },
    onError: (e) => setError(e.message),
  });
  const run = useMutation({
    mutationFn: () =>
      api.run({
        dataset_id: upload!.dataset_id,
        target,
        seed_size: seed,
        generator,
        synthetic_rows: rows,
        ctgan_steps: steps,
      }),
    onSuccess: (r) => setJobId(r.job_id),
    onError: (e) => setError(e.message),
  });
  const busy = uploadMutation.isPending || run.isPending || Boolean(jobId && !terminal(job.data));
  function select(f: File) {
    if (busy) return;
    setFile(f.name);
    setUpload(null);
    setJobId("");
    setError("");
    local.current = undefined;
    uploadMutation.mutate(f);
  }
  const profile = upload?.profile;
  const balance = useMocks
    ? (local.current?.balanceFor(target) ?? {})
    : (profile?.class_balance ?? {});
  const balanceEntries = Object.entries(balance);
  const averageMissing = profile?.columns.length
    ? Object.values(profile.missing).reduce((sum, v) => sum + v, 0) / profile.columns.length
    : undefined;
  const phase = job.data
    ? terminal(job.data)
      ? job.data.result != null
        ? "done"
        : "stopped"
      : "running"
    : jobId
      ? "running"
      : profile
        ? "configure"
        : "upload";
  return (
    <section id="lab" className="section lab-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="09 / TRY IT YOURSELF"
            title="Curious? Make it your own."
            description="Bring a dataset, choose your settings, and explore what synthetic data can do for you."
          />
        </Reveal>
        <div className="lab-layout">
          <div className="lab-copy">
            <span className="feature-icon feature-0">
              <FlaskConical size={21} />
            </span>
            <h3>
              From your data
              <br />
              to a new perspective.
            </h3>
            <p>
              {useMocks
                ? "Profile a CSV locally to preview the workflow. This mode does not send your file anywhere or pretend to train a generator."
                : "Upload a CSV to profile it, choose a generator, and inspect the results. Only upload data you have permission to process."}
            </p>
            <div className="lab-steps" aria-label="Experiment stages">
              {[
                ["upload", "Upload a CSV"],
                ["configure", "Configure the run"],
                ["running", "Generate & evaluate"],
                ["done", "Explore the results"],
              ].map(([key, label], i) => (
                <div key={key} aria-current={key === phase ? "step" : undefined}>
                  <span>0{i + 1}</span>
                  {label}
                </div>
              ))}
            </div>
          </div>
          <div className="card upload-card">
            <div className="upload-header">
              <div>
                <h3>Start an experiment</h3>
                <p>Your dataset is the starting point.</p>
              </div>
              <span>{useMocks ? "LOCAL PREVIEW" : "CSV"}</span>
            </div>
            <label
              className="dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) select(f);
              }}
            >
              <UploadCloud size={27} />
              <strong>{file || "Drop your CSV file here"}</strong>
              <span>
                or <b>browse files</b> on your device
              </span>
              <input
                ref={input}
                aria-label="Upload CSV"
                type="file"
                accept=".csv,text/csv"
                disabled={busy}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) select(f);
                  e.target.value = "";
                }}
              />
            </label>
            {uploadMutation.isPending && (
              <div className="skeleton chart-skeleton" role="status" aria-label="Profiling CSV" />
            )}
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            {profile && (
              <>
                <div className="profile-stats">
                  <div>
                    <span>Rows</span>
                    <strong>{formatNumber(profile.rows)}</strong>
                  </div>
                  <div>
                    <span>Columns</span>
                    <strong>{formatNumber(profile.columns.length)}</strong>
                  </div>
                  <div>
                    <span>Mean missing</span>
                    <strong>
                      {averageMissing === undefined ? "—" : `${formatNumber(averageMissing, 1)}%`}
                    </strong>
                  </div>
                </div>
                {profile.warnings.map((w) => (
                  <ContextNote key={w}>{w}</ContextNote>
                ))}
                <form
                  className="lab-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (useMocks || busy) return;
                    setError("");
                    setJobId("");
                    run.mutate();
                  }}
                >
                  <Control
                    label="Target column"
                    value={target}
                    options={profile.columns}
                    onChange={setTarget}
                  />
                  <Control
                    label="Generator"
                    value={generator}
                    options={generators.map((value) => ({ value, label: generatorLabel(value) }))}
                    onChange={setGenerator}
                  />
                  <label>
                    Seed size
                    <input
                      type="number"
                      min={1}
                      max={profile.rows}
                      step={1}
                      required
                      value={seed || ""}
                      onChange={(e) => setSeed(+e.target.value)}
                    />
                  </label>
                  <label>
                    Synthetic rows
                    <input
                      type="number"
                      min={1}
                      step={1}
                      required
                      value={rows || ""}
                      onChange={(e) => setRows(+e.target.value)}
                    />
                  </label>
                  {generator === "CTGAN" && (
                    <>
                      <label>
                        CTGAN steps
                        <input
                          type="number"
                          min={1}
                          step={1}
                          required
                          value={steps || ""}
                          onChange={(e) => setSteps(+e.target.value)}
                        />
                      </label>
                      <p>CTGAN may be slow; more steps usually take longer.</p>
                    </>
                  )}
                  <button
                    className="button primary"
                    type="submit"
                    disabled={useMocks || busy || !target}
                  >
                    {run.isPending ? "Starting…" : "Run experiment"} <ArrowRight size={16} />
                  </button>
                </form>
              </>
            )}
            {jobId && !job.data && (
              <QueryState query={job} message="Waiting for an update from the experiment." />
            )}
            {job.data && (
              <div className="job-status">
                <div className="job-progress" role="status">
                  <strong>{job.data.stage}</strong>
                  <span>{formatNumber(job.data.progress)}%</span>
                </div>
                <progress max={100} value={job.data.progress} aria-label="Experiment progress" />
                <pre aria-label="Experiment log" tabIndex={0}>
                  {job.data.log.join("\n")}
                </pre>
                {job.data.error && (
                  <p role="alert" className="form-error">
                    {job.data.error}
                  </p>
                )}
              </div>
            )}
            <p className="upload-note">
              <LockKeyhole size={14} />
              {useMocks
                ? "Local profiling only. Connect the backend to generate records."
                : "Backend eligibility checks run before generation."}
            </p>
          </div>
        </div>
        {profile && (
          <div className="section-stack">
            <ResearchChart
              title="Uploaded class balance"
              subtitle={
                useMocks
                  ? `Reported counts for ${target}.`
                  : "Class balance reported by the uploaded dataset profile."
              }
              summary="Uploaded data only, not part of the published research results."
              option={
                balanceEntries.length
                  ? {
                      grid: { left: 50, right: 20, top: 22, bottom: 42 },
                      tooltip: { trigger: "axis", renderMode: "richText" },
                      xAxis: { type: "category", data: balanceEntries.map(([name]) => name) },
                      yAxis: { type: "value" },
                      series: [
                        {
                          type: "bar",
                          data: balanceEntries.map(([, n]) => n),
                          itemStyle: { color: colors.slate },
                        },
                      ],
                    }
                  : undefined
              }
            >
              {!balanceEntries.length && (
                <ContextNote>
                  No class counts are available for this selection. Local preview only lists
                  low-cardinality, non-identifier values.
                </ContextNote>
              )}
            </ResearchChart>
          </div>
        )}
        {job.data?.result != null && (
          <LabResult key={jobId} result={job.data.result} id={jobId} generator={generator} />
        )}
      </div>
    </section>
  );
}
