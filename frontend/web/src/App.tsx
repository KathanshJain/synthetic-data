import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  motion,
  useInView,
  useMotionValue,
  useReducedMotion,
  useScroll,
  useSpring,
  useTransform,
} from "framer-motion";
import Lenis from "lenis";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  Database,
  GitBranch,
  LockKeyhole,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { api, useMocks } from "./api/client";
import { formatNumber } from "./lib/transforms";
import { Reveal, SectionHeading, transition } from "./components/ui/Primitives";
import { Utility } from "./components/sections/Utility";
import { FidelityStudio, PrivacyRadar } from "./components/sections/Studios";
import { Cost } from "./components/sections/Cost";
import { Lab } from "./components/sections/Lab";
import { DataSnapshot } from "./components/sections/DataSnapshot";
import { Count } from "./components/cards/StatCard";
import { QueryState } from "./components/ui/QueryState";
import { seedSizes, generatorLabel, generatorColor, frozenRules } from "./lib/study";
import { colors } from "./lib/theme";

const nav = [
  ["story", "The story"],
  ["data", "The data"],
  ["utility", "Results"],
  ["criteria", "The finding"],
  ["lab", "Try it"],
];
const steps = [
  ["01", "Clean the data", "Remove duplicates and prepare the dataset for analysis."],
  ["02", "Lock the test set", "Set aside unseen records before any modeling begins."],
  [
    "03",
    "Build nested seeds",
    "Sample progressively larger subsets from the same development pool.",
  ],
  ["04", "Generate new data", "Train two generators on each seed, then create synthetic records."],
  ["05", "Check the details", "Compare distributions and look for potential privacy warnings."],
  ["06", "Put utility to the test", "Train and compare models across five experimental arms."],
  ["07", "Repeat & analyze", "Repeat across random seeds and evaluate the frozen criteria."],
];

function Logo() {
  return (
    <a className="brand" href="#top" aria-label="Forged Data home">
      <span className="brand-mark">
        <i />
        <i />
        <i />
        <i />
      </span>
      <span>
        forged<span className="brand-accent">.</span>data
      </span>
    </a>
  );
}
function Header() {
  const [active, setActive] = useState("");
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) {
            const id = e.target.id;
            setActive(
              (
                {
                  vault: "data",
                  seeds: "data",
                  fidelity: "utility",
                  privacy: "utility",
                  cost: "criteria",
                } as Record<string, string>
              )[id] ?? id,
            );
          }
        }),
      { rootMargin: "-30% 0px -60% 0px" },
    );
    [
      "top",
      "story",
      "data",
      "utility",
      "criteria",
      "lab",
      "vault",
      "seeds",
      "fidelity",
      "privacy",
      "cost",
      "methods",
    ].forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);
  return (
    <header className="header">
      <div className="header-inner">
        <Logo />
        <nav className={open ? "nav open" : "nav"} aria-label="Main navigation" id="section-nav">
          {nav.map(([id, label]) => (
            <a
              className={active === id ? "active" : ""}
              aria-current={active === id ? "location" : undefined}
              key={id}
              href={`#${id}`}
              onClick={() => setOpen(false)}
            >
              {label}
            </a>
          ))}
        </nav>
        <a className="header-link" href="#methods">
          About the project <ArrowUpRight size={14} />
        </a>
        <button
          className="menu-button"
          aria-controls="section-nav"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          aria-label={open ? "Close navigation" : "Open navigation"}
        >
          {open ? <X /> : <Menu />}
        </button>
      </div>
    </header>
  );
}
function FloatCard({
  kind,
  y,
  children,
}: {
  kind: "real" | "synth";
  y: ReturnType<typeof useTransform<number, number>>;
  children: React.ReactNode;
}) {
  const reduced = useReducedMotion();
  const x = useMotionValue(0),
    tilt = useMotionValue(0);
  const rotateY = useSpring(x, { stiffness: 180, damping: 23 }),
    rotateX = useSpring(tilt, { stiffness: 180, damping: 23 });
  return (
    <motion.div
      className={`float-anchor float-${kind}`}
      style={reduced ? undefined : { y, perspective: 620 }}
      onPointerMove={(e) => {
        if (reduced || e.pointerType !== "mouse") return;
        const r = e.currentTarget.getBoundingClientRect();
        x.set(((e.clientX - r.left - r.width / 2) / r.width) * 24);
        tilt.set(((r.height / 2 - e.clientY + r.top) / r.height) * 18);
      }}
      onPointerLeave={() => {
        x.set(0);
        tilt.set(0);
      }}
    >
      <motion.div
        className={`visual-card card-${kind}`}
        style={reduced ? undefined : { rotateY, rotateX, rotate: kind === "real" ? -5 : 4 }}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}
function Hero() {
  const { scrollY } = useScroll();
  const reduced = useReducedMotion();
  const y1 = useTransform(scrollY, [0, 700], [0, 100]);
  const y2 = useTransform(scrollY, [0, 700], [0, -90]);
  const rotate = useTransform(scrollY, [0, 700], [0, 18]);
  const gridY = useTransform(scrollY, [0, 700], [0, 35]);
  const dotsY = useTransform(scrollY, [0, 700], [0, -38]);
  const dotsX = useTransform(scrollY, [0, 550], [0, 32]);
  const morph = useTransform(scrollY, [0, 525], [colors.indigo, colors.teal]);
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });

  return (
    <section className="hero" id="top">
      <div className="hero-orb orb-one" />
      <div className="hero-orb orb-two" />
      <div className="container hero-grid">
        <motion.div
          className="hero-copy"
          initial={{ opacity: 0, y: reduced ? 0 : 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={transition}
        >
          <div className="pill">
            <span className="pulse-dot" /> MACHINE LEARNING ON ARTIFICIALLY FORGED DATA
          </div>
          <h1>
            Real questions.
            <br />
            <span>Synthetic</span>
            <br />
            possibilities<span className="period">.</span>
          </h1>
          <p className="hero-description">
            How much real data does it take to create synthetic data we can actually trust?
          </p>
          <p className="hero-sub">
            A closer look at the trade-offs between realism, usefulness, and privacy.
          </p>
          <div className="hero-actions">
            <motion.a
              className="button primary"
              href="#utility"
              onPointerMove={(e) => {
                if (reduced || e.pointerType !== "mouse") return;
                const box = e.currentTarget.getBoundingClientRect();
                e.currentTarget.style.setProperty(
                  "--mag-x",
                  `${(e.clientX - box.left - box.width / 2) * 0.13}px`,
                );
                e.currentTarget.style.setProperty(
                  "--mag-y",
                  `${(e.clientY - box.top - box.height / 2) * 0.2}px`,
                );
              }}
              onPointerLeave={(e) => {
                e.currentTarget.style.removeProperty("--mag-x");
                e.currentTarget.style.removeProperty("--mag-y");
              }}
            >
              Explore results <ArrowRight size={17} />
            </motion.a>
            <a className="button secondary" href="#lab">
              Try it yourself <ArrowUpRight size={17} />
            </a>
          </div>
          <div className="hero-note">
            <span className="tiny-line" /> AN EXPERIMENT IN DATA, NOT ASSUMPTIONS
          </div>
        </motion.div>
        <div
          className="hero-visual"
          aria-label="Illustration of real data transforming into synthetic data"
        >
          <motion.div className="visual-grid" style={reduced ? undefined : { y: gridY }} />
          <motion.div
            className="visual-ring ring-one"
            transformTemplate={(_, transform) =>
              `translate(-50%, -50%) ${transform === "none" ? "" : transform}`
            }
            style={reduced ? undefined : { y: y1, rotate }}
          />
          <motion.div
            className="visual-ring ring-two"
            transformTemplate={(_, transform) =>
              `translate(-50%, -50%) ${transform === "none" ? "" : transform}`
            }
            style={reduced ? undefined : { y: y2 }}
          />
          <motion.div className="particle-field" style={reduced ? {} : { y: dotsY, x: dotsX }}>
            {Array.from({ length: 45 }, (_, i) => (
              <motion.span
                key={i}
                style={{
                  left: `${8 + ((i * 37) % 84)}%`,
                  top: `${8 + ((i * 61) % 80)}%`,
                  width: i % 5 === 0 ? 9 : 5,
                  height: i % 5 === 0 ? 9 : 5,
                  backgroundColor: reduced ? (i % 3 === 0 ? colors.teal : colors.indigo) : morph,
                  opacity: 0.25 + (i % 4) * 0.15,
                }}
              />
            ))}
          </motion.div>
          <FloatCard kind="real" y={y1}>
            <div className="mini-card-icon indigo-bg">
              <Database size={17} />
            </div>
            <div>
              <small>INPUT DATA</small>
              <strong>Real records</strong>
              <span>Where it all begins</span>
            </div>
            <div className="mini-dots">
              <i />
              <i />
              <i />
            </div>
          </FloatCard>
          <motion.div className="visual-center" style={reduced ? {} : { y: y2 }}>
            <span className="center-spark">
              <Sparkles size={30} />
            </span>
            <span className="orbit orbit-a" />
            <span className="orbit orbit-b" />
          </motion.div>
          <FloatCard kind="synth" y={y2}>
            <div className="mini-card-icon teal-bg">
              <Sparkles size={17} />
            </div>
            <div>
              <small>GENERATED DATA</small>
              <strong>Synthetic records</strong>
              <span>Similar, not the same</span>
            </div>
            <div className="mini-dots teal-dots">
              <i />
              <i />
              <i />
            </div>
          </FloatCard>
          <div className="visual-caption">
            <span>THE IDEA</span> Real data in. New possibilities out. <ArrowUpRight size={14} />
          </div>
        </div>
      </div>
      <div className="container hero-bottom">
        <div className="hero-scroll">
          <span>SCROLL TO EXPLORE</span>
          <ArrowDown size={15} />
        </div>
        <div className="hero-metrics">
          {[
            [overview.data?.rows_clean, "clean records"],
            [overview.data?.n_features, "features studied"],
            [overview.data?.dev_rows, "development rows"],
            [overview.data?.locked.n_rows, "locked test rows"],
          ].map(([value, label]) => (
            <div key={label}>
              <strong>
                <Count value={value as number | undefined} />
              </strong>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
function Story() {
  const [step, setStep] = useState(0);
  const stepsRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) setStep(Number((e.target as HTMLElement).dataset.step));
        }),
      { rootMargin: "-36% 0px -48% 0px" },
    );
    stepsRef.current?.querySelectorAll("[data-step]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
  return (
    <section id="story" className="section story-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="THE APPROACH"
            title="Good questions deserve a good process."
            description="From real records to reliable conclusions, every step is designed to keep the experiment honest."
          />
        </Reveal>
        <div className="story-layout">
          <div className="story-sticky card">
            <div className="diagram-top">
              <span className="section-kicker">THE EXPERIMENT</span>
              <span className="diagram-index">0{step + 1} / 07</span>
            </div>
            <div className="diagram">
              <div className="diagram-line" />
              <svg
                className="diagram-draw"
                aria-hidden="true"
                viewBox="0 0 10 100"
                preserveAspectRatio="none"
              >
                <motion.path
                  d="M 5 0 V 100"
                  stroke={colors.indigo}
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                  strokeDasharray="100"
                  initial={false}
                  animate={{ strokeDashoffset: String(100 - (step / (steps.length - 1)) * 100) }}
                  transition={reduced ? { duration: 0 } : transition}
                />
              </svg>
              {steps.map((s, i) => (
                <button
                  key={s[0]}
                  className={
                    i === step
                      ? "diagram-node current"
                      : i < step
                        ? "diagram-node done"
                        : "diagram-node"
                  }
                  onClick={() => setStep(i)}
                  aria-pressed={i === step}
                  aria-label={`Show step ${i + 1}: ${s[1]}`}
                >
                  <span>{i < step ? <Check size={15} /> : s[0]}</span>
                  <b>{s[1]}</b>
                </button>
              ))}
            </div>
            <div className="diagram-footer">
              <span className="diagram-dot" /> {steps[step][2]}
            </div>
          </div>
          <div className="story-steps" ref={stepsRef}>
            {steps.map((s, i) => (
              <Reveal key={s[0]}>
                <div
                  className={i === step ? "story-step selected" : "story-step"}
                  data-step={i}
                  onMouseEnter={() => setStep(i)}
                  onFocus={() => setStep(i)}
                  tabIndex={0}
                >
                  <span>{s[0]}</span>
                  <div>
                    <h3>{s[1]}</h3>
                    <p>{s[2]}</p>
                  </div>
                  <ArrowUpRight size={17} />
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
function FingerprintText({ hash }: { hash: string }) {
  const ref = useRef<HTMLElement>(null),
    seen = useInView(ref, { once: true });
  const reduced = useReducedMotion();
  const [length, setLength] = useState(0);
  useEffect(() => {
    if (!seen || reduced) return;
    let last = 0;
    const start = performance.now();
    function frame(t: number) {
      const count = Math.min(hash.length, Math.ceil(((t - start) / 740) * hash.length));
      setLength(count);
      if (count < hash.length) last = requestAnimationFrame(frame);
    }
    last = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(last);
  }, [seen, reduced, hash]);
  return (
    <code ref={ref} aria-label={hash}>
      <span aria-hidden="true">{hash.slice(0, reduced ? hash.length : length)}</span>
    </code>
  );
}
function Vault() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const reduced = useReducedMotion();
  const locked = overview.data?.locked;
  return (
    <section className="section" id="vault">
      <div className="container vault-layout">
        <div className="vault-art">
          <div className="vault-circle">
            <div className="vault-square">
              <svg
                width="66"
                height="66"
                viewBox="0 0 64 64"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                aria-hidden="true"
              >
                <motion.path
                  d="M21 29v-8a11 11 0 0 1 22 0v8"
                  strokeDasharray="60"
                  initial={reduced ? false : { strokeDashoffset: "60" }}
                  whileInView={{ strokeDashoffset: "0" }}
                  viewport={{ once: true }}
                  transition={transition}
                />
                <motion.rect
                  x="15"
                  y="29"
                  width="34"
                  height="25"
                  rx="7"
                  strokeDasharray="120"
                  initial={reduced ? false : { strokeDashoffset: "120" }}
                  whileInView={{ strokeDashoffset: "0" }}
                  viewport={{ once: true }}
                  transition={{ ...transition, delay: reduced ? 0 : 0.14 }}
                />
                <circle cx="32" cy="39" r="2.5" />
                <path d="M32 42v4" />
              </svg>
            </div>
          </div>
          <span className="vault-floating">
            <ShieldCheck size={16} /> HELD BACK & PROTECTED
          </span>
        </div>
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            02 / THE LOCKED VAULT
          </div>
          <h2>
            Some things are better
            <br />
            <em>left untouched.</em>
          </h2>
          <p className="large-copy">
            A fair test starts with data the model has never seen. Our test set is locked away
            before the experiment begins.
          </p>
          <div className="vault-info card">
            <div>
              <span>
                TEST SET FINGERPRINT{" "}
                <span
                  className="hash-hint"
                  tabIndex={0}
                  aria-label="A hash is a fingerprint. It detects accidental edits; it is not encryption."
                  title="A hash is a fingerprint. It detects accidental edits; it is not encryption."
                >
                  ⓘ
                </span>
              </span>
              {locked?.sha256 ? (
                <FingerprintText hash={locked.sha256} />
              ) : (
                <code>Fingerprint not supplied</code>
              )}
            </div>
            <span role="status" className={`status-chip ${locked?.verified ? "verified" : ""}`}>
              <span /> {locked ? (locked.verified ? "VERIFIED" : "NOT VERIFIED") : "AWAITING DATA"}
            </span>
          </div>
          <div className="vault-meta">
            <span>
              Locked rows <b>{formatNumber(locked?.n_rows)}</b>
            </span>
            <span>
              Test class ratio{" "}
              <b>
                {overview.data
                  ? `${formatNumber((1 - overview.data.positive_rate_test) * 100, 1)}% / ${formatNumber(overview.data.positive_rate_test * 100, 1)}%`
                  : "—"}
              </b>
            </span>
          </div>
          <div className="guard">
            <ShieldCheck size={18} />
            <strong>Leakage guard</strong>
          </div>
          <ul className="guard-list">
            <li>
              {locked?.verified ? <Check size={14} /> : <span className="guard-dash">—</span>}{" "}
              {locked?.verified
                ? "Test set fingerprint matches"
                : locked
                  ? "Test set fingerprint not verified"
                  : "Fingerprint verification pending"}
            </li>
            <li>
              {health.data?.tests_total && health.data.tests_total === health.data.tests_passing ? (
                <Check size={14} />
              ) : (
                <span className="guard-dash">—</span>
              )}{" "}
              {health.data
                ? `${health.data.tests_passing} of ${health.data.tests_total} pipeline tests passing`
                : "Pipeline test counts not supplied"}
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
function Seeds() {
  const q = useQuery({ queryKey: ["seeds"], queryFn: api.seeds });
  const [index, setIndex] = useState(0);
  const available = q.data?.sizes.length ? q.data.sizes : seedSizes;
  const size = available[Math.min(index, available.length - 1)];
  const counts = q.data?.class_counts[String(size)];
  const reduced = useReducedMotion();
  return (
    <section className="section soft-section" id="seeds">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="03 / NESTED SEEDS"
            title="A little data goes a long way. Or does it?"
            description="We start small, then add more real records. Each larger seed contains the smaller one before it."
          />
        </Reveal>
        <div className="seed-layout">
          <div className="card seed-card">
            <div className="chart-header">
              <div>
                <h3>Growing the training set</h3>
                <p>Each subset builds on the one before it.</p>
              </div>
              <GitBranch size={20} />
            </div>
            <div className="seed-bars">
              {available.map((n, i) => (
                <button
                  key={n}
                  className={index === i ? "seed-row chosen" : "seed-row"}
                  onClick={() => setIndex(i)}
                >
                  <span>{n.toLocaleString()}</span>
                  <div>
                    <motion.i
                      className="seed-extent"
                      initial={reduced ? false : { width: 0 }}
                      whileInView={{ width: `${(n / Math.max(...available)) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ ...transition, delay: reduced ? 0 : i * 0.08 }}
                    >
                      {q.data?.class_counts[n] && (
                        <b
                          title="Minority class share"
                          style={{ width: `${(q.data.class_counts[n].minority / n) * 100}%` }}
                        />
                      )}
                    </motion.i>
                  </div>
                  <span>records</span>
                </button>
              ))}
            </div>
            <input
              aria-label="Select seed size"
              aria-valuetext={`${size} records`}
              type="range"
              min="0"
              max={available.length - 1}
              value={Math.min(index, available.length - 1)}
              onChange={(e) => setIndex(Number(e.target.value))}
            />
            <div className="range-labels">
              <span>SMALLER SEED</span>
              <span>LARGER SEED</span>
            </div>
          </div>
          <div className="seed-aside">
            <span className="section-kicker">SELECTED SEED SIZE</span>
            <strong>{size?.toLocaleString()}</strong>
            <p>real records to learn from</p>
            <div className="split-line" />
            <div className="split-stat">
              <span>Majority class</span>
              <b>{formatNumber(counts?.majority)}</b>
            </div>
            <div className="split-stat">
              <span>Minority class</span>
              <b>{formatNumber(counts?.minority)}</b>
            </div>
            <div className="note-small">
              {counts
                ? q.data?.nested_ok
                  ? "Nested membership verified. Indigo represents majority; teal represents minority."
                  : "The API did not verify nested membership."
                : "Class counts weren’t supplied. The bars show planned seed sizes only."}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
function Criteria() {
  const q = useQuery({ queryKey: ["criteria"], queryFn: api.criteria });
  const [open, setOpen] = useState(false);
  const reduced = useReducedMotion();
  const hasScores = !!q.data?.rules.length;
  const ruleColumns = q.data?.rules.length ? q.data.rules : frozenRules;
  const total = q.data?.evaluations.length;
  const passing = q.data?.evaluations.filter((row) => row.all_passed).length;
  return (
    <section id="criteria" className="section finding-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="07 / THE FINDING"
            title="A result worth finding."
            description="The criteria were decided in advance. The answer is meaningful, even when it isn't the one we hoped for."
          />
        </Reveal>
        <div className="finding-card">
          <div className="finding-main">
            <span className="finding-badge">
              <span /> THE PRE-REGISTERED RESULT
            </span>
            <h3>
              {q.data?.min_useful_seed === null
                ? "No tested seed size passed every frozen rule."
                : q.data?.min_useful_seed != null
                  ? `Minimum useful seed: ${q.data.min_useful_seed}`
                  : "The finding is waiting for data."}
            </h3>
            <p>
              {q.data?.min_useful_seed === null
                ? "That is a valid, pre-registered finding."
                : q.data?.explanation ||
                  "Connect the research API to view the frozen criteria and evaluations."}
            </p>
            {!!total && (
              <div className="finding-count">
                <b>
                  {passing} / {total}
                </b>{" "}
                combinations passed
              </div>
            )}
            <button className="why-button" onClick={() => setOpen(!open)} aria-expanded={open}>
              Why? <ChevronDown size={16} className={open ? "rotated" : ""} />
            </button>
            {open && (
              <div className="why-content">
                {q.data?.explanation || "The criteria data is not available yet."}{" "}
                {q.data?.evaluations.map((e) => {
                  const failures = q.data!.rules.filter((r) => e.passed[r.id] === false);
                  return failures.length ? (
                    <p key={`${e.generator}-${e.seed_size}`}>
                      {e.generator} · {e.seed_size}:{" "}
                      {failures
                        .map(
                          (r) =>
                            `${r.label} ${formatNumber(e.values[r.id], 3)} (${r.direction} ${r.threshold})`,
                        )
                        .join("; ")}
                    </p>
                  ) : null;
                })}
              </div>
            )}
          </div>
          <div className="finding-art" aria-hidden="true">
            <div className="finding-ring">
              <span>
                <Check size={35} />
              </span>
            </div>
            <div className="finding-label">
              A VALID
              <br />
              FINDING <ArrowUpRight size={18} />
            </div>
          </div>
        </div>
        {q.data && q.data.evaluations.length > 0 ? (
          <div className="scoreboard card">
            <div className="chart-header">
              <div>
                <h3>Frozen criteria scoreboard</h3>
                <p>
                  {hasScores
                    ? "Hover or focus on a result to see its value and threshold."
                    : "Overall outcomes are known. A dash means the individual rule score was not supplied."}
                </p>
              </div>
            </div>
            <div
              className="score-scroll"
              role="region"
              aria-label="Frozen criteria results"
              tabIndex={0}
            >
              <table>
                <thead>
                  <tr>
                    <th>Generator / seed</th>
                    {ruleColumns.map((rule) => (
                      <th key={rule.id}>{rule.label}</th>
                    ))}
                    <th>Overall</th>
                  </tr>
                </thead>
                <tbody>
                  {q.data.evaluations.map((e) => (
                    <tr key={`${e.generator}-${e.seed_size}`}>
                      <th scope="row">
                        <i
                          aria-hidden="true"
                          className="generator-dot"
                          style={{ background: generatorColor(e.generator) }}
                        />{" "}
                        {generatorLabel(e.generator)} · {e.seed_size}
                      </th>
                      {ruleColumns.map((rule, index) => (
                        <td key={rule.id}>
                          <motion.span
                            initial={e.passed[rule.id] === undefined ? false : { opacity: 0 }}
                            whileInView={{ opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{
                              ...transition,
                              delay: reduced ? 0 : Math.min(0.3, index * 0.05),
                            }}
                            tabIndex={0}
                            className={`score-cell ${e.passed[rule.id] === undefined ? "score-unknown" : e.passed[rule.id] ? "score-pass" : "score-fail"}`}
                            aria-label={`${rule.label}: ${e.passed[rule.id] === undefined ? "unavailable" : e.passed[rule.id] ? "pass" : "fail"}, value ${formatNumber(e.values[rule.id], 3)}, threshold ${rule.threshold ?? "not supplied"}`}
                          >
                            {e.passed[rule.id] === undefined ? (
                              "—"
                            ) : e.passed[rule.id] ? (
                              <Check size={15} />
                            ) : (
                              <X size={15} />
                            )}
                            <span className="score-tip" aria-hidden="true">
                              {e.values[rule.id] === undefined
                                ? "Score not supplied"
                                : `${formatNumber(e.values[rule.id], 3)} ${rule.direction} ${rule.threshold}`}
                            </span>
                          </motion.span>
                        </td>
                      ))}
                      <td>
                        <span className={e.all_passed ? "score-pass" : "score-fail"}>
                          {e.all_passed ? "Pass" : "Fail"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="card section-stack">
            <QueryState
              query={q}
              message="Detailed frozen rules and per-combination values were not supplied with the brief."
            />
          </div>
        )}
        <div className="finding-foot">
          <span>
            <i /> THE VALUE OF AN HONEST ANSWER
          </span>
          <p>
            Research isn't just about proving what works. It's also about understanding what
            doesn't.
          </p>
        </div>
      </div>
    </section>
  );
}
function Methods() {
  const items = [
    [
      "What is synthetic data?",
      "Artificially generated records that aim to preserve useful patterns from real data without simply reproducing it.",
    ],
    [
      "What do fidelity, utility, and privacy mean?",
      "Fidelity asks whether data looks statistically similar. Utility asks whether models trained on it are useful. Privacy considers whether individual records may be exposed.",
    ],
    [
      "What are KS, TVD, and DCR?",
      "KS compares numerical distributions, TVD compares categorical distributions, and DCR measures the distance to the closest real record. DCR is not a privacy guarantee.",
    ],
    [
      "What are PR-AUC, TSTR, and TRTR?",
      "PR-AUC summarizes precision and recall across thresholds. TSTR means train on synthetic, test on real; TRTR means train and test on real data.",
    ],
    [
      "What are the limitations?",
      "This study considers one dataset, two generators, and three random seeds. The CTGAN budget is fixed, and DCR does not guarantee privacy.",
    ],
    [
      "What comes next?",
      "Test more datasets and generators, vary the training budget, and investigate stronger privacy evaluations.",
    ],
  ];
  return (
    <section id="methods" className="section">
      <div className="container methods-layout">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            10 / METHODS & LIMITATIONS
          </div>
          <h2>
            Good science
            <br />
            asks <em>why.</em>
          </h2>
          <p className="large-copy">
            The details matter. Here's a plain-English guide to the ideas behind the experiment.
          </p>
        </div>
        <div className="accordion">
          {items.map(([title, body]) => (
            <details key={title}>
              <summary>
                {title}
                <ChevronDown size={18} />
              </summary>
              <p>{body}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
function Footer() {
  return (
    <footer>
      <div className="container">
        <div className="footer-top">
          <div>
            <Logo />
            <p>Real questions. Synthetic possibilities.</p>
          </div>
          <a href="#top">Back to top ↑</a>
        </div>
        <div className="footer-bottom">
          <span>A RESEARCH PROJECT AT BENNETT UNIVERSITY</span>
          <span>Aryan Gupta · GenAI & ML &nbsp; / &nbsp; Kathansh Jain · Data Analysis</span>
          <span>Supervised by Dr. Akhil Kumar</span>
        </div>
      </div>
    </footer>
  );
}
export default function App() {
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll();
  useEffect(() => {
    if (reduced) return;
    const lenis = new Lenis();
    let id: number;
    function raf(time: number) {
      lenis.raf(time);
      id = requestAnimationFrame(raf);
    }
    id = requestAnimationFrame(raf);
    return () => {
      cancelAnimationFrame(id);
      lenis.destroy();
    };
  }, [reduced]);
  return (
    <>
      <motion.div className="progress-bar" style={{ scaleX: scrollYProgress }} />
      <Header />
      <main>
        <Hero />
        {useMocks && (
          <aside className="container data-notice">
            <span className="data-notice-dot" />
            <strong>Research preview</strong>
            <span>
              Study design and reported finding are available. Unprovided measurements are shown as
              —.
            </span>
            <a href="#criteria">
              Read the finding <ArrowUpRight size={14} />
            </a>
          </aside>
        )}
        <Story />
        <DataSnapshot />
        <Vault />
        <Seeds />
        <Utility />
        <FidelityStudio />
        <PrivacyRadar />
        <Criteria />
        <Cost />
        <Lab />
        <Methods />
      </main>
      <Footer />
    </>
  );
}
