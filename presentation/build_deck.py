"""Build the editable viva slides from the repository's saved experiment outputs."""

import argparse
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
IDENTITY = json.loads((HERE / "identity.json").read_text(encoding="utf-8"))
W, H = 13 + 1 / 3, 7.5
FONT = "Calibri"
NAVY = "102747"
BLUE = IDENTITY["logo_blue"].lstrip("#")
RED = IDENTITY["logo_red"].lstrip("#")
INK = "20354D"
MUTED = "4F6279"
WHITE = "FFFFFF"
LINE = "D7E1EC"
LIGHT = "EFF4FA"
PINK = "FCEEF0"
TEAL = "166F6A"
GOLD = "9B671C"
SLATE = "5B6B82"
THRESHOLD = "6D7785"
RESULTS = "results/RESULTS.md"
FACTS = "reports/report_facts.json"
RUN_CONFIG = "results/run_config.yaml"
SUMMARY = "results/summary.csv"
SHOTS = "frontend/web/screenshots/top-1440.png"

GENS = ("CTGAN", "GaussianCopula")
GEN_NAMES = ("CTGAN", "Gaussian Copula")
GEN_COLORS = (RED, TEAL)
ARMS = (
    ("TRTR-full", "Full real", NAVY),
    ("TRTR-seed", "Real seed", BLUE),
    ("SMOTE-NC", "SMOTE-NC", GOLD),
    ("TSTR-CTGAN", "CTGAN", RED),
    ("TSTR-GaussianCopula", "Copula", TEAL),
)


def project_text(path):
    """Read a tracked source as UTF-8 text."""
    return (ROOT / path).read_text(encoding="utf-8")


def csv_records(path):
    """Read a result CSV when it exists with at least a header."""
    if not (ROOT / path).is_file():
        return []
    return list(csv.DictReader(io.StringIO(project_text(path))))


def load_utility(facts, sizes):
    """Prefer refreshed CSV results and fall back to the tracked JSON export."""
    saved = csv_records(SUMMARY)
    if saved:
        scores = {
            (row["arm"], int(row["seed_size"])): float(row["pr_auc_mean"])
            for row in saved if row["model"].lower() == "xgboost"
        }
        source = SUMMARY
    else:
        scores = {
            (arm, int(size)): row["pr_auc"]["mean"]
            for arm, arm_rows in facts["utility"]["xgboost"].items()
            for size, row in arm_rows.items()
        }
        source = FACTS
    for arm, _, _ in ARMS:
        keys = [0] if arm == "TRTR-full" else sizes
        for size in keys:
            if (arm, size) not in scores:
                raise ValueError(f"{source} has no XGBoost PR-AUC for {arm} at {size}.")
    return scores, source


def font(style, size, color, bold=False):
    """Set the shared presentation font and an accessible text color."""
    style.name = FONT
    style.size = Pt(size)
    style.bold = bold
    style.color.rgb = RGBColor.from_string(color)


def box(slide, x, y, w, h, fill, border=None, rounded=False):
    """Draw an editable accent, card or content block."""
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape._element.xpath("./p:style/a:effectRef")[0].set("idx", "0")
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    if border:
        shape.line.color.rgb = RGBColor.from_string(border)
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    if rounded:
        shape.adjustments[0] = 0.10
    return shape


def label(slide, x, y, w, h, text, size=20, color=INK, bold=False,
          align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    """Lay out explicit lines with Calibri and no automatic shrinkage."""
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Pt(0)
    frame.margin_top = frame.margin_bottom = Pt(0)
    frame.vertical_anchor = valign
    for i, line in enumerate(text.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.06
        p.space_after = Pt(1)
        run = p.add_run()
        run.text = line
        font(run.font, size, color, bold)
    return shape


def arrow(slide, x, y, width=0.38, down=False, color=SLATE):
    """Draw a native connector symbol between adjacent boxes."""
    kind = MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(0.18 if down else width),
                                   Inches(width if down else 0.14))
    shape._element.xpath("./p:style/a:effectRef")[0].set("idx", "0")
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(color)
    shape.line.fill.background()
    return shape


def contained_picture(slide, path, x, y, w, h):
    """Embed an undistorted image and center it within its allocated area."""
    with Image.open(path) as picture:
        pw, ph = picture.size
    scale = min(w / pw, h / ph)
    width, height = pw * scale, ph * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - width) / 2),
                                    Inches(y + (h - height) / 2),
                                    width=Inches(width), height=Inches(height))


def screenshot_path():
    """Use a genuine repository screenshot, verifying any hydrated LFS object."""
    original = ROOT / SHOTS
    cached = ASSETS / "react-top-1440.png"
    if not original.exists():
        return None
    data = original.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return original
    pointer = next((line.split("sha256:", 1)[-1] for line in data.decode().splitlines()
                    if line.startswith("oid sha256:")), None)
    if pointer and cached.exists() and hashlib.sha256(cached.read_bytes()).hexdigest() == pointer:
        return cached
    return None


def line_chart(slide, x, y, w, h, sizes, series, *, lo=0, hi=1, step=0.2,
               number_format="0.0"):
    """Build an editable line chart with its own embedded Excel workbook."""
    data = CategoryChartData()
    data.categories = [f"{size:,}" for size in sizes]
    for name, values, _, _ in series:
        data.add_series(name, values)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y),
                                   Inches(w), Inches(h), data).chart
    chart.has_legend = False
    chart.has_title = False
    for axis in (chart.category_axis, chart.value_axis):
        font(axis.tick_labels.font, 14, MUTED)
        axis.major_tick_mark = XL_TICK_MARK.NONE
        axis.minor_tick_mark = XL_TICK_MARK.NONE
    value_axis = chart.value_axis
    value_axis.minimum_scale, value_axis.maximum_scale = lo, hi
    value_axis.major_unit = step
    value_axis.tick_labels.number_format = number_format
    value_axis.has_major_gridlines = True
    value_axis.major_gridlines.format.line.color.rgb = RGBColor.from_string(LINE)
    for spec, line in zip(series, chart.series):
        _, _, color, dashed = spec
        line.format.line.color.rgb = RGBColor.from_string(color)
        line.format.line.width = Pt(2.1)
        if dashed:
            line.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        marker = line.marker
        marker.style = XL_MARKER_STYLE.NONE if dashed else XL_MARKER_STYLE.CIRCLE
        if not dashed:
            marker.size = 6
            marker.format.fill.solid()
            marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
            marker.format.line.color.rgb = RGBColor.from_string(color)
    return chart


def legend(slide, x, y, entries):
    """Write a consistent one-line legend shared across chart layouts."""
    for name, color, span in entries:
        box(slide, x, y + 0.12, 0.22, 0.045, color)
        label(slide, x + 0.31, y, span - 0.32, 0.32, name, 17, MUTED)
        x += span


def table(slide, x, y, widths, row_height, rows, *, level=18):
    """Use a native table with a high-contrast header and readable rows."""
    shape = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_height * len(rows)))
    tbl = shape.table
    tbl.first_row = tbl.horz_banding = False
    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_height)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(NAVY if i == 0 else (LIGHT if i % 2 else WHITE))
            cell.margin_left = cell.margin_right = Inches(0.16)
            cell.margin_top = cell.margin_bottom = Pt(2)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.space_after = Pt(0)
            run = p.add_run()
            run.text = str(value)
            color = WHITE if i == 0 else (RED if str(value).startswith("FAIL") else INK)
            font(run.font, level, color, i == 0 or j == 0)
    return tbl


class Deck:
    """Keep shared layout and the projected and spoken content in sync."""

    def __init__(self):
        """Create a blank widescreen presentation and an empty speaking order."""
        self.ppt = Presentation()
        self.ppt.slide_width, self.ppt.slide_height = Inches(W), Inches(H)
        self.ppt.core_properties.title = IDENTITY["title"]
        self.ppt.core_properties.subject = "Mini-project final evaluation"
        self.ppt.core_properties.author = ", ".join(person["name"] for person in IDENTITY["team"])
        self.records = []

    def slide(self, section, title, source, speaker, seconds):
        """Create an editorial header, source footer and visible number."""
        slide = self.ppt.slides.add_slide(self.ppt.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(WHITE)
        box(slide, 0.65, 0.40, 0.35, 0.055, RED)
        label(slide, 1.08, 0.28, 11.7, 0.35, section, 12, MUTED, True)
        if title:
            label(slide, 0.65, 0.78, 12.1, 0.89, title, 34, NAVY, True)
        box(slide, 0.65, 6.98, 12.03, 0.012, LINE)
        label(slide, 0.65, 7.10, 11.45, 0.26, f"Source: {source}", 10, MUTED)
        number = len(self.ppt.slides)
        label(slide, 12.06, 7.06, 0.62, 0.29, f"{number:02}", 12, NAVY, True, PP_ALIGN.RIGHT)
        self.records.append({"number": number, "title": title or IDENTITY["title"],
                             "speaker": speaker, "seconds": seconds})
        return slide

    def notes(self, slide, spoken, question, answer):
        """Record a timed script and one two-sentence viva answer."""
        count = len(spoken.split())
        if not 60 <= count <= 90:
            raise ValueError(f"Slide {len(self.records)} speaker script has {count} words.")
        current = self.records[-1]
        minutes, seconds = divmod(current["seconds"], 60)
        timing = f"{minutes}:{seconds:02}" if current["seconds"] else "Backup only"
        slide.notes_slide.notes_text_frame.text = (
            f"{current['speaker']} | {timing}\n\n{spoken}\n\n"
            f"Likely viva question: {question}\nAnswer: {answer}"
        )
        current.update(spoken=spoken, question=question, answer=answer)


def generator_series(rows, metric, sizes):
    """Align the two generator metrics to the configured seed order."""
    indexed = {(row["generator"], row["seed_size"]): row[metric] for row in rows}
    return [(name, [indexed[(gen, size)] for size in sizes], color, False)
            for gen, name, color in zip(GENS, GEN_NAMES, GEN_COLORS)]


def card(slide, x, y, w, h, heading, detail, accent=BLUE):
    """Use an editable, softly tinted block for a short argument."""
    box(slide, x, y, w, h, PINK if accent == RED else LIGHT, rounded=True)
    box(slide, x + 0.23, y + 0.22, 0.32, 0.055, accent)
    label(slide, x + 0.23, y + 0.46, w - 0.46, 0.42, heading, 23, NAVY, True)
    label(slide, x + 0.23, y + 1.02, w - 0.46, h - 1.08, detail, 19, INK)


def rule_result(row):
    """Summarize the failed frozen rules for one generator and one size."""
    keys = (("utility_ratio", "utility"), ("ks_mean", "KS"), ("tvd_mean", "TVD"),
            ("dcr_ratio", "DCR"), ("no_extra_exact_copies", "copies"))
    failed = [label for prefix, label in keys
              if any(key.startswith(prefix) and value == "FAIL" for key, value in row.items())]
    return "PASS" if row["ALL_PASS"] else "FAIL · " + " and ".join(failed)


def thresholds_from(facts):
    """Read criteria values from the results export and check them against the frozen YAML."""
    rules = {rule["id"]: rule["threshold"] for rule in facts["criteria"]["rules"]}
    criteria = yaml.safe_load(project_text("criteria.yaml"))
    for saved, configured in (("utility_ratio", "utility_ratio_min"), ("ks_mean", "ks_mean_max"),
                              ("tvd_mean", "tvd_mean_max"), ("dcr_ratio", "dcr_ratio_min")):
        if rules[saved] != criteria[configured]:
            raise ValueError(f"Saved run and criteria.yaml differ for {saved}.")
    return rules


def cover(deck):
    """Credit the team, course, mentor and institution."""
    slide = deck.slide("MINI-PROJECT  /  FINAL EVALUATION", None,
                       "Presentation brief and submitted course report", "Aryan", 30)
    box(slide, 0.66, 1.77, 0.10, 2.42, RED)
    label(slide, 0.97, 1.81, 9.58, 2.15, IDENTITY["title"], 48, NAVY, True)
    contained_picture(slide, ASSETS / "bennett-logo.png", 9.78, 0.68, 2.80, 1.12)
    panel_width = 5.78
    for index, person in enumerate(IDENTITY["team"]):
        x = 0.97 + index * 6
        box(slide, x, 4.43, panel_width, 0.92, LIGHT, rounded=True)
        label(slide, x + 0.22, 4.55, panel_width - 0.44, 0.36, person["name"], 22, NAVY, True)
        label(slide, x + 0.22, 4.95, panel_width - 0.44, 0.33, person["id"], 18, MUTED)
    label(slide, 0.97, 5.67, 11.9, 0.74,
          f'{IDENTITY["course"]}  ·  {IDENTITY["university"]}\nMentor  {IDENTITY["mentor"]}', 20, INK)
    deck.notes(slide,
               "Good morning. We are Aryan Gupta and Kathansh Jain, and this is our mini-project on "
               "machine learning with generated tabular data. We wanted to know how far a small real "
               "sample can take us when we use it to train a data generator. We will explain how we "
               "made the comparison fair, what each measurement means, and what we actually found on "
               "the held-out Adult Census Income data.",
               "Why use the word forged in the title?",
               "Here it means artificially generated tabular records. We use the records for a controlled machine-learning experiment.")


def problem(deck):
    """Introduce the bank analogy and what the actual study evaluates."""
    slide = deck.slide("MOTIVATION", "A team may receive only a small sample.",
                       "README.md · Project Overview", "Aryan", 35)
    xs = [0.65, 4.85, 9.05]
    details = [("Full table", "A bank holds the full table.\nSharing is restricted."),
               ("Small sample", "A team can use a limited\nset of real rows."),
               ("Artificial rows", "Can generated rows train\na useful model?")]
    for i, (x, (head, detail)) in enumerate(zip(xs, details)):
        card(slide, x, 2.36, 3.62, 2.42, head, detail, RED if i == 2 else BLUE)
        if i < len(xs) - 1:
            arrow(slide, x + 3.71, 3.53, 0.40)
    label(slide, 0.69, 5.35, 11.8, 0.43,
          "Bank scenario for illustration. Our experiment uses public census data.", 20, NAVY, True)
    deck.notes(slide,
               "Imagine a bank with a large table of customer records. Its analysts may receive "
               "only a sample because the full table cannot be shared freely. Could they use that "
               "sample to generate extra rows and train a useful classifier? That is a motivating "
               "example. Our actual data are public census records. The distance checks in our study "
               "measure record similarity, and we do not infer that generated data can always be shared.",
               "Did you use any bank customer records?",
               "We used the public Adult Census Income dataset. The bank scenario illustrates why a limited sample can matter.")


def goals(deck):
    """Keep resemblance, downstream learning and similarity risk distinct."""
    slide = deck.slide("RESEARCH QUESTION", "Useful synthetic data has to clear several checks.",
                       "PROJECT_BRIEF.md · Research question", "Kathansh", 45)
    label(slide, 0.67, 1.80, 11.85, 0.57,
          "How much real data does a generator need before its output is useful?", 23, INK)
    for i, (header, body) in enumerate([
        ("Fidelity", "Do the distributions\nlook like real data?"),
        ("Utility", "Can we train a classifier\nthat works on real rows?"),
        ("Privacy risk", "Are artificial rows\nunusually close to the seed?"),
    ]):
        card(slide, 0.65 + i * 4.05, 2.75, 3.88, 2.42, header, body, RED if i == 1 else BLUE)
    deck.notes(slide,
               "We changed the number of real training rows given to each generator and measured "
               "the output in three ways. Fidelity asks whether columns and relationships resemble "
               "real data. Utility asks whether a model trained on the generated rows can predict "
               "income on unseen real data. Privacy risk asks whether a generated row sits unusually "
               "close to the records used to train the generator. We required the checks to pass together.",
               "Why look beyond fidelity?",
               "A generator can match individual column distributions while missing combinations that predict income. The utility test checks whether those patterns transfer.")


def dataset_slide(deck, dataset):
    """Use measured dataset facts and an editable development-class figure."""
    slide = deck.slide("DATA", "Adult income is an imbalanced prediction task.",
                       "reports/report_facts.json · reports/eda_report.md · results/run_config.yaml",
                       "Kathansh", 50)
    label(slide, 0.67, 1.76, 7.0, 0.40, "Adult Census Income", 23, NAVY, True)
    label(slide, 0.66, 2.23, 7.15, 0.49,
          f'{dataset["rows_raw"]:,} raw rows  →  {dataset["rows_clean"]:,} after deduplication', 22, NAVY)
    label(slide, 0.68, 3.04, 7.24, 1.15,
          f'{dataset["n_features"]} predictors: age, education level, capital gain/loss, work hours, '
          'workclass, marital status, occupation, relationship, race, sex and native country.', 19, INK)
    config = yaml.safe_load(project_text(RUN_CONFIG))
    weight, education = config["dataset"]["drop_columns"]
    label(slide, 0.68, 4.60, 7.17, 0.96,
          f"{weight} is a survey weight. We kept the numeric education level and dropped {education}.", 19)
    x, y = 8.06, 1.86
    box(slide, x, y, 4.59, 4.62, LIGHT, rounded=True)
    label(slide, x + 0.27, y + 0.24, 4.05, 0.43, "Development class balance", 22, NAVY, True)
    positive = dataset["positive_rate_dev"]
    positive_label = dataset["positive_label"].replace(">", "> ")
    negative_label = "≤ " + positive_label.removeprefix("> ")
    # A two-segment class-balance figure keeps the EDA values directly editable.
    length = 3.98
    box(slide, x + 0.29, y + 2.20, length * (1 - positive), 0.42, BLUE)
    box(slide, x + 0.29 + length * (1 - positive), y + 2.20, length * positive, 0.42, RED)
    label(slide, x + 0.27, y + 0.87, 4.05, 0.78, f"{positive:.1%}", 49, RED, True)
    label(slide, x + 0.29, y + 1.64, 3.92, 0.40, f"{positive_label} annual income", 19, NAVY)
    legend(slide, x + 0.29, y + 2.94, [(f"{negative_label}  {(1 - positive):.1%}", BLUE, 2.16),
                                      (f"{positive_label}  {positive:.1%}", RED, 1.87)])
    deck.notes(slide,
               f"We used the public Adult Census Income dataset. It has {dataset['rows_raw']:,} raw rows "
               f"and {dataset['rows_clean']:,} after exact deduplication. We excluded the survey weight "
               "and kept the numeric education representation. Missing categories were recorded explicitly "
               f"and row IDs were excluded from modelling. About {positive:.0%} of development rows are "
               f"above the income cutoff, so the classes are uneven. That is why we chose average precision "
               "for our main utility comparison.",
               "Why drop the survey weight?",
               "It is an estimate of how many people each census row represents. The experiment models individual predictors.")


def experiment_design(deck, facts, config):
    """Draw nesting and show the disjoint, stratified evaluation split."""
    slide = deck.slide("METHOD", "Nested seeds make the comparison fairer.",
                       "reports/report_facts.json · results/run_config.yaml · CODE_WALKTHROUGH.md",
                       "Kathansh", 60)
    dataset = facts["dataset"]
    sizes = facts["seed_design"]["seed_sizes"]
    repeats = facts["seed_design"]["random_seeds"]
    test_share = config["split"]["test_size"]
    label(slide, 0.68, 1.67, 7.08, 0.37, "Each smaller sample stays inside the next.", 20, NAVY, True)
    for i, size in enumerate(reversed(sizes)):
        x, y = 0.73 + i * 0.37, 2.18 + i * 0.58
        depth = len(sizes) - i
        box(slide, x, y, 6.75 - i * 0.75, 0.61 + depth * 0.58,
            WHITE if i % 2 else LIGHT, BLUE if i < len(sizes) - 1 else RED, rounded=True)
        label(slide, x + 0.21, y + 0.10, 4.0, 0.39, f"{size:,} real rows", 18, NAVY, True)
    seeds = ", ".join(map(str, repeats))
    label(slide, 0.71, 5.94, 7.1, 0.49,
          f"{len(repeats)} repeats with random seeds {seeds}", 20, NAVY, True)
    box(slide, 8.03, 2.18, 4.59, 3.88, LIGHT, rounded=True)
    label(slide, 8.30, 2.45, 3.98, 0.43, "Stratified split", 24, NAVY, True)
    label(slide, 8.30, 3.13, 4.06, 0.76,
          f'{1 - test_share:.0%} development  ·  {dataset["dev_rows"]:,} rows\n'
          f'{test_share:.0%} locked test  ·  {dataset["test_rows"]:,} rows', 19, INK)
    box(slide, 8.30, 4.23, 3.97, 0.012, LINE)
    label(slide, 8.30, 4.57, 4.04, 0.86, "SHA-256 checks test-file integrity before scoring.", 21, NAVY, True)
    deck.notes(slide,
               f"We used an approximately {1 - test_share:.0%} development and {test_share:.0%} locked test "
               "split while keeping the income-class ratio close in both. The SHA-256 fingerprint verifies "
               "that the test file still has the same bytes before scoring. Inside development, every "
               "smaller seed is a subset of the next, with a similar class balance. Repeating the whole "
               f"sampling procedure {len(repeats)} times helps us see whether one sample ordering drove the result.",
               "Is the test file encrypted by SHA-256?",
               "SHA-256 verifies the integrity of the file bytes. Anyone with read access can still see the CSV contents.")


def methods(deck):
    """Contrast two generative models with a minority oversampling baseline."""
    slide = deck.slide("METHOD", "The methods create rows in different ways.",
                       "CODE_WALKTHROUGH.md · src/generators.py", "Aryan", 55)
    specs = [
        ("CTGAN", "A maker learns from a critic\nthat scores its synthetic rows.",
         ["Noise", "Maker", "Rows"], RED),
        ("Gaussian Copula", "It learns column distributions\nand how they vary together.",
         ["Cols", "Link", "Rows"], TEAL),
        ("SMOTE-NC", "It adds minority examples\nusing nearby real rows.",
         ["Seed", "Mix", "Balance"], BLUE),
    ]
    for i, (name, detail, nodes, color) in enumerate(specs):
        x = 0.65 + 4.05 * i
        card(slide, x, 2.02, 3.86, 3.83, name, detail, color)
        for j, node in enumerate(nodes):
            nx = x + 0.22 + j * 1.15
            box(slide, nx, 4.35, 0.94, 0.68, WHITE, LINE, rounded=True)
            label(slide, nx + 0.02, 4.51, 0.90, 0.34, node, 18, NAVY, True, PP_ALIGN.CENTER)
            if j < len(nodes) - 1:
                arrow(slide, nx + 0.97, 4.63, 0.15, color=color)
    deck.notes(slide,
               "CTGAN learns a table by asking a generator to make rows and a critic to score them. "
               "Conditioning gives rare categories more training attention. Gaussian Copula learns "
               "each column's distribution and an estimate of how columns vary together. SMOTE-NC "
               "has a different job. It uses nearest minority neighbours to make extra minority "
               "examples. We compare that balanced real-data arm with the models trained entirely on generated data.",
               "What is the role of SMOTE-NC in your comparison?",
               "It is a class-imbalance baseline that keeps the original seed rows. It adds minority examples to the seed.")


def workflow(deck, facts):
    """Use colored team lanes with a shared experiment sequence."""
    slide = deck.slide("ARCHITECTURE", "The two tracks meet in one experiment.",
                       "README.md · PROJECT_BRIEF.md · CODE_WALKTHROUGH.md", "Kathansh", 50)
    tracks = [
        ("Kathansh", "Data analysis", BLUE, LIGHT,
         ["Clean and split", "Build nested seeds", "Check fidelity and DCR"]),
        ("Aryan", "GenAI and ML", RED, PINK,
         ["Set model baselines", "Generate and balance", "Fit and score each arm"]),
    ]
    for i, (name, track, color, fill, steps) in enumerate(tracks):
        y = 2.06 + i * 1.55
        box(slide, 0.66, y, 2.14, 1.06, color, rounded=True)
        label(slide, 0.85, y + 0.13, 1.75, 0.39, name, 21, WHITE, True)
        label(slide, 0.85, y + 0.61, 1.75, 0.32, track, 16, WHITE)
        for j, step in enumerate(steps):
            x = 3.15 + j * 3.21
            box(slide, x, y, 2.98, 1.06, fill, rounded=True)
            label(slide, x + 0.10, y + 0.11, 2.78, 0.82,
                  step, 19, NAVY, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
            if j < len(steps) - 1:
                arrow(slide, x + 3.00, y + 0.46, 0.17, color=color)
    arrow(slide, 7.76, 3.13, 0.43, down=True, color=BLUE)
    label(slide, 0.69, 5.22, 11.91, 0.73,
          f'CTGAN and Copula each generate {facts["seed_design"]["synthetic_rows"]:,} rows per fitted seed.\n'
          "XGBoost leads the utility comparison, with Random Forest as a secondary model.", 20, NAVY)
    deck.notes(slide,
               "I worked on the data side, from cleaning and the split through nested seeds, "
               "distribution comparisons and the distance check. Aryan worked on the generators, "
               "classifiers, experiment runner and application. Both tracks use the same development "
               "data. Each arm fits its own preprocessing and classifier. We used Random Forest and "
               "XGBoost, and the final decision uses XGBoost average precision because a single, "
               "consistent evaluator makes the training-source comparison easier to read.",
               "Where do the locked rows enter the pipeline?",
               "They enter only when the fitted classifiers make final evaluation predictions. Fitting, seed selection and generator diagnostics use development data.")


METRICS = (
    ("KS", "Largest gap between numerical distributions", "maxₓ |Freal(x) − Fsynth(x)|"),
    ("TVD", "Difference in category shares", "½ Σc |p(c) − q(c)|"),
    ("Correlation difference", "Change in numerical relationships", "mean |rreal − rsynth| across pairs"),
    ("DCR", "Distance to the nearest seed row", "min over s in seed: ‖x − s‖₂"),
    ("PR-AUC / AP", "Precision–recall performance", "Σn (Rn − Rn−1) × Pn"),
)


def metrics_slide(deck):
    """Give the plain meaning and formula for each headline metric."""
    slide = deck.slide("MEASURES", "Each metric asks a different question.",
                       "src/fidelity.py · src/privacy.py · src/models.py", "Kathansh", 50)
    table(slide, 0.66, 1.87, [2.93, 5.14, 3.95], 0.65,
          [("Metric", "What it measures", "Formula"), *METRICS])
    deck.notes(slide,
               "For numerical columns, KS is the largest vertical gap between the real and synthetic "
               "cumulative curves. TVD compares categorical shares. Correlation difference looks at "
               "relationships between numeric columns. DCR is the distance to the closest seed row "
               "after encoding and scaling. For utility, our PR-AUC is average precision. It adds "
               "precision weighted by the gain in recall. That emphasizes the ability to retrieve "
               "the smaller positive income class.",
               "Which direction is better for the metrics?",
               "Lower KS, TVD and correlation difference mean a closer match. Higher average precision is better, while DCR needs a genuine holdout reference.")


def utility_slide(deck, sizes, scores, source):
    """Plot all XGBoost training arms and state the measured benchmark gap."""
    baseline = scores[("TRTR-full", 0)]
    best = max(scores[(arm, size)] for arm in ("TSTR-CTGAN", "TSTR-GaussianCopula") for size in sizes)
    real_wins = all(scores[("TRTR-seed", size)] >
                    max(scores[("TSTR-CTGAN", size)], scores[("TSTR-GaussianCopula", size)])
                    for size in sizes)
    title = ("Training on the real seed gave better utility." if real_wins
             else "Synthetic utility depends on the training size.")
    slide = deck.slide("RESULTS  /  UTILITY", title, f"{source} · {RESULTS}", "Aryan", 65)
    series = [(name, [baseline] * len(sizes) if i == 0 else [scores[(arm, n)] for n in sizes],
               color, i == 0) for i, (arm, name, color) in enumerate(ARMS)]
    legend(slide, 0.70, 1.66, [(name, color, 2.29) for _, name, color in ARMS])
    line_chart(slide, 0.68, 2.08, 12.0, 3.65, sizes, series)
    label(slide, 0.72, 5.75, 11.9, 0.36, "Real seed size (rows)  ·  XGBoost average precision", 18, MUTED)
    label(slide, 0.72, 6.18, 11.80, 0.60,
          f"Full-real reference {baseline:.3f}. Best synthetic-only score {best:.3f}.\n"
          "Saved TSTR results predate the missing-value repair.", 19, NAVY, True)
    direction = ("The real seed beat both synthetic-only arms at every tested size. " if real_wins
                 else "The real-seed benchmark helps us see when generation changes utility. ")
    deck.notes(slide,
               "On the horizontal axis is the number of real rows given to each seed-based arm. The "
               f"vertical axis is XGBoost average precision. The dashed full-real reference is {baseline:.3f}, "
               f"and the best synthetic-only score was {best:.3f}. {direction}The README says the saved "
               "synthetic data predates rare-category missing-value repair, so these TSTR scores may be "
               "slightly pessimistic. We report the saved measurements and do not estimate the effect of the fix.",
               "What do TRTR and TSTR stand for?",
               "TRTR means train on real data and test on real data. TSTR means train on synthetic data and test on real data.")


def fidelity_slide(deck, sizes, facts, rules):
    """Show numerical and categorical fidelity with matching axes and rule lines."""
    rows = facts["fidelity"]
    ks_series = generator_series(rows, "ks_mean_mean", sizes)
    tvd_series = generator_series(rows, "tvd_mean_mean", sizes)
    ctgan_ks, copula_ks = ks_series[0][1], ks_series[1][1]
    ctgan_tvd, copula_tvd = tvd_series[0][1], tvd_series[1][1]
    pattern_holds = all(a < b for a, b in zip(ctgan_ks, copula_ks)) and all(
        a > b for a, b in zip(ctgan_tvd, copula_tvd))
    title = ("CTGAN led on KS. Copula led on TVD." if pattern_holds
             else "Numerical and categorical fidelity tell different stories.")
    slide = deck.slide("RESULTS  /  FIDELITY", title, f"{FACTS} · {RESULTS}", "Kathansh", 50)
    legend(slide, 0.71, 1.79, [("CTGAN", RED, 1.30), ("Gaussian Copula", TEAL, 2.30),
                               ("Frozen limit", THRESHOLD, 1.95)])
    for x, name, series, metric, ceiling, step in [
        (0.68, "Mean numerical KS", ks_series, "ks_mean", 0.6, 0.1),
        (6.80, "Mean categorical TVD", tvd_series, "tvd_mean", 0.12, 0.03),
    ]:
        label(slide, x + 0.11, 2.31, 5.80, 0.37, name, 20, NAVY, True)
        series = series + [("Frozen limit", [rules[metric]] * len(sizes), THRESHOLD, True)]
        line_chart(slide, x, 2.76, 5.82, 2.90, sizes, series, hi=max(ceiling, max(v for _, vals, _, _ in series for v in vals)),
                   step=step, number_format="0.00")
    label(slide, 0.78, 5.77, 11.75, 0.33,
          "Both charts use real seed size (rows). Lower values mean closer distributions.", 18, MUTED)
    label(slide, 0.78, 6.25, 11.76, 0.43,
          f"CTGAN mean KS {min(ctgan_ks):.2f}–{max(ctgan_ks):.2f} against a limit of {rules['ks_mean']:.2f}. "
          f"Best Copula TVD {min(copula_tvd):.3f}.", 20, NAVY, True)
    deck.notes(slide,
               "The left chart compares numerical distributions. CTGAN's lowest mean KS is "
               f"{min(ctgan_ks):.3f}, against the frozen limit of {rules['ks_mean']:.2f}. The right chart "
               f"compares categorical shares. The lowest Copula TVD is {min(copula_tvd):.3f}. The two "
               "metrics look at different properties. An accurate category distribution can coexist "
               "with inaccurate numeric distributions or loss of combinations that help an income "
               "classifier. The plots show mean scores at each tested seed size.",
               "Why can Copula match category shares yet have lower utility?",
               "Category shares describe columns one at a time. An income model can need feature combinations and numerical distributions that those shares miss.")


def privacy_slide(deck, sizes, facts, rules):
    """Show mean nearest-record distance ratios and the limits of the diagnostic."""
    curves = generator_series(facts["privacy"], "dcr_ratio_mean", sizes)
    floor = min(value for _, values, _, _ in curves for value in values)
    failure = any(row["no_extra_exact_copies"] == "FAIL" for row in facts["criteria"]["evaluations"])
    warning = floor < rules["dcr_ratio"] or failure
    title = ("The distance check flagged a memorisation warning." if warning
             else "The distance check raised no memorisation warning.")
    slide = deck.slide("RESULTS  /  PRIVACY RISK", title, f"{FACTS} · {RESULTS}", "Kathansh", 50)
    legend(slide, 0.70, 1.75, [("CTGAN", RED, 1.32), ("Gaussian Copula", TEAL, 2.33),
                               ("Frozen threshold", THRESHOLD, 2.31)])
    line_chart(slide, 0.69, 2.19, 8.22, 3.88, sizes,
               curves + [("Frozen threshold", [rules["dcr_ratio"]] * len(sizes), THRESHOLD, True)],
               hi=max(3.2, max(value for _, values, _, _ in curves for value in values)), step=0.8)
    label(slide, 0.80, 6.19, 8.04, 0.39, "Real seed size (rows)  ·  Mean DCR ratio", 18, MUTED)
    box(slide, 9.27, 2.29, 3.40, 3.90, LIGHT, rounded=True)
    label(slide, 9.55, 2.54, 2.88, 0.83, f"{floor:.3f}", 44, NAVY, True)
    label(slide, 9.55, 3.32, 2.87, 0.80,
          f"Lowest mean ratio\nFrozen floor {rules['dcr_ratio']:.2f}", 19, INK)
    label(slide, 9.55, 4.31, 2.81, 1.39,
          "DCR checks closeness. Formal privacy needs separate evidence.", 21, NAVY, True)
    deck.notes(slide,
               "For every generated row, we measure the distance to its closest seed record. We also "
               "measure genuine holdout rows against that same seed. We divide the generated median "
               f"distance by the holdout median. The smallest reported mean ratio is {floor:.3f}. Our "
               "copy-rate rule is another check on near duplicates. These tests check one kind of "
               "record similarity. They cannot tell us whether any specific privacy attack would succeed.",
               "Can these DCR results certify that you may share synthetic records?",
               "DCR measures nearest-record similarity under a chosen representation. A privacy guarantee requires a separate formal mechanism and analysis.")


def minimum_slide(deck, sizes, facts, rules):
    """Display the archived pass/fail table and give the decision in plain words."""
    records = facts["criteria"]["evaluations"]
    by_key = {(row["generator"], row["seed_size"]): row for row in records}
    winner = facts["criteria"]["min_useful_seed"]
    best = max(row["utility_ratio"] for row in records)
    slide = deck.slide("RESULTS  /  DECISION",
                       "None of the tested sizes passed." if winner is None else f"The minimum useful seed was {winner:,} rows.",
                       f"criteria.yaml · {FACTS} · {RESULTS}", "Aryan", 55)
    table(slide, 0.66, 1.77, [2.30, 4.85, 4.85], 0.61,
          [("Real seed", "CTGAN", "Gaussian Copula"),
           *[(f"{n:,}", *(rule_result(by_key[(gen, n)]) for gen in GENS)) for n in sizes]])
    label(slide, 0.72, 5.63, 11.91, 0.44,
          f"Best utility ratio {best:.3f} versus {rules['utility_ratio']:.2f} required.", 22, NAVY, True)
    passed_other = all(row["no_extra_exact_copies"] == "PASS" and
                       all(value == "PASS" for key, value in row.items()
                           if key.startswith(("tvd_mean<=", "dcr_ratio>="))) for row in records)
    detail = ("Every configuration met the TVD, DCR and copy-rate rules." if passed_other
              else "The pass rules were frozen before scoring the locked test set.")
    label(slide, 0.72, 6.14, 11.91, 0.44, detail, 19, MUTED)
    selection = ("None of the sizes cleared all the checks. In this run, the blocking rules were "
                 "utility and numerical KS. " if winner is None
                 else f"The smallest size that cleared all criteria was {winner:,} rows. ")
    deck.notes(slide,
               "A useful seed had to clear every frozen rule, including utility, numerical and "
               "categorical fidelity, and both record-similarity checks. "
               f"{selection}The best synthetic utility ratio was {best:.3f}, while the "
               f"required ratio was {rules['utility_ratio']:.2f}. We kept the thresholds declared before "
               "test scoring. This decision answers the question for the sizes and methods we tested "
               "and gives us a clear starting point for another experiment.",
               "Why keep the original thresholds when the candidates fall short?",
               "Changing the threshold after scoring would let the test outcomes drive the rules. A future experiment can specify different thresholds in advance.")


def software_slide(deck):
    """Show the actual web UI image and how React, FastAPI and Python connect."""
    screenshot = screenshot_path()
    source = f"{SHOTS} · README.md · API contract" if screenshot else "README.md · api/openapi.json"
    slide = deck.slide("SOFTWARE", "The web app makes the experiment explorable.",
                       source, "Aryan", 45)
    stack_x = 8.51 if screenshot else 3.58
    width = 4.13 if screenshot else 6.17
    if screenshot:
        box(slide, 0.65, 1.73, 7.53, 4.83, LIGHT, LINE, rounded=True)
        contained_picture(slide, screenshot, 0.76, 1.91, 7.31, 4.27)
        label(slide, 0.86, 6.22, 7.15, 0.32, "Repository screenshot of the React interface", 17, MUTED)
    for i, (head, detail) in enumerate([
        ("React web app", "Study, results and CSV lab"),
        ("FastAPI", "Typed routes and lab jobs"),
        ("Python pipeline", "Saved outputs and generators"),
    ]):
        y = 2.03 + 1.42 * i
        box(slide, stack_x, y, width, 0.98, LIGHT if i != 1 else PINK, rounded=True)
        label(slide, stack_x + 0.20, y + 0.11, width - 0.40, 0.36, head, 22, NAVY, True)
        label(slide, stack_x + 0.20, y + 0.55, width - 0.40, 0.36, detail, 18, INK)
        if i < 2:
            arrow(slide, stack_x + width / 2 - 0.09, y + 1.04, 0.31, down=True)
    deck.notes(slide,
               "The React front end presents the story, figures and an interactive CSV lab. FastAPI "
               "connects those views to saved experiment outputs and wraps the Python generator "
               "functions for lab jobs. The app can browse results without rerunning the full "
               "experiment during a demonstration. This interface is a separate layer from the "
               "research pipeline. The chart data in these slides come from the saved research "
               "outputs, so they can be checked independently of the app.",
               "Do you need to run the full experiment to show the web app?",
               "The presentation can use the saved results and screenshots. A live lab needs the connected backend and appropriate input data.")


def closing_slide(deck):
    """Close with the scope, caveat and careful interpretation."""
    slide = deck.slide("CONCLUSION", "The result is specific to this experiment.",
                       "README.md · VIVA_ANSWERS.md · Project storytelling and viva guide", "Kathansh", 50)
    card(slide, 0.67, 1.83, 5.85, 2.97, "Limits",
         "One census dataset and a fixed training budget.\nSaved TSTR scores predate NaN repair.", RED)
    card(slide, 6.78, 1.83, 5.87, 2.97, "Next",
         "Try other datasets and budgets.\nEvaluate privacy against defined attacks.", BLUE)
    box(slide, 0.67, 5.17, 11.98, 1.14, NAVY, rounded=True)
    label(slide, 0.96, 5.40, 11.38, 0.67,
          "Our contribution is a controlled comparison of fidelity, utility and nearest-record risk.", 23, WHITE, True)
    deck.notes(slide,
               "Our project does not claim that synthetic data is automatically private or that it can "
               "replace real data in every setting. We evaluated a specific tabular dataset and a defined "
               "set of generators. Our contribution is a controlled comparison of how seed size affects "
               "statistical fidelity, downstream predictive utility, and a nearest-record memorisation "
               "indicator. We reported the results honestly. None of the tested sizes satisfied all "
               "the predefined criteria.",
               "What would you test next?",
               "I would rerun the fixed pipeline on a separately defined evaluation protocol and vary the training budget as its own factor. I would also test record disclosure under a defined threat model.")


def formulas_backup(deck, rules):
    """Keep operational formulas and the frozen thresholds at hand."""
    slide = deck.slide("BACKUP B1  /  FORMULAS", "The frozen rules use measurable quantities.",
                       "criteria.yaml · src/fidelity.py · src/privacy.py · src/models.py",
                       "Kathansh", 0)
    table(slide, 0.67, 1.79, [2.87, 6.14, 2.98], 0.65, [
        ("Quantity", "Definition", "Rule"),
        ("Utility ratio", "Mean AP from TSTR ÷ mean AP from TRTR-full", f"≥ {rules['utility_ratio']:.2f}"),
        ("Mean KS", "Average max |Freal(x) − Fsynth(x)| across numeric columns", f"≤ {rules['ks_mean']:.2f}"),
        ("Mean TVD", "Average ½ Σc |p(c) − q(c)| across categorical columns", f"≤ {rules['tvd_mean']:.2f}"),
        ("DCR ratio", "Median synthetic DCR ÷ median holdout DCR", f"≥ {rules['dcr_ratio']:.2f}"),
        ("Exact-match rate", "Fraction at near-zero nearest-seed distance", "Synthetic ≤ holdout"),
        ("Average precision", "Σn (Rn − Rn−1) × Pn across score thresholds", "Higher is better"),
    ], level=18)
    deck.notes(slide,
               "The criteria file specifies what must pass together. The utility ratio compares a synthetic-only "
               "classifier with the full-real reference on the same test. We average KS across numeric columns "
               "and TVD across categorical columns. The distance ratio uses medians, while the exact-copy rule "
               "compares near-zero distances with the holdout. Average precision is the summary of the "
               "precision-recall curve used in the utility ratio.",
               "Why use a holdout reference for copies?",
               "Genuine records can share identical measured predictors. The holdout gives us a comparison rate for these natural matches.")


def column_backup(deck, dataset):
    """Prefer measured per-column results and otherwise mark the gap honestly."""
    path = "results/fidelity_per_column.csv"
    records = csv_records(path)
    if records:
        from collections import defaultdict
        groups = defaultdict(list)
        for row in records:
            for metric, display in (("ks_statistic", "KS"), ("tvd", "TVD")):
                try:
                    value = float(row[metric])
                except (KeyError, ValueError, TypeError):
                    continue
                if value == value:
                    groups[(row["column"], display)].append(value)
        rows = [(column, metric, sum(values) / len(values)) for (column, metric), values in groups.items()]
        rows = sorted(rows, key=lambda row: row[2], reverse=True)[:5]
    else:
        rows = []
    source = path if rows else "reports/eda_report.md · reports/report_facts.json"
    slide = deck.slide("BACKUP B2  /  PER-COLUMN FIDELITY", "Some columns are harder to reproduce." if rows
                       else "EDA flagged skew and rare categories.",
                       source, "Kathansh", 0)
    if rows:
        table(slide, 0.65, 1.84, [3.22, 3.38, 5.40], 0.71,
              [("Column", "Metric", "Mean across archived configurations"),
               *[(column, metric, f"{value:.3f}") for column, metric, value in rows]])
        detail = "The table ranks the recorded per-column differences by mean magnitude. "
    else:
        missing = {row["name"]: row["missing_pct"] for row in dataset["features"] if row["missing_pct"]}
        table(slide, 0.65, 1.84, [3.48, 8.52], 0.73, [
            ("Column(s)", "Evidence from development EDA"),
            ("capital-gain and capital-loss", "Concentrated at zero, with a long upper tail"),
            ("native-country", f"Rare levels and {missing['native-country']:.2f}% missing"),
            ("occupation and workclass",
             f"{missing['occupation']:.2f}% and {missing['workclass']:.2f}% missing, respectively"),
        ])
        label(slide, 0.70, 5.20, 11.90, 0.46, "The checkout has aggregate fidelity and EDA, with per-column scores unavailable.",
              19, NAVY, True)
        detail = "The available per-feature evidence comes from development EDA. We would need the archived per-column scores to rank synthetic differences. "
    deck.notes(slide,
               "A single mean can hide a difficult feature. In Adult, capital gain and capital loss have "
               "many zeros and a small number of much larger values. Native country includes rare levels, "
               "and some work and occupation entries are missing in the real data. "
               f"{detail}A feature-level check can help distinguish problems in rare groups from problems "
               "that affect the whole numerical distribution.",
               "Which features would you inspect first after a refresh?",
               "I would compare the empirical curves for capital gain and loss, then compare rare-category shares. Those are clear places where small seeds can leave gaps.")


def runtime_backup(deck, sizes, facts, config):
    """Show actual fit times and the equal-iteration CTGAN settings."""
    slide = deck.slide("BACKUP B3  /  COMPUTE", "CTGAN took longer under a fixed step budget.",
                       f"{FACTS} · {RUN_CONFIG} · src/generators.py", "Aryan", 0)
    curves = generator_series(facts["runtime"], "fit_seconds_mean", sizes)
    min_ct, max_ct = min(curves[0][1]), max(curves[0][1])
    min_gc, max_gc = min(curves[1][1]), max(curves[1][1])
    ctgan = config["generators"]["ctgan"]
    rows = facts["seed_design"]["synthetic_rows"]
    legend(slide, 0.70, 1.77, [("CTGAN", RED, 1.33), ("Gaussian Copula", TEAL, 2.43)])
    line_chart(slide, 0.68, 2.17, 8.22, 3.73, sizes, curves,
               hi=max(200, max(max_ct, max_gc)), step=50, number_format="0")
    label(slide, 0.78, 6.04, 8.05, 0.43, "Real seed size (rows)  ·  Mean fit seconds", 18, MUTED)
    box(slide, 9.21, 2.24, 3.46, 3.75, LIGHT, rounded=True)
    label(slide, 9.44, 2.44, 2.99, 0.49, "CTGAN settings", 22, NAVY, True)
    label(slide, 9.44, 3.13, 2.98, 1.49,
          f'{ctgan["total_training_steps"]:,} steps per fit\n'
          f'{ctgan["batch_size"]:,} rows per batch\n'
          f"{rows:,} sampled rows\n{'CPU' if not ctgan['use_gpu'] else 'GPU'} training", 19)
    deck.notes(slide,
               f"Mean fit time for CTGAN ranged from {min_ct:.1f} to {max_ct:.1f} seconds. Gaussian "
               f"Copula took {min_gc:.1f} to {max_gc:.1f} seconds. The CTGAN budget was "
               f"{ctgan['total_training_steps']:,} iterations per fit, with a batch size of "
               f"{ctgan['batch_size']:,}. The code converts that budget to a different number of epochs "
               "when the number of steps per epoch changes. This keeps compute more comparable when "
               "we vary the real seed size. The chart reports fitting time, with sampling stored separately.",
               "Why use steps per fit instead of epochs per fit?",
               "Different seed sizes can require different numbers of batches per epoch. A fixed step budget keeps the number of training updates comparable.")


def write_notes(deck, notes_path, source, facts, rules, utility):
    """Write the timed speaker handoff, full scripts and source discrepancies."""
    main = [row for row in deck.records if row["seconds"]]
    total = sum(row["seconds"] for row in main)
    if not 10 * 60 <= total <= 12 * 60:
        raise ValueError(f"Speaking order is {total} seconds.")
    full = utility[("TRTR-full", 0)]
    ct_ks = [row["ks_mean_mean"] for row in facts["fidelity"] if row["generator"] == "CTGAN"]
    ct_best = max(row["utility_ratio"] for row in facts["criteria"]["evaluations"])
    copula_ks = [row["ks_mean_mean"] for row in facts["fidelity"] if row["generator"] == "GaussianCopula"]
    dcr_min = min(row["dcr_ratio_mean"] for row in facts["privacy"])
    none_passed = facts["criteria"]["min_useful_seed"] is None
    elapsed = 0
    rows = ["# Final evaluation speaking order", "",
            f"**Total planned time {total // 60}:{total % 60:02}.** "
            "Use the main deck in order. Leave the three backup slides for questions.", "",
            "| Starts | Slide | Speaker | Topic |",
            "| --- | --- | --- | --- |"]
    for row in main:
        rows.append(f"| {elapsed // 60}:{elapsed % 60:02} | {row['number']} | {row['speaker']} | {row['title']} |")
        elapsed += row["seconds"]
    rows.extend(["", "## Source checks", "",
                 f"- Utility chart source is `{source}`. "
                 f"The full-real XGBoost reference is **{full:.4f}** ({full:.3f} on the slide).",
                 f"- The best archived utility ratio is **{ct_best:.3f}** against **{rules['utility_ratio']:.2f}**.",
                 f"- The README’s KS range refers to CTGAN. Its archived values span "
                 f"**{min(ct_ks):.4f}–{max(ct_ks):.4f}** against **{rules['ks_mean']:.2f}**. "
                 f"Gaussian Copula ranges from **{min(copula_ks):.4f} to {max(copula_ks):.4f}**.",
                 f"- The lowest reported mean DCR ratio is **{dcr_min:.4f}**. "
                 "The checkout contains aggregate ratios, so the claim is about reported means.",
                 f"- Frozen outcome is **{'none passed' if none_passed else 'a passing size exists'}**.",
                 "- The GitHub checkout contains React and FastAPI. The older project brief calls the UI Streamlit.",
                 "- The repository has no figures directory in this checkout. The class balance figure uses the tracked EDA counts.",
                 "- The web screenshot is the original repository image retrieved from Git LFS and matched to its SHA-256 object ID.",
                 "- Per-column fidelity is described from development EDA when the measured per-column CSV is absent.",
                 "- The reported TSTR results predate the missing-value repair. The README describes them as slightly pessimistic.",
                 "", "## Scripts and viva prompts", ""])
    for row in deck.records:
        rows.extend([f"### {row['number']:02}. {row['title']}", "",
                     f"**{row['speaker']}**", "", row["spoken"], "",
                     f"**Likely viva question**  {row['question']}", "",
                     row["answer"], ""])
    notes_path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")


def main():
    """Build the deck without editing source data, then optionally copy it to a Desktop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", type=Path, nargs="?", const=Path.home() / "Desktop",
                        help="Copy the completed PPTX to an existing Desktop directory.")
    args = parser.parse_args()
    facts = json.loads(project_text(FACTS))
    config = yaml.safe_load(project_text(RUN_CONFIG))
    rules = thresholds_from(facts)
    sizes = facts["seed_design"]["seed_sizes"]
    scores, source = load_utility(facts, sizes)
    deck = Deck()
    cover(deck)
    problem(deck)
    goals(deck)
    dataset_slide(deck, facts["dataset"])
    experiment_design(deck, facts, config)
    methods(deck)
    workflow(deck, facts)
    metrics_slide(deck)
    utility_slide(deck, sizes, scores, source)
    fidelity_slide(deck, sizes, facts, rules)
    privacy_slide(deck, sizes, facts, rules)
    minimum_slide(deck, sizes, facts, rules)
    software_slide(deck)
    closing_slide(deck)
    formulas_backup(deck, rules)
    column_backup(deck, facts["dataset"])
    runtime_backup(deck, sizes, facts, config)

    # Main slide times include the live chart walkthroughs and speaker handoffs.
    output = HERE / "Synthetic_Data_Project.pptx"
    deck.ppt.save(output)
    write_notes(deck, HERE / "SLIDE_NOTES.md", source, facts, rules, scores)
    print(f"Wrote {output} ({len(deck.ppt.slides)} slides). Utility data: {source}.")
    if args.desktop:
        if args.desktop.is_dir():
            shutil.copy2(output, args.desktop / output.name)
        else:
            print(f"Desktop directory is unavailable: {args.desktop}")


if __name__ == "__main__":
    main()
