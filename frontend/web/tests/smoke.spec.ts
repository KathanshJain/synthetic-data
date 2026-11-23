import { test, expect } from "@playwright/test";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
const base = process.env.PLAYWRIGHT_BASE_URL || pathToFileURL(resolve("dist/index.html")).href;
test("research sections render in mock mode", async ({ page }) => {
  await page.goto(base);
  for (const heading of [
    "Real questions.",
    "Good questions deserve a good process.",
    "It starts with the real thing.",
    "Some things are better",
    "A little data goes a long way. Or does it?",
    "Do the models actually learn?",
    "Looks similar. But how similar?",
    "Similar shouldn't",
    "A result worth finding.",
    "Every experiment has a cost.",
    "Curious? Make it your own.",
    "Good science",
  ])
    await expect(page.getByRole("heading", { name: heading, exact: false }).first()).toBeVisible();
});

test("mock mode discloses its data limits, opens a drawer, and profiles a local CSV", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(base);
  await expect(page.getByText("No tested seed size passed every frozen rule.")).toBeVisible();
  await page.getByRole("button", { name: "Why?", exact: true }).click();
  await expect(page.getByText(/0 of 10 generator/)).toBeVisible();
  await page
    .locator("#data")
    .getByRole("button", { name: /^Rows / })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await page
    .locator("#lab input[type=file]")
    .setInputFiles({
      name: "local-test.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("customer_id,target,value\nA,yes,1\nB,no,\nC,yes,3"),
    });
  await expect(page.getByText(/looks like an identifier/)).toBeVisible();
  await page.getByLabel("Target column").selectOption("target");
  await expect(page.locator("#lab canvas").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Run experiment" })).toBeDisabled();
  expect(errors).toEqual([]);
});

for (const width of [375, 1440])
  test(`capture sections at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(base);
    for (const id of [
      "top",
      "story",
      "data",
      "vault",
      "seeds",
      "utility",
      "fidelity",
      "privacy",
      "criteria",
      "cost",
      "lab",
      "methods",
    ]) {
      const section = page.locator(`#${id}`);
      await section.evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
      await section.locator("h1,h2").first().waitFor({ state: "visible" });
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
      await section.screenshot({ path: `screenshots/${id}-${width}.png`, animations: "disabled" });
    }
  });
