import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";
const executablePath =
  process.env.CHROMIUM_PATH || (existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined);
export default defineConfig({
  testDir: "./tests",
  use: {
    headless: true,
    launchOptions: { executablePath, args: ["--allow-file-access-from-files"] },
  },
});
