// npm install, then npm run test:dashboard. Windows uses installed Edge.
// Other platforms: npx playwright install chromium (or set FPL_BROWSER_CHANNEL).
module.exports = {
  testDir: "./tests",
  testMatch: "dashboard.spec.cjs",
  timeout: 20000,
  use: {
    browserName: "chromium",
    channel:
      process.env.FPL_BROWSER_CHANNEL ||
      (process.platform === "win32" ? "msedge" : undefined),
    headless: true,
  },
  reporter: "list",
  outputDir: "test-results",
};
