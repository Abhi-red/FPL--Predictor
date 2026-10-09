const { test, expect } = require("playwright/test");
const fs = require("node:fs");
const path = require("node:path");
const root = path.resolve(__dirname, "../site");
const squad = JSON.parse(fs.readFileSync(path.join(root, "data/squad.json")));
async function open(page) {
  await page.route("https://dashboard.test/**", (route) => {
    const url = new URL(route.request().url());
    const file = path.join(
      root,
      url.pathname === "/" ? "index.html" : url.pathname,
    );
    const type = file.endsWith(".js")
      ? "application/javascript"
      : file.endsWith(".css")
        ? "text/css"
        : file.endsWith(".json")
          ? "application/json"
          : "text/html";
    return route.fulfill({ body: fs.readFileSync(file), contentType: type });
  });
  await page.goto("https://dashboard.test/");
}
test("overview preserves squad, totals and captain and exposes freshness", async ({
  page,
}) => {
  await open(page);
  await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.locator("#pitch .player-card")).toHaveCount(11);
  await expect(page.locator("#bench .player-card")).toHaveCount(4);
  await expect(page.locator("#projectedPoints")).toHaveText(
    squad.predicted_points.toFixed(1),
  );
  await expect(page.locator('#pitch [data-captain="true"]')).toContainText(
    squad.captain.web_name,
  );
  await expect(page.locator("#freshness")).toContainText(
    "Potentially outdated",
  );
});
test("search, club filters, sorting and keyboard details work", async ({
  page,
}) => {
  await open(page);
  await page.getByRole("tab", { name: "Players", exact: true }).click();
  await page.getByLabel("Search players").fill("Saka");
  const players = JSON.parse(
    fs.readFileSync(path.join(root, "data/players.json")),
  ).players;
  await expect(page.locator("#playersBody tr[data-id]")).toHaveCount(
    players.filter((p) =>
      (p.web_name + " " + p.team).toLowerCase().includes("saka"),
    ).length,
  );
  await page.getByRole("button", { name: "View Saka", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "View Saka", exact: true }),
  ).toBeFocused();
  await page.getByLabel("Search players").fill("no-such-player");
  await expect(page.locator("#playersBody")).toContainText("No players match");
  await page.getByRole("button", { name: "Clear filters" }).click();
  await page.getByLabel("Club", { exact: true }).selectOption("Arsenal");
  const clubs = await page
    .locator("#playersBody tr[data-id] td:nth-child(2)")
    .allTextContents();
  expect(clubs.length).toBeGreaterThan(0);
  expect(clubs.every((x) => x === "Arsenal")).toBeTruthy();
  await page.getByRole("button", { name: "Sort by Predicted points" }).click();
  await expect(page.locator('th[data-sort="adjusted_points"]')).toHaveAttribute(
    "aria-sort",
    "ascending",
  );
});
test("external text stays text and unsafe news links are omitted", async ({
  page,
}) => {
  await open(page);
  await page.route("**/data/players.json", async (route) => {
    const data = JSON.parse(
      fs.readFileSync(path.join(root, "data/players.json")),
    );
    data.players[0].web_name = "<img src=x onerror=alert(1)>";
    data.players[0].adjustment_reason = "<script>bad()</script>";
    data.players[0].news_url = "javascript:alert(1)";
    await route.fulfill({ json: data });
  });
  await page.reload();
  await page.getByRole("tab", { name: "Players", exact: true }).click();
  await page.getByLabel("Search players").fill("<img");
  await page.locator("#playersBody button").click();
  await expect(page.locator("#drawerBody img, #drawerBody script")).toHaveCount(
    0,
  );
  await expect(page.locator("#drawerBody a")).toHaveCount(0);
});
test("failed squad fetch leaves players usable", async ({ page }) => {
  await open(page);
  await page.route("**/data/squad.json", (route) =>
    route.fulfill({ status: 500, body: "error" }),
  );
  await page.reload();
  await expect(page.locator("#pitch")).toContainText("Squad unavailable");
  await page.getByRole("tab", { name: "Players", exact: true }).click();
  expect(
    await page.locator("#playersBody tr[data-id]").count(),
  ).toBeGreaterThan(100);
});
test("responsive layout and modal focus remain usable", async ({ page }) => {
  await open(page);
  for (const width of [375, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    const rows = await page
      .locator("#pitch .pitch-row")
      .evaluateAll((rows) =>
        rows.map((r) =>
          Array.from(r.children).map((c) => c.getBoundingClientRect().top),
        ),
      );
    expect(
      rows.every((row) => row.every((top) => Math.abs(top - row[0]) < 2)),
    ).toBeTruthy();
  }
  await page.emulateMedia({ reducedMotion: "reduce", colorScheme: "dark" });
  await page.locator("#pitch .player-card").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Close player details" }),
  ).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  expect(
    await page.evaluate(() =>
      document.querySelector("#drawer").contains(document.activeElement),
    ),
  ).toBeTruthy();
});

test("missing statistics, long names, zoom and timestamp boundary are readable", async ({
  page,
}) => {
  await open(page);
  await page.route("**/data/squad.json", (route) => {
    const data = structuredClone(squad);
    data.xi[0].web_name =
      "A very long player name that should remain accessible";
    return route.fulfill({ json: data });
  });
  await page.route("**/data/meta.json", (route) =>
    route.fulfill({
      json: {
        season: "2026-27",
        gameweek: 6,
        generated_at: new Date(Date.now() - 7 * 86400000).toISOString(),
      },
    }),
  );
  await page.reload();
  await expect(page.locator("#freshness")).toContainText("Weekly snapshot");
  await page.setViewportSize({ width: 375, height: 1000 });
  await page.evaluate(() => (document.documentElement.style.fontSize = "200%"));
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("tab", { name: "Players", exact: true }).click();
  await page.getByLabel("Search players").fill("Saka");
  await page.getByRole("button", { name: "View Saka", exact: true }).click();
  await expect(page.locator("#drawerBody")).not.toContainText("undefined");
  await expect(page.locator("#drawerBody")).not.toContainText("NaN");
});
test("missing explanation and players are handled without hiding the squad", async ({
  page,
}) => {
  await open(page);
  await page.route("**/data/players.json", (route) =>
    route.fulfill({ status: 500, body: "unavailable" }),
  );
  await page.route("**/data/explanation.json", (route) =>
    route.fulfill({ status: 500, body: "unavailable" }),
  );
  await page.reload();
  await expect(page.locator("#pitch .player-card")).toHaveCount(11);
  await page.locator("#pitch .player-card").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator("#drawerBody")).toContainText("—");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("tab", { name: "Players", exact: true }).click();
  await expect(page.locator("#playersBody")).toContainText(
    "Player predictions unavailable",
  );
  await page.getByRole("tab", { name: "Insights", exact: true }).click();
  await expect(page.locator("#explanation")).toContainText(
    "Gameweek insights unavailable",
  );
});

test("late data responses preserve modal trigger focus", async ({ page }) => {
  await open(page);
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  await page.route("**/data/players.json", async (route) => {
    await gate;
    await route.fulfill({
      json: JSON.parse(fs.readFileSync(path.join(root, "data/players.json"))),
    });
  });
  await page.reload();
  await expect(page.locator("#pitch .player-card")).toHaveCount(11);
  const trigger = page.locator("#pitch .player-card").first();
  const id = await trigger.getAttribute("data-id");
  await trigger.click();
  release();
  await expect(page.locator("#playerCount")).toContainText(
    String(
      JSON.parse(fs.readFileSync(path.join(root, "data/players.json"))).players
        .length,
    ),
  );
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(page.locator(`#pitch button[data-id="${id}"]`)).toBeFocused();
});
test("pending squad stays in a loading state while other data is usable", async ({
  page,
}) => {
  await open(page);
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  await page.route("**/data/squad.json", async (route) => {
    await gate;
    await route.fulfill({ json: squad });
  });
  await page.reload();
  await expect(page.locator("#playerCount")).toContainText("players");
  await expect(page.locator("#pitch")).toContainText("Loading squad");
  release();
  await expect(page.locator("#pitch .player-card")).toHaveCount(11);
});
