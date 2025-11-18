// @ts-check
const { test, expect, chromium } = require('@playwright/test');
const { spawn } = require('child_process');

const PORT = 8732;
const ORIGIN = `http://localhost:${PORT}`;

/** Wait for the local server to respond */
async function waitForServer(url, timeoutMs = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (res.ok) return true;
    } catch {}
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error('Server did not start in time');
}

let serverProc;

test.beforeAll(async () => {
  serverProc = spawn('python', ['-m', 'http.server', String(PORT)], {
    cwd: process.cwd(),
    stdio: 'ignore',
    shell: false,
  });
  await waitForServer(`${ORIGIN}/index.html`);
});

test.afterAll(async () => {
  try { serverProc.kill(); } catch {}
});

test('compare page loads and overlays traces for two stations', async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Navigate via index -> compare link
  await page.goto(`${ORIGIN}/index.html`, { waitUntil: 'domcontentloaded' });
  await page.locator('a[href="compare.html"]').click();
  await page.waitForURL(`**/compare.html`);

  // Trigger auto-load
  const autoBtn = page.locator('#autoReloadBtn');
  await expect(autoBtn).toBeVisible();
  await autoBtn.click();

  // Wait for status to show loaded rows (allow plenty of time for many CSVs)
  await expect(page.locator('#status')).toHaveText(/Loaded .* rows from .* CSV files\./, { timeout: 120000 });

  // Ensure station selects are populated
  const selA = page.locator('#stationSelectA');
  const selB = page.locator('#stationSelectB');
  await expect(selA).toHaveCount(1);
  await expect(selB).toHaveCount(1);
  const countA = await selA.locator('option').count();
  const countB = await selB.locator('option').count();
  expect(countA).toBeGreaterThan(1);
  expect(countB).toBeGreaterThan(1);

  // Pick first two distinct stations explicitly
  const aFirst = await selA.locator('option').nth(0).getAttribute('value');
  const bSecond = await selB.locator('option').nth(1).getAttribute('value');
  await selA.selectOption(aFirst);
  await selB.selectOption(bSecond);

  // Keep only Salinity on (reduce chance of empty variable)
  const wl = page.locator('#kind-water_level');
  const q = page.locator('#kind-discharge');
  const rain = page.locator('#kind-rain');
  if (await wl.isChecked()) await wl.uncheck();
  if (await q.isChecked()) await q.uncheck();
  if (await rain.isChecked()) await rain.uncheck();

  // Wait for a rendered legend containing A: or B:
  await page.waitForTimeout(1000);
  const traceCount = await page.evaluate(() => {
    const el = document.getElementById('chartC-salinity');
    const d = el && (el.data || el.__plotly?.data || el._fullData);
    if (Array.isArray(d)) return d.length;
    if (d && typeof d === 'object' && Array.isArray(d.data)) return d.data.length;
    return 0;
  });
  expect(traceCount).toBeGreaterThan(0);

  await browser.close();
});
