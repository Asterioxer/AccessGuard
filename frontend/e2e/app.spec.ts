import { test, expect } from '@playwright/test';

test.describe('AccessGuard End-to-End Workflow', () => {
  test('loads map dashboard and navigation tabs', async ({ page }) => {
    await page.goto('/');

    // Check title and branding
    await expect(page.locator('text=AccessGuard')).toBeVisible();
    await expect(page.locator('text=Public Benefit Intelligence')).toBeVisible();

    // Verify all primary tabs exist
    await expect(page.getByRole('button', { name: /Map Dashboard/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Accessible Routing/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Verification Queue/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Analytics/i })).toBeVisible();
  });

  test('navigates to Accessible Routing tab and displays route controls', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('button', { name: /Accessible Routing/i }).click();
    await expect(page.locator('text=Wheelchair Route Planner')).toBeVisible();
    await expect(page.locator('text=Origin Coordinates')).toBeVisible();
    await expect(page.locator('text=Destination Coordinates')).toBeVisible();
    await expect(page.getByRole('button', { name: /Calculate Accessible Route/i })).toBeVisible();
  });

  test('navigates to Verification Queue tab', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('button', { name: /Verification Queue/i }).click();
    await expect(page.locator('text=Human Verification Queue')).toBeVisible();
    await expect(page.locator('text=Community peer review')).toBeVisible();
  });

  test('navigates to Analytics tab', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('button', { name: /Analytics/i }).click();
    await expect(page.locator('text=City Accessibility Index')).toBeVisible();
    await expect(page.locator('text=Barrier Classification Breakdown')).toBeVisible();
    await expect(page.locator('text=Engineering Constraints & Privacy Safeguards')).toBeVisible();
  });

  test('opens upload observation modal', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('button', { name: /Upload Observation/i }).click();
    await expect(page.locator('text=Upload Sidewalk Observation')).toBeVisible();
    await expect(page.locator('text=Interchangeable CV Adapter')).toBeVisible();
    await expect(page.locator('text=Latitude')).toBeVisible();
    await expect(page.locator('text=Longitude')).toBeVisible();
  });
});
