# Repository contribution rules

## Commit messages

- Use Conventional Commits: `<type>(optional-scope): <imperative summary>`.
- Use one of these prefixes unless another established type is more precise:
  `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
  or `revert`.
- Keep the title concise (preferably 72 characters or fewer), imperative, and
  without a trailing period.
- Always add a commit body after a blank line. Explain what changed and why;
  include important design choices, compatibility notes, or test evidence when
  useful.
- Mark breaking changes with `!` in the prefix and add a `BREAKING CHANGE:`
  footer describing the migration impact.
- Keep each commit focused on one coherent change and do not mix unrelated
  formatting or cleanup.

Example:

```text
feat(mesh): add geodesic grid generation

Generate an icosphere and its pentagon-hexagon dual topology with NumPy.
Document the supported subdivision levels and verify spherical area closure.
```

## Merge workflow

The `master` branch is protected by an active repository ruleset with no bypass
actors. Use this workflow for every change:

1. Start from an up-to-date `master` and create a topic branch. Never commit or
   push directly to `master`, force-push it, or delete it.
2. Keep commits focused and follow the commit-message rules above. Push the
   topic branch and open a pull request targeting `master`.
3. Format the pull request title as a Conventional Commit because it becomes
   the squashed commit title. Write a useful pull request body because it
   becomes the squashed commit body.
4. Resolve every review conversation. No approving review is currently
   required, but unresolved conversations block merging.
5. Keep the pull request branch up to date with `master`. The required status
   checks run under a strict policy and must pass against the latest target
   branch state:
   - `test (3.11)`
   - `test (3.12)`
6. Wait for CodeQL analysis to finish. Do not merge if CodeQL reports a security
   alert rated High or higher, a code-scanning alert rated Error, a failed
   analysis, or missing results. Investigate and fix a real alert; dismiss an
   alert only when there is documented evidence that it is a false positive.
7. Squash-merge the pull request. Merge commits and rebase merges are disabled,
   and linear history is required, so `master` receives one commit per merged
   pull request.
8. After the merge, update the local `master` with a fast-forward pull and
   remove the topic branch when it is no longer needed. GitHub does not delete
   merged branches automatically for this repository.

## Verification reports

- Keep verification reports concise and in English. Do not create parallel
  translated copies.
- Maintain exactly three canonical reports under `docs/verification/`: the
  A0--A4 analytic verification, Simpson--Taflove 2004 reproduction, and
  Simpson--Heikes--Taflove 2006 reproduction.
- Record final models, equations, configurations, evidence, verdicts, and
  reproducibility limits. Keep exploratory failure history out of the final
  reports unless it is required to interpret the accepted result.

## Verification code

- Keep reusable solver, mesh, material, source, backend, and visualization
  functionality under `src/ionosphere_fdtd/`.
- Keep paper-specific models, reproduction CLIs, report generators, and
  directional verification under the repository-level `verification/`
  package. Run these workflows as `python -m verification.<workflow>` from a
  source checkout; do not add them to the distribution's console scripts.
- Do not include `verification/` or its paper-specific tests in wheels or
  source distributions. Confirm both archive contents after packaging changes.
- Extract a function into the runtime package only when its API and semantics
  are independent of a particular paper, figure, or acceptance criterion.

## Verification comparison plots

Use the Simpson–Taflove 2004 Figure 7 and Figure 8 comparison images as the
layout reference for new published-versus-reproduced verification plots.

- Place the published plot on the left and the reproduced plot on the right.
  Keep only the compact `Published` and `Reproduced` column headings; omit
  composite titles, paper captions, and paper panel markers such as `(a)` and
  `(b)`.
- Preserve every plot axis, tick label, axis label, legend, and scientifically
  meaningful annotation. Crop the published source tightly, but include the
  complete plot frame and enough space for all labels. Never cover, redraw, or
  clip a source frame merely to reduce whitespace.
- Match panel sizes by the data frame: the rectangular region enclosed by the
  axis spines, excluding tick labels, axis labels, and legends. Measure the
  published frame bounds in source pixels and place the reproduced Matplotlib
  `Axes` at the same target pixel width and height. Do not compare or match the
  outer raster-crop dimensions.
- Preserve the measured frame aspect ratio. When a figure contains multiple
  rows, measure each published frame independently, then scale and position
  each row so every published/reproduced pair has identical target frame
  dimensions.
- Maximize the data frames by trimming unused outer whitespace and using a
  sufficiently large canvas. Keep a clear inter-column gap so the reproduced
  y-axis label does not crowd the published frame. Prefer widening the canvas
  over shrinking matched frames or clipping labels.
- For reproduced plots, use 18-point axis labels and 16-point major tick labels
  and legends unless the source requires a larger accessible size. Use a white
  background and keep line, marker, grid, and legend styling consistent across
  comparison figures.
- Treat plot-layout changes as presentation-only work: do not recompute,
  smooth, rescale, truncate, or otherwise change verification data or metrics.
- Save final comparison images under `docs/verification/images/`. Before
  committing, inspect the rendered image visually, verify it with Pillow,
  assert the intended canvas and data-frame dimensions, run `git diff --check`,
  and confirm that no unrelated files changed.

Current geometry references:

- Figure 7 uses two matched `1200 x 955` pixel data frames per row on a
  `2880 x 2400` pixel canvas.
- Figure 8 uses matched `1250 x 988` pixel data frames on a `3040 x 1230`
  pixel canvas, with additional inter-column spacing for the enlarged
  reproduced y-axis label. Its reproduced panel uses 20-point axis labels and
  18-point major tick labels and legend text.
- Simpson–Taflove 2006 comparison plots use a `3400 x 1460` pixel canvas for
  Figure 5, a tighter `3320 x 1460` canvas for Figure 6, and a wider
  `3500 x 1460` canvas for Figure 7. Figures 5–6 use 22-point reproduced axis
  labels and 20-point reproduced major tick labels and legend text; Figure 7
  uses 20-point axis labels and 18-point major tick labels and legend text.
  Their matched published/reproduced data frames are
  `1400 x 1111` pixels for Figure 5, `1400 x 1104` pixels for Figure 6, and
  `1400 x 1099` pixels for Figure 7. The data-frame gaps are 290 pixels for
  Figure 5, 280 pixels for Figure 6, and 400 pixels for Figure 7, leaving room
  for their different reproduced labels. Measure these ratios from the
  original PDF page crops, not from a previously composed comparison image.
