# Reproducing Fig. 6 (Lecci et al. 2017) — Project Guideline

**Paper:** Lecci S., Fernandez L.M.J., Weber F.D., Cardis R., Chatton J.-Y., Born J., Lüthi A.
*Coordinated infraslow neural and cardiac oscillations mark fragility and offline periods
in mammalian sleep.* **Science Advances** 3:e1602026 (2017).

**Goal:** Reproduce **Figure 6** — *"The 0.02-Hz oscillation aligns with heart rate in both
mice and humans"* — on human sleep data: show that during N2 sleep the sigma-band
(11.5–16 Hz) EEG power envelope and the instantaneous heart rate co-oscillate on the ~0.02-Hz
(≈50-s) time scale, quantified by a cross-correlogram.

---

## 0. Read this first (scope)

Figure 6 has six panels. Only the **human** panels are relevant to this project:

| Panel | Content | In scope? |
|------|---------|-----------|
| A | Mouse example bout: sigma power + heart rate | ❌ mouse only |
| B | Mouse single-animal cross-correlogram | ❌ mouse only |
| C | **Human example bout: sigma power (red) + heart rate (black)** | ✅ **yes** |
| D | **Human single-subject cross-correlogram** | ✅ **yes** |
| E | Mouse grand-average cross-correlogram | ❌ mouse only |
| F | **Human grand-average cross-correlogram, ± SEM** | ✅ **yes** |

Your final figure reproduces **panels C, D, and F**. The tiny raw-ECG insets inside panel C
(illustrating R-wave detection) are cosmetic — you do **not** need to reproduce them.

**What "success" looks like.** In humans the paper reports that heart rate declines rapidly
once sigma power peaks and rises again during sigma minima, so the sigma-vs-HR cross-correlogram
is **not** a sharp negative peak at lag 0 (that is the mouse result) but a lag-shifted structure
with a peak at a **short positive lag (~+5 s)** and modest amplitude (|r| up to ~0.1–0.2). Aim to
match the *shape* of panel F, not an exact r value — this cohort is much larger than the paper's
n = 27, so the numbers will differ.

---

## 1. Background (why this works)

During non-REM sleep, spindle/sigma-band power is not steady: it waxes and wanes with a
dominant periodicity near **0.02 Hz** (one cycle ≈ 50 s). Autonomic activity (heart rate) tracks
the same infraslow rhythm. Fig. 6 demonstrates this coupling by cross-correlating two 1-Hz time
courses within continuous non-REM (here: N2) sleep:

1. **Sigma power** — band-limited (11.5–16 Hz) EEG amplitude envelope (already provided).
2. **Heart rate** — instantaneous HR (bpm) derived from ECG R-peaks (you will build this).

You align the two, cut them into short windows, cross-correlate, and average.

---

## 2. What you are given

You have one project folder containing two subfolders:

- **`Sigma_Envelope_N2/`** — one CSV per subject, named `{ID}.csv` (`ID` is a numeric subject
  identifier). Columns:
  - `time` — **integer seconds from recording start**. The file contains **only N2 (stage-2)
    sleep seconds**, so `time` is not continuous: it jumps wherever the subject was not in N2.
  - `envelope` — sigma-band (11.5–16 Hz) EEG amplitude, one value per second (1 Hz).
  - *Because the file already lists exactly the N2 seconds, this `time` column is your N2 mask —
    you do not need any separate sleep-staging file.*

- **`ecg/`** — ECG beat files, named `{ID}_1.ecg`. **These are not raw ECG waveforms.** Each file:
  - Line 1 is a header (semicolon-separated metadata).
  - Every following line is a **cumulative R-peak time in milliseconds** from recording start.
  - So the interval between two consecutive lines is one R–R interval (in ms), and instantaneous
    heart rate = 60000 / (R–R interval in ms).
  - The folder may also contain `{ID}_2.ecg` files (a different recording night). **Ignore those;**
    use only `{ID}_1.ecg` files whose `{ID}` has a matching sigma CSV.

**Analysis population.** The subjects to analyze are exactly those with a file in
`Sigma_Envelope_N2/`. For each, the matching ECG is `ecg/{ID}_1.ecg`.

**Time base.** Both the sigma `time` (seconds) and the ECG beat times (ms) are measured **from
the same recording start**, so once you convert ECG ms → seconds, the two share one clock and can
be aligned second-by-second. (Verify this early — see §4 validation.)

---

## 3. Pipeline overview

```
   ecg/{ID}_1.ecg  (R-peak times, ms)
            │
   STEP 1   ├─ RR intervals → instantaneous HR → clean artifacts → resample to 1 Hz
            ▼
     per-subject heart-rate trace @ 1 Hz (seconds from recording start)
            │
   STEP 2   ├─ align with Sigma_Envelope_N2/{ID}.csv on shared seconds (this also
            │   restricts HR to N2); normalize sigma; 4-s smooth sigma
            ▼
   STEP 3   ├─ split each continuous N2 stretch into 120-s windows; z-score each window
            ▼
   STEP 4   ├─ cross-correlate (heart rate = source) per window;
            │   average within subject, then across subjects (+ SEM)
            ▼
   STEP 5   └─ Panel C (example bout), Panel D (one subject), Panel F (grand mean ± SEM)
```

---

## 4. STEP 1 — Build the 1-Hz heart-rate trace

For each subject, turn `{ID}_1.ecg` into an instantaneous heart-rate time course sampled once
per second.

1. **Read beat times.** Skip line 1 (header). Parse the remaining lines as integers = R-peak
   times in **milliseconds**. Convert to seconds: `t_beat = ms / 1000`.
2. **Instantaneous HR.** For each pair of consecutive beats: `RR = t_beat[i] − t_beat[i−1]` (s);
   `HR = 60 / RR` (bpm), assigned to time `t_beat[i]`.
3. **Reject artifacts (important).** Before resampling, discard non-physiological beats:
   - Drop any R–R giving HR outside **[30, 220] bpm** (ectopic beats, missed/double detections).
   - Recommended: also drop isolated beats whose HR differs from the local median (e.g. a 5-beat
     window) by more than ~30 %. Keep the rule simple and write down exactly what you used.
4. **Resample to a 1-Hz grid.** From the surviving (time, HR) points build a continuous
   instantaneous-HR curve (linear interpolation of the tachogram is fine) and sample it at every
   integer second `s = 0, 1, 2, …`. Result: `hr[s]` in bpm.
   - **Do not interpolate across long gaps.** If the two beats bracketing a second are more than a
     few seconds apart (e.g. > 5 s — an artifact or asystole gap), mark that second as missing
     rather than inventing a value.

You may store this as a full-night 1-Hz trace; the restriction to N2 happens automatically in
STEP 2 when you align with the sigma file.

**Validation of STEP 1 (do before continuing):**
- Population median HR should sit around **60–75 bpm**; review subjects with implausible values.
- **Clock alignment (critical):** the seconds present in the sigma CSV (all N2) should have heart
  beats available in the same second range. Check that, for several subjects, N2 seconds fall
  well inside the ECG's covered time span and that the two overlap. If HR is systematically
  missing where sigma exists, the two clocks are misaligned — resolve before continuing.
- Plot a few minutes of HR for 3–5 subjects and eyeball it for spikes/dropouts.

---

## 5. STEP 2 — Align and preprocess the two traces

For each subject, load the sigma CSV and the heart-rate trace and **align them on their common
integer seconds** (inner join on `time`). Because the sigma file only contains N2 seconds, this
join automatically keeps only N2. Then:

1. **Normalize sigma to % of its mean** (for the panel-C plot, matching the paper's y-axis):
   `sigma_pct = 100 × envelope / mean(envelope over all N2 seconds)`.
2. **Smooth sigma** with a **4-s centered (symmetric) moving average** (this is the paper's human
   preprocessing). Leave heart rate unsmoothed.
3. *(Optional) power vs amplitude:* the paper uses sigma *power*; the provided envelope is
   *amplitude*. If you want strict power units, square the envelope before normalizing. It barely
   changes the z-scored cross-correlation. Note which you used.

*(The paper additionally upsampled both traces to 100 Hz for a finer lag axis. The data here is
natively 1 Hz; keeping 1 Hz is fine and gives lags in whole seconds. Only upsample if you want a
smoother-looking x-axis, and then interpolate both traces identically.)*

---

## 6. STEP 3 — Continuous stretches and 120-s windows

1. **Find continuous N2 stretches.** After the join, a stretch is a maximal run of seconds where
   `time` increases by exactly 1. A jump ends the stretch. (Never assume the trace is continuous —
   compute it from the `time` differences.)
2. **Cut into 120-s windows.** Split each stretch into non-overlapping **120-s windows**; drop any
   trailing piece shorter than the window. (If you prefer to keep more data, a floor of ≥ 100 s is
   acceptable — state your choice.)
3. **z-score each window.** Within every 120-s window, z-transform **both** signals independently:
   `(x − mean) / SD`. (This is the paper's "the 120-s intervals were z-transformed" step; it
   removes per-window level/scale so the correlation reflects co-fluctuation, not absolute values.)

Subjects with little consolidated N2 yield few or no valid windows — that is expected; they simply
contribute less to the average.

---

## 7. STEP 4 — Cross-correlation

For each 120-s window, compute the **normalized cross-correlation** between the z-scored sigma and
z-scored heart-rate traces as a function of lag τ.

- **Heart rate is the source signal.** Fix and document the convention:
  `R(τ) = mean_t [ sigma_z(t + τ) · hr_z(t) ]`, so **positive τ means sigma lags heart rate**.
  Check the sign against panel F (human peak at small *positive* lag); if your average correlogram
  is mirrored, flip the lag axis.
- **Lag range.** Compute lags to at least **±60 s** (half the window). Report ±60 s (the published
  human panels show a wider ±100–200 s axis, but with 120-s windows the informative structure lives
  within ~±60 s).
- **Normalization.** Since each window is z-scored, use the normalized correlation (divide by the
  number of overlapping samples at each lag) so that `R(0)` is a Pearson r.
- **Average in two stages, in this order:**
  1. **Within subject** — average the correlograms over all that subject's 120-s windows → one
     correlogram per subject.
  2. **Across subjects** — average the per-subject correlograms → the grand-mean correlogram;
     compute **SEM across subjects** at each lag (for panel F shading).

Keep the per-subject correlograms (a `subjects × lags` table): you need them for panel D (one
subject), panel F (mean ± SEM), and statistics.

**Minimum-data rule.** Require each subject to contribute at least a few windows (e.g. ≥ 3) before
including them in the grand average. Report how many subjects passed.

---

## 8. STEP 5 — Make the figure (panels C, D, F)

**Panel C — example bout (one subject).** Choose a subject/stretch with a clearly oscillating
sigma trace (~50-s rhythm) and good HR coverage (a representative example; say so in the caption).
Plot a ~120–220-s segment: **sigma in % of mean (red, left y-axis)** and **heart rate in bpm
(black, right y-axis)** vs time (s).

**Panel D — single-subject cross-correlogram.** Plot that subject's within-subject averaged
correlogram: correlation r (y) vs lag τ in seconds (x), with a dashed vertical line at τ = 0.

**Panel F — grand-average cross-correlogram.** Plot the across-subject mean correlogram with
**± SEM shading** and a dashed line at τ = 0. Annotate the peak height and its lag, and state
**n** (subjects in the average). Compare qualitatively to the paper's human panel F.

*Optional statistics:* at the peak lag, test whether mean r differs from 0 across subjects
(one-sample t-test); report the peak lag and value. Don't over-interpret magnitude.

---

## 9. Parameter reference (all fixed numbers)

| Parameter | Value | Why |
|----------|-------|-----|
| Sigma band | 11.5–16.0 Hz | already applied in the provided envelopes |
| Sleep stage | **N2 only** | Fig. 6 human analysis was run separately for N2 |
| Trace sampling rate | 1 Hz | provided data (upsampling optional) |
| HR plausibility filter | 30–220 bpm | artifact rejection |
| Max HR interpolation gap | ~5 s | don't interpolate across longer gaps |
| Sigma smoothing | 4-s centered moving average | paper (human) |
| Window length (z-score unit) | **120 s** | paper |
| Min windows per subject | ≥ 3 (recommended) | QC |
| Cross-corr source signal | **heart rate** | paper |
| Lag axis | ±60 s (report) | window-limited |
| Averaging | within-subject → across-subject; SEM across subjects | paper |

---

## 10. QC / validation checklist

- [ ] Population median HR ≈ 60–75 bpm; outliers reviewed.
- [ ] Clock alignment confirmed: N2 seconds (sigma) have heart beats available in the same range.
- [ ] Continuous stretches computed from `diff(time) == 1`, not assumed.
- [ ] Each 120-s window z-scored independently (both signals).
- [ ] Cross-correlation sign convention fixed and checked against panel F.
- [ ] Grand average done as within-subject first, then across subjects.
- [ ] n reported for panel F; number of subjects dropped by QC reported.
- [ ] Panel C: % of mean sigma (red) + bpm (black), dual y-axis.

---

## 11. Common pitfalls

1. **Treating `.ecg` as a waveform.** It is a list of **R-peak times in milliseconds**.
   Differencing consecutive lines gives R–R intervals directly.
2. **Wrong units.** Values are ms, not samples. Sanity check: last value / 1000 / 3600 ≈ recording
   length in hours (should be ~8–9 h).
3. **Skipping artifact rejection.** A few spurious short R–R intervals produce 200+ bpm spikes that
   survive z-scoring and corrupt correlograms. Filter to 30–220 bpm first.
4. **Interpolating HR across long gaps** (arousals, sensor loss) — mark missing instead.
5. **Averaging across subjects without within-subject averaging first** — over-weights subjects with
   many windows.
6. **Forgetting per-window z-scoring** — absolute level differences then dominate the correlation.
7. **Sign/lag confusion** — always validate the correlogram's peak lag sign against the paper.
8. **Expecting the mouse result** — humans do **not** show a sharp negative peak at lag 0; that is
   mice. Humans show a weaker, lag-shifted structure (peak ~+5 s).
9. **Using `_2` ECG files** — those are a different night; analyze only `_1` files matching a sigma ID.

---

## 12. Deliverables

1. A script that builds the per-subject 1-Hz heart-rate traces from the ECG beat files.
2. A script that aligns sigma + HR, windows and z-scores them, cross-correlates, and saves the
   per-subject cross-correlograms (a `subject × lag` table).
3. The **Fig. 6 figure** (panels C, D, F) as a vector/PNG file.
4. A short methods paragraph (mirroring §4–§8) and the QC numbers from §10.

---

## 13. Out of scope / notes

- **Mouse panels (A, B, E):** not reproducible on human data.
- **The 0.02-Hz spectral analysis (Fig. 1 of the paper)** is *not* required for Fig. 6 — Fig. 6
  works directly on the normalized 1-Hz traces. Optionally, band-pass filtering both traces to
  0.01–0.03 Hz (as in the paper's "Filtered" insets) can make the coupling visually clearer, but it
  is not needed for panels C/D/F.
