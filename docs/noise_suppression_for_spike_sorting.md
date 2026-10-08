# Noise suppression and downstream spike sorting

Which of NPXplorer's noise-suppression steps (`src/noise.rs`) could be useful in an
exported file that is then spike-sorted, and what it would take to export them.

Short answer: none of the three steps should be applied to data that goes into a
template-matching sorter as they are now. Wavelet shrinkage is the only one with a
plausible benefit, and only in a modified form. The isolated-events detector is more
useful as an artifact *mask* than as a filter. The noise gate should stay visual.

## What the sorter expects

Kilosort 4 (and most sorters built on SpikeInterface) assume that, after highpass and
common-reference subtraction:

- **noise is stationary and roughly Gaussian.** The whitening matrix is estimated from
  the data's channel covariance, and detection thresholds (`Th_universal`,
  `Th_learned`) are in units of whitened noise σ.
- **processing is linear.** A unit's waveform is assumed to be the same shape at every
  amplitude, scaled. Template matching subtracts scaled templates from the data.
- **small spikes are kept.** Drift estimation is based on the positions of all
  detected spikes, most of which are small.
- **amplitudes are meaningful.** Quality metrics (amplitude cutoff, SNR, presence
  ratio of small units) are computed on the exported values.

All three noise-suppression steps are non-linear in amplitude, so each one breaks at
least one of these assumptions. The question is whether the gain outweighs that.

## 1. Wavelet shrinkage

**What it does:** per channel, the detail coefficients of a decimated orthogonal
wavelet transform are shrunk by k × σ of their level (σ from the median absolute
coefficient). With the defaults (sym4, 2 levels, k = 3, soft) at 30 kHz, level 1 covers
roughly 7.5–15 kHz and level 2 3.75–7.5 kHz. Most spike energy is at 300–3000 Hz, so
this mainly removes high-frequency noise and keeps the sharp onset of large spikes.

**Possible benefit:** a modest reduction of high-frequency noise, which could lower the
noise σ the sorter sees and slightly improve detection of mid-sized units.

**Problems for sorting:**

- *Amplitude bias.* Soft shrinkage removes k·σ from every coefficient above the
  threshold, so large and small spikes lose a different fraction of their
  high-frequency content. Waveform shape then depends on amplitude, against the
  linear-template assumption.
- *Not shift-invariant.* A decimated DWT gives different results when the signal is
  shifted by one sample. Within one chunk this is invisible; at chunk boundaries of an
  export, and between spikes of the same unit at different phases, it adds jitter.
- *σ estimated per window.* In the display, σ comes from the current view. In an
  export, every chunk would get its own thresholds, so the noise level could step at
  chunk boundaries.
- *Little to gain.* Template matching is already a matched filter: noise outside the
  spike band contributes little to the match score. A linear lowpass (e.g. 6–7 kHz)
  would get most of the same benefit without the non-linear distortion.

**If it were exported:**

1. Replace the decimated DWT with a stationary (undecimated) wavelet transform, or
   average over cycle spins, so the result is shift-invariant and seamless at chunk
   boundaries.
2. Estimate each channel's per-level σ once in the export's pre-pass (the same pass
   that estimates DC offsets and destripe's AGC floor) and keep it fixed for the whole
   file.
3. Use only levels above the spike band (≥ 6 kHz at 30 kHz sampling), and prefer a
   gentler non-negative garrote or firm threshold to soft shrinkage, to reduce the
   amplitude bias.
4. Margin per chunk: the transform's support at the deepest level
   (taps × 2^levels samples), well below the export's existing margins.

**Verdict:** low priority. Worth trying only after a comparison against a plain linear
lowpass (see *Validation* below).

## 2. Isolated events

**What it does:** an event (a same-sign run whose peak reaches the event level) is
kept only if it sits inside at least `min_channels` physically adjacent channels that
all reach `neighbour_frac` of its peak within ±0.3 ms. Unsupported events are scaled
down by `attenuation`.

**Possible benefit:** removes events confined to a single channel: electrical pops,
glitches on one bad channel, and similar artifacts that would otherwise be detected as
spikes or distort the whitening covariance.

**Problems for sorting:**

- *Removes real units.* With the defaults (3 channels at ≥ 90 % of the peak), many real
  units fail. A spike's amplitude often falls off steeply between neighbouring sites,
  especially for small, close units and axonal spikes. Those units would be attenuated,
  some below detection.
- *Chops waveforms.* Only the same-sign run around the peak is attenuated, not the
  whole waveform. A spike that fails the test loses its trough but keeps its
  repolarisation phase, producing a shape the sorter never sees otherwise.
- *Redundant.* Template matching already models each unit's spatial footprint and
  rejects single-channel events that match no template.

**A better use for export:** turn it into an artifact *mask* rather than a filter.

- Use a stricter, artifact-specific rule: an event far larger than physiological
  spikes on one channel, or a near-identical event on all channels of a shank at once
  (the opposite case, e.g. stimulation or movement artifacts).
- Write the masked time ranges to the provenance file and to a separate file in a
  format sorters accept (SpikeInterface can exclude periods or blank them with
  `remove_artifacts`).
- Or blank the masked samples in the export by interpolating across them, which is
  linear and leaves every other sample untouched.

Single-channel problems are better handled by removing the channel (Remove channels,
or the classification's "noisy" channels) than by event-wise attenuation.

**Verdict:** don't export as it is. Reuse the detector for an artifact mask if
artifacts turn out to be a real problem in sorting results.

## 3. Soft-knee noise gate

**What it does:** scales every sample by a smooth function of its amplitude, ~0 below
the centre level (default 20 µV) and ~1 well above it.

**Problems for sorting:** this one removes exactly what the sorter needs.

- Every spike below ~20 µV disappears. Many cortical units peak at 30–60 µV, and the
  tails of every waveform are below 20 µV, so even large units lose their shape.
- The noise becomes zero-inflated and non-Gaussian. Whitening estimates its covariance
  from that noise, so whitened data and all thresholds in σ units become unreliable.
- Drift estimation loses most of its spikes.
- Amplitude-based quality metrics become meaningless.

**Verdict:** keep it strictly visual. It is useful for seeing large-unit structure in
the heatmap and harmful in any analysis.

## Where the real gains are

Things that would help sorting more than any of the above, all linear or masking:

- **Stimulation-artifact blanking.** For recordings with electrical stimulation, blank
  or interpolate a short window around each stimulus, using the TTL onsets the app
  already loads. This is the standard approach and is likely the largest gain for
  stimulation recordings.
- **The existing preprocessing.** Phase shift, notch filters and destripe are linear
  and sorter-safe; they are what the export already writes.
- **Channel removal** for dead, noisy and outside-brain channels, from the channel
  classification.

## How a noise step would fit into the export

If any step is exported later, it slots in as one more stage after `preprocess()`, in
the same chunk pipeline:

- **Parameters fixed for the whole file:** anything data-dependent (wavelet σ, event
  levels) is estimated in the pre-pass, never per chunk.
- **Margins:** `noise::margin_samples` (10 ms) or the wavelet support, added to the
  chunk margins that are already there.
- **Provenance:** recorded as a separate step with all parameters, and marked as
  non-linear.
- **UI:** a separate, default-off checkbox in the export dialog, labelled
  experimental, not tied to whether the step is enabled in the view.

## Validation before shipping any of it

Unit counts alone don't show whether a step helps; a step can increase them by
splitting units. Two checks that do:

1. **Hybrid ground truth.** Inject known templates at known times into a real
   recording (SpikeInterface's hybrid tools), export with and without the step, sort,
   and compare recall and precision per injected unit, split by amplitude.
2. **Paired sorting of real data.** Sort the same recording with and without the step
   and compare the matched units' quality metrics (ISI violations, amplitude cutoff,
   presence ratio) and the number of units with isolation-distance-level separation.

A step is worth exporting only if it improves recall for small units without lowering
precision, compared to the same export with a linear lowpass in its place.
