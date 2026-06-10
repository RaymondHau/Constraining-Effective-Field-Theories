# Possible Improvements

## Sample Preparation

- Add a cutflow summary for each benchmark and part. The current progress logs report parsed and accepted events, but they do not show which cuts reject events. A cutflow table should track counts after each major requirement:
  - at least two jets
  - at least two photons
  - photon `pt` and `eta` cuts
  - diphoton mass window
  - VBF jet `m_jj`, `delta_eta_jj`, and opposite-hemisphere cuts
  - object separation cuts
  - final accepted count

- Save the cutflow as both a CSV and a printed summary. This would make it easy to see whether an acceptance rate like `32,253 / 300,000 = 10.8%` is expected or caused by one overly aggressive cut.

- Report per-benchmark acceptance rates in `sample_summary.csv`. The current summary focuses on generated training rows and unique sampled events; adding base parsed/accepted counts by benchmark would make production runs easier to sanity-check.

- Consider making the diphoton mass window configurable by run profile. The current `122-128 GeV` window is reasonable for a smeared Higgs-to-diphoton selection, but it is likely one of the dominant sources of rejection.

- Add an optional `--dry-run-cutflow` mode that parses a limited number of events, applies the same smearing and cuts, prints the cutflow, and exits before building the large training samples.

## Neural Training

- Add explicit progress reporting per training epoch for streamed CSV chunks. The training loader now streams the train split, but the epoch log could also report how many chunks/rows were consumed.

- Add config controls for limiting validation/test prediction rows. The paper evaluation size is small enough for current loading, but quick diagnostics could run faster with a capped evaluation sample.

- Persist fitted scaler metadata as a standalone JSON or NPZ file before training starts. It is currently stored inside model checkpoints, but saving it separately would make debugging and reuse simpler.

## Repository Hygiene

- Add a short "large files" note to the README explaining which generated artifacts should be kept out of git, especially LHE archives, prepared CSV tables, trained models, and plots.

- Add a quick-test command sequence to the README that runs event generation, sample preparation, and neural training with the small config.
