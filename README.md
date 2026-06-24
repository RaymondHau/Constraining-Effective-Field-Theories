# EFT Script Workflow

This repository provides script-based event generation, sample preparation,
neural-estimator training, validation-event generation, and EFT constraint
evaluation.

## Environment

The Python workflow requires Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

On CSD3:

```bash
bash slurm/setup_environment.sh
```

## Process configuration

Configuration is organized by physics process. The current process is WBF:

```text
configs/
  WBF/
    workflow.json
    event_generation.json
    sample_preparation.json
    neural_training.json
    validation_events_config.json
    constrains_config.json
```

All wrappers default to `EFT_PROCESS=WBF`. To add another process later, create
the same six files under its own `configs/<PROCESS>/` directory, then select it
without changing code:

```bash
EFT_PROCESS=VBF ./run_workflow.sh
EFT_PROCESS=VBF bash slurm/submit_hybrid_workflow.sh
```

An explicit config path can still be passed as the first argument to an
individual stage wrapper.

### Paper benchmark (arXiv:1805.00020)

The WBF configuration follows the example analysis in arXiv:1805.00020:

- `qq -> qq h -> qq ZZ -> qq 4l`, with `l = e, mu`;
- the dimensionless parameters `fW v^2 / Lambda^2` and
  `fWW v^2 / Lambda^2`, represented as `fW` and `fWW`;
- the square parameter domain `[-1, 1]^2` and fixed ratio reference
  `(0.393, 0.492)`;
- a 15-point, total-degree-four morphing basis;
- 5.5 million generated parton-level events, 10 million parameterized
  training examples, and 50,000 evaluation examples;
- 42 features from the two tagging jets, four leptons, four-lepton system,
  two reconstructed Z candidates, and dijet system;
- five hidden layers of 100 `tanh` units, 50 epochs, and the paper values
  `alpha=100` for RASCAL and `alpha=5` for CASCAL;
- expected constraints from 36 observed events over the published parameter
  range.

The paper does not tabulate its 15 numerical morphing coordinates. The config
therefore uses a deterministic full-rank basis spanning the same parameter
domain (condition number 75.35). The default preparation uses the paper's
idealized parton-level detector model; shower and detector simulation remain
disabled.

## Workspace layout

```text
madgraph_work/
  external/MG5_aMC_v3_7_1/
  generated_lhe_archive/
  mg5_commands/
  processes/
  validation_events/
table_outputs/
  madminer_style_training/
plotting_outputs/
logs/
```

## Local execution

Run the complete configured WBF workflow:

```bash
./run_workflow.sh
```

Run stages individually:

```bash
./run_event_generation.sh
./run_sample_preparation.sh
./run_validation_events.sh
./run_validation_morphing_comparison.sh
./run_neural_training.sh
./run_constrains.sh
```

## CSD3 execution

Submit individual stages:

```bash
bash slurm/submit_event_generation.sh
bash slurm/submit_sample_preparation.sh
bash slurm/submit_validation_events.sh
bash slurm/submit_validation_morphing_comparison.sh
bash slurm/submit_neural_training.sh
bash slurm/submit_constrains.sh
```

Submit the dependency-chained CPU/GPU workflow:

```bash
bash slurm/submit_hybrid_workflow.sh
```

Event preparation and validation generation use `icelake` with account
`mphil-dis-sl2-cpu`. Neural training and gradient-based score constraints use
`ampere`, account `mphil-dis-sl2-gpu`, and one GPU per job.

## Neural-estimator loss conventions

Ratio samples use `y=1` for numerator events drawn from `theta0` and `y=0` for
denominator events drawn from the fixed reference point. CASCAL and ALICES use
classifier loss plus numerator-only joint-score regression. RASCAL implements
the ratio / inverse-ratio regression of Eq. (37) in arXiv:1805.00020, plus the
same numerator-only score term. Configured `alpha` values multiply raw squared
score errors in the dimensionless coordinates defined by
`physics.morphing_theta_scale`, matching the paper's convention rather than
data-standardized MSEs.

Training writes `test_metrics.csv` for clearly labelled joint-target
diagnostics and `test_objective_metrics.csv` for classifier, ratio/inverse-ratio,
and numerator-score held-out risks. Log-r metrics and plots use the continuous
raw network prediction directly; no probability calibration is applied.

## Constraint statistics

The constraint stage reports two distinct diagnostics. `q_score` is the primary
score-test statistic: at each parameter point it sums the network gradient,
centres it with the held-out numerator mean, and normalizes it with the held-out
score covariance. The covariance includes both event fluctuations and the
finite-sample uncertainty of the estimated calibration mean, including a
conservative interpolation term between sparse calibration points. Its
two-parameter confidence contours use the configured chi-square thresholds.
`q_relative` is the raw learned likelihood-ratio scan and is retained as a
diagnostic of global log-r quality.

Score calibration is read from the held-out `test_predictions.csv`, using only
`y=1` numerator rows and never the validation events being constrained. The old
per-event log-r residual rescaling is disabled by default and is not used for
the published score-test contours.

Validation datasets have known truth points, so score closure plots use an
independent fine grid centred on that truth. Raw likelihood plots use their own
likelihood-selected refinement grid. A third full-range coarse score plot is
written as a global audit for disconnected or boundary minima; it must not be
mistaken for the fine closure plot.

### Optuna alpha scan

Run resumable 100-epoch logarithmic alpha scans for RASCAL, CASCAL, and ALICES
as a three-task Ampere job array. Each method uses one GPU, so all three scans can
run concurrently when the scheduler has three GPUs available:

```bash
bash slurm/submit_optuna_alpha_scan.sh
```

The default is 20 trials per method. Override it without editing the batch file:

```bash
TRIALS_PER_METHOD=30 SCAN_EPOCHS=100 bash slurm/submit_optuna_alpha_scan.sh
```

Each scan minimizes raw hard-label marginal binary cross-entropy, averaged
uniformly over theta points. The Brier score and joint-log-r diagnostics are also
recorded without altering the continuous prediction. To
avoid concurrent SQLite writes on the shared filesystem, each method has its
own resumable database and results directory:
`table_outputs/optuna_alpha_scan/{rascal,cascal,alices}/`. Each directory contains
`optuna_alpha.db`, `best_alpha_summary.csv`, `best_alphas.json`, and the trial
logs. Re-running the submission command resumes completed studies and fills only
the remaining trials.

After copying the completed result directories back to the local repository,
generate the alpha-versus-marginal-BCE figure and joint-target diagnostic
performance plots from each winning checkpoint with:

```bash
python scripts/optuna_alpha_scan.py --plot-only
```

Matplotlib PNGs are written to `plotting_outputs/optuna_alpha_scan/`; best-model
performance plots are under its `best_performance/` subdirectory. No performance
plots are made for non-winning checkpoints. During optimization, checkpoints from
non-winning trials are deleted as soon as a better trial is known; trial logs,
histories, and scalar metrics remain for diagnostics.

## Generated artifact archive

Large generated artifacts are available from
[Google Drive](https://drive.google.com/file/d/1TevN3HIQ7Pl6PdZSvJ4BLHunqynZnBVT/view?usp=sharing).

```text
file:   eft_generated_artifacts_2026-06-10.zip
size:   about 6.2 GB
sha256: 3b7aa01501ca9e4697848c006e25eab5daf0b489cd8422378df925c4fc8607f9
```

Place it in the repository root and unpack it there:

```bash
sha256sum eft_generated_artifacts_2026-06-10.zip
unzip eft_generated_artifacts_2026-06-10.zip
```

The configured MadGraph installation is:

```text
madgraph_work/external/MG5_aMC_v3_7_1/
```
