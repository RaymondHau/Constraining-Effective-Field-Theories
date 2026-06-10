# EFT Script Workflow

This workspace contains Python versions of the notebooks plus JSON configuration files.

Install Python dependencies into your active environment:

```bash
python -m pip install -r requirements.txt
```

Run from Ubuntu, preferably in `tmux`:

```bash
cd ~/Dis_Project/MadGraph_tutorial/eft_script_workflow
tmux new -s eft_10m
./run_workflow.sh configs/paper_10m.json
```

Run individual stages:

```bash
./run_event_generation.sh
./run_sample_preparation.sh
./run_neural_training.sh
```

Each wrapper defaults to the `paper_10m` config for that stage. To use another
stage config, pass it as the first argument:

```bash
./run_event_generation.sh configs/quick_test/event_generation.json
./run_sample_preparation.sh configs/quick_test/sample_preparation.json
./run_neural_training.sh configs/quick_test/neural_training.json
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t eft_10m
```

Useful workflow configs:

- `configs/paper_10m.json`: stage order for the 10M paper-style setup.
- `configs/generation_only_10m.json`: stage order for event generation only.
- `configs/quick_test.json`: stage order for the smaller smoke-test workflow.

Each workflow points to separate stage configs:

- `configs/paper_10m/event_generation.json`
- `configs/paper_10m/sample_preparation.json`
- `configs/paper_10m/neural_training.json`

The same layout exists under `configs/quick_test/` and
`configs/generation_only_10m/`.

The stage JSON files are the workflow control panel. Each one only contains
settings that are relevant to that stage:

- Event generation config: MadGraph paths, process/model/EFT card settings, benchmark generation budgets.
- Sample preparation config: LHE/table paths, observables, benchmark/morphing information, smearing, cuts, sample sizes, target stability cuts.
- Neural training config: prepared-sample/model paths, observables/operators, estimator methods, neural-network hyperparameters.

The runner still also sets a few environment variables from the JSON for
backwards compatibility with the original converted notebook code.

Outputs and logs now default to project-local folders:

```text
logs/
tables/
processes/
```

Generated LHE files are read first from:

```text
generated_lhe_archive/
```

Large generated artifacts are not committed directly to this repository. Download
the generated tables, LHE archive, and bundled MadGraph installation from:

[Google Drive: `eft_generated_artifacts_2026-06-10.zip`](https://drive.google.com/file/d/1TevN3HIQ7Pl6PdZSvJ4BLHunqynZnBVT/view?usp=sharing)

Archive details:

```text
file:   eft_generated_artifacts_2026-06-10.zip
size:   about 6.2 GB
sha256: 3b7aa01501ca9e4697848c006e25eab5daf0b489cd8422378df925c4fc8607f9
```

After downloading, copy the ZIP into the repository root and unpack it there:

```bash
cd ~/Dis_Project/MadGraph_tutorial/eft_script_workflow
sha256sum eft_generated_artifacts_2026-06-10.zip
unzip eft_generated_artifacts_2026-06-10.zip
```

The checksum command should print the SHA-256 value shown above. Unzipping from
the repository root restores the expected project-local paths:

```text
external/MG5_aMC_v3_7_1/
generated_lhe_archive/
tables/
```

If those folders or files already exist and `unzip` asks whether to replace
them, answer `A` to replace all files from the archive.

The omitted generated files include the multi-GB training CSVs, `end_to_end_events.csv`,
the compressed LHE archive, and the local MG5 installation. Lightweight summaries,
trained RASCAL outputs, metrics, and performance plots may still be present in
`tables/madminer_style_training/`.

Reproduction/provenance files copied from the original workspace are stored in:

```text
repro/mg5_commands/
repro/mg5_cards/
repro/event_generation_logs/
```

These include the MG5 process command, the generated run/param/reweight cards,
default detector/shower cards, and logs for the archived event-generation runs.

The MadGraph installation used for event generation is expected at:

```text
external/MG5_aMC_v3_7_1/
```

The event-generation configs point to that local copy. The `EWdim6-full` model
resolves successfully in this bundled MG5 installation.
MG5 3.7.1 prints a warning that reweighting is best supported with Python 3.6.x;
if rerunning event generation fails under a newer Python, use an environment
compatible with MG5's reweighting feature.

Live MadGraph process directories stay under `processes/` when event generation
is rerun. Python itself is still supplied by your active environment or virtual
environment; the required Python packages are listed in `requirements.txt`.

This script workflow should avoid IDE/Jupyter reconnect crashes. It does not by itself make 10M-event sampling/training out-of-core; if preparation or training runs out of memory, the next step is to refactor those stages to chunked HDF5/Parquet/memmap pipelines.
