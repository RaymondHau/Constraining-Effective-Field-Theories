# Slurm submission scripts

## Environment setup

From the repository root on CSD3:

```bash
bash slurm/setup_environment.sh
```

The scripts require Python 3.10+ and default to:

```bash
module purge
module load python/3.11.0-icl
```

The default virtual environment is `$HOME/eft_venv`. Override it with `VENV`
if required.

## Individual stages

```bash
bash slurm/submit_event_generation.sh
bash slurm/submit_sample_preparation.sh
bash slurm/submit_neural_training.sh
bash slurm/submit_validation_events.sh
bash slurm/submit_constrains.sh
```

Each helper accepts a stage config as its first argument:

```bash
bash slurm/submit_neural_training.sh configs/WBF/neural_training.json
```

All helpers default to `EFT_PROCESS=VBF`. After adding a matching
`configs/WBF/` directory, switch process with:

```bash
EFT_PROCESS=WBF bash slurm/submit_hybrid_workflow.sh
```

## Hybrid workflow

```bash
bash slurm/submit_hybrid_workflow.sh
```

The dependency chain is:

```text
event generation (icelake)
  -> sample preparation (icelake)
  -> neural training (ampere GPU)
  -> validation events (icelake)
  -> constraints (icelake)
```

Each stage starts only after the previous stage succeeds.

Default resources:

| Stage | Partition | Account | Resources |
| --- | --- | --- | --- |
| Event generation | `icelake` | `mphil-dis-sl2-cpu` | 1 node, 2 CPUs, 16G |
| Sample preparation | `icelake` | `mphil-dis-sl2-cpu` | 1 node, 4 CPUs, 64G |
| Neural training | `ampere` | `mphil-dis-sl2-gpu` | 1 node, 4 CPUs, 64G, 1 GPU |
| Validation events | `icelake` | `mphil-dis-sl2-cpu` | 1 node, 1 CPU, 16G |
| Constraints | `icelake` | `mphil-dis-sl2-cpu` | 1 node, 2 CPUs, 16G |

Override hybrid accounts independently when necessary:

```bash
CPU_ACCOUNT=... GPU_ACCOUNT=... bash slurm/submit_hybrid_workflow.sh
```

Pass extra Slurm options through `SBATCH_ARGS`:

```bash
SBATCH_ARGS="--time=12:00:00" bash slurm/submit_neural_training.sh
```

Stage logs are written under:

```text
$HOME/eft_slurm_logs/<job-name>/<job-id>/
```
