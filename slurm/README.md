# Slurm submission scripts

These scripts cover the event-generation, sample-preparation, and neural-training
stages only.

Set up the Python environment on CSD3 from the repository root:

```bash
bash slurm/setup_environment.sh
```

Submit individual stages:

```bash
bash slurm/submit_event_generation.sh
bash slurm/submit_sample_preparation.sh
bash slurm/submit_neural_training.sh
```

Pass a non-default config as the first argument:

```bash
bash slurm/submit_neural_training.sh configs/quick_test/neural_training.json
```

The stage defaults are:

| Stage | Partition | Account | Resources |
| --- | --- | --- | --- |
| Event generation | `icelake` | `mphil-dis-sl2-cpu` | 2 CPUs, 16G |
| Sample preparation | `icelake` | `mphil-dis-sl2-cpu` | 4 CPUs, 64G |
| Neural training | `ampere` | `mphil-dis-sl2-gpu` | 4 CPUs, 64G, 1 GPU |

Each batch script starts from a clean module state by default:

```bash
module purge
module load ${MODULES:-python}
```

Environment setup can be controlled with optional variables:

```bash
MODULES="python/3.11 cuda/12.1" bash slurm/setup_environment.sh
VENV="$HOME/my_eft_venv" bash slurm/setup_environment.sh
```

Submit with an existing virtual environment:

```bash
sbatch --export=ALL,VENV="$HOME/eft_venv" slurm/run_neural_training.sbatch
```

Pass extra `sbatch` options through helper scripts with `SBATCH_ARGS`:

```bash
SBATCH_ARGS="--export=ALL,VENV=$HOME/eft_venv" bash slurm/submit_neural_training.sh
```
