# multichains

Coarse-grained molecular dynamics workflows built around [OpenMM](https://openmm.org/), with support for **replicated multi-copy systems** on GPU, **temperature-quench** runs, and **benchmark** notebooks. Typical use cases include Go-model–style protein simulations and throughput scaling studies.

## What’s in this repo

| Path | Purpose |
|------|---------|
| `benchmark/` | GPU scaling benchmark (`bench.py`), example control file (`control.cntrl`), Jupyter notebooks (`combined_benchmark*.ipynb`), and a `setup/` tree (PSF, coordinates, force field XML, secondary-structure defs). |
| `temp_quench/` | **Temperature-quench** workflow: `temp_quench.py`, `control.cntrl`, **SLURM** example (`job.sh`), and `setup/`. Run the driver from this directory (or ensure it and the control file are on your working path) so e.g. `python temp_quench.py -f control.cntrl` resolves. |

Both drivers read a **control file** (plain key/value pairs) for the system and run parameters—see [Control file](#control-file). Driver-specific keys: **`temp_quench.py` uses `n_copies`** (chains per simulation); **`bench.py` uses `copies_list`** (comma-separated replica counts for the throughput sweep—see the **GPU benchmark** section).

## Requirements

- **Python 3.10** (as pinned in `env.yml`)
- **Conda** or **Mamba** (recommended: [micromamba](https://mamba.readthedocs.io/) or [conda-forge](https://conda-forge.org/))
- **CUDA-capable GPU** (optional but expected for `use_gpu = yes` and for `bench.py` benchmarking, which uses NVML via `pynvml`)
- For cluster runs: **SLURM** (optional; adapt `job.sh` to your site’s partitions, accounts, and GPU flags)

### Python packages for the driver scripts

The files `benchmark/bench.py` and `temp_quench/temp_quench.py` import only the packages below (plus the Python standard library). **`env.yml` lists exactly these**; exact versions are chosen by the conda solver when you create the environment (run `conda list` after install to see them).

| Package | Role |
|--------|------|
| **Python** 3.10 | Runtime |
| **OpenMM** | MD engine, CUDA/CPU platforms |
| **NumPy** | Arrays and numerics |
| **pandas** | Tabular output (e.g. benchmark stats) |
| **ParmEd** | PSF/topology I/O with OpenMM |
| **nvidia-ml-py** | Supplies the **`pynvml`** module for GPU stats (`bench.py` / GPU runs) |
| **openpyxl** | Excel export (`bench.py` writes `bench.xlsx` via `pandas`) |

The Jupyter notebooks under `benchmark/` may need extra packages (e.g. `jupyterlab`, plotting, MDAnalysis)—install those in the same env as needed.

## Environment setup

1. Clone the repository:

   ```bash
   git clone git@github.com:vuqv/multichains.git
   cd multichains
   ```

2. Create the conda environment from `env.yml`:

   ```bash
   conda env create -f env.yml
   conda activate bioenv
   ```

## Control file

Control files are simple `key = value` assignments (comments start with `#`). Paths are relative to the working directory from which you run the script.

Shared keys include topology, coordinates, force field, `temp_prod`, step counts, `traj_dir`, GPU flag, and output frequency. **Driver-specific keys:**

- **`temp_quench.py`:** `n_copies`, `temp_heating`, `heating_steps`, and the usual production keys (see `temp_quench/control.cntrl`).
- **`bench.py`:** `copies_list` — comma-separated integers for which replica counts to benchmark (e.g. `copies_list = 1, 2, 4`). If omitted, `bench.py` defaults to `1, 2, 4`. See `benchmark/control.cntrl`.

Invocation for both: `python <script.py> -f <control_file>` (or `--ctrlfile`).

**Temperature quench** (representative keys; trim or edit to match your system):

```text
# Forcefield definition
psffile = setup/Q04894_clean_ca.psf
prmfile = setup/Q04894_clean_nscal1_fnn1_go_bt.xml
secondary_structure_def = setup/secondary_struc_defs.txt
native_cor = setup/Q04894_clean_ca.cor
corfile = setup/Q04894_clean_ca.cor
#output-based name
outname = test

#Number of chains in single simulations.
n_copies = 1
#Directory to store the trajectory
traj_dir = traj

# Heating to unfold protein
temp_heating = 600
# equilibrium in 15 ns
heating_steps = 1_000

#temperature quenching, folding at low temperature
temp_prod = 300
# performed production run in 10^8 steps= 1.5 microseconds
mdsteps = 10_000

# Running simulation on GPU or CPU
use_gpu = yes
ppn = 1

# frequency to write dcd and checkpoint file
nstxout = 10
nstlog = 10
restart = no

```

**GPU benchmark** (adds `copies_list`; heating keys are not used by `bench.py`):

```text
copies_list = 1, 2, 4
# ... plus shared keys as in benchmark/control.cntrl
```

## Temperature quench (`temp_quench.py`)

Runs a **single** heating → production-style workflow with **temperature quench** stages and writes DCD/log files under `traj_dir`.

- **Replica count:** set **`n_copies`** in the control file (number of independent chains in one simulation). Example: `n_copies = 1` (see `temp_quench/control.cntrl`).
- **Script location:** `temp_quench/temp_quench.py`.

```bash
cd temp_quench
python temp_quench.py -f control.cntrl
```

## GPU benchmark (`bench.py`)

Measures **throughput vs. system size** by running the same template system at **many replica counts** in a loop, collecting steps/s, ns/day, and GPU stats (via `pynvml`), and writing **`bench.xlsx`** plus per-replica DCD/log files under `traj_dir`.

- **Replica counts:** set **`copies_list`** in the **control file** as comma-separated integers (e.g. `copies_list = 1, 2, 4, 8`). If the key is missing, `bench.py` uses `[1, 2, 4]`.
- **Control file:** supplies paths, `mdsteps`, `traj_dir`, `use_gpu`, `copies_list`, etc.—see `benchmark/control.cntrl`.

```bash
cd benchmark
python bench.py -f control.cntrl
```

## Notebooks

Open `benchmark/combined_benchmark.ipynb` (and `combined_benchmark_extend.ipynb`) in JupyterLab or Jupyter Notebook after activating `bioenv`. They may need extra packages beyond `env.yml`.

## Cluster example (SLURM)

`temp_quench/job.sh` is an **example** batch script (partition, account, node list, and GPU type are **site-specific**). Copy it, replace scheduler directives with your cluster’s settings, run from the directory that contains `temp_quench.py` and `control.cntrl` (or adjust paths), then submit with `sbatch job.sh` (or your local equivalent).


## License

This project is licensed under the [GNU General Public License v3.0](LICENSE) (GPL-3.0). See `LICENSE` for the full text.

## Citation

If this code supports a publication, cite OpenMM and any force fields or models you use, and describe your own modifications here as needed.
