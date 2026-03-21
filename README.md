# multichains

Coarse-grained molecular dynamics workflows built around [OpenMM](https://openmm.org/), with support for **replicated multi-copy systems** on GPU, **temperature-quench** runs, and **benchmark** notebooks. Typical use cases include Go-model–style protein simulations and throughput scaling studies.

## What’s in this repo

| Path | Purpose |
|------|---------|
| `benchmark/` | GPU scaling benchmark (`bench.py`), example control file (`control.cntrl`), Jupyter notebooks (`combined_benchmark*.ipynb`), and a `setup/` tree (PSF, coordinates, force field XML, secondary-structure defs). |
| `temp_quench/` | **Temperature-quench** workflow: `control.cntrl`, **SLURM** example (`job.sh`), and `setup/`. The driver is `benchmark/temp_quench.py`—run it from `benchmark/` or place a copy next to your control file so `job1.sh` can call `python temp_quench.py -f control.cntrl`. |

Both drivers read a **control file** (plain key/value pairs) for the system and run parameters—see [Control file](#control-file). How you choose the **number of replica chains** differs: **`temp_quench.py` uses `n_copies` in the control file**; **`bench.py` uses `copies_list` in the Python script** (see the **Temperature quench** and **GPU benchmark** sections below).

## Requirements

- **Python 3.10** (as pinned in `env.yml`)
- **Conda** or **Mamba** (recommended: [micromamba](https://mamba.readthedocs.io/) or [conda-forge](https://conda-forge.org/))
- **CUDA-capable GPU** (optional but expected for `use_gpu = yes` and for `bench.py` benchmarking, which uses NVML via `pynvml`)
- For cluster runs: **SLURM** (optional; adapt `job1.sh` to your site’s partitions, accounts, and GPU flags)

### Python packages for the driver scripts

The files `benchmark/bench.py` and `benchmark/temp_quench.py` import only the packages below (plus the Python standard library). **`env.yml` lists exactly these**; exact versions are chosen by the conda solver when you create the environment (run `conda list` after install to see them).

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

Shared keys used by both drivers include topology, coordinates, force field, temperatures, step counts, `traj_dir`, GPU flag, and output frequency. **`n_copies` is required for `temp_quench.py`** (see below); **`bench.py` ignores it for the sweep** and uses `copies_list` in the script instead.

Example (trim keys to match your run):

```text
psffile = setup/Q04894_clean_ca.psf
prmfile = setup/Q04894_clean_nscal1_fnn1_go_bt.xml
secondary_structure_def = setup/secondary_struc_defs.txt
native_cor = setup/Q04894_clean_ca.cor
corfile = setup/Q04894_clean_ca.cor

temp_heating = 600
temp_prod = 300
ppn = 1
outname = test
mdsteps = 200_000
heating_steps = 1_000

traj_dir = traj
use_gpu = yes
nstxout = 5000
nstlog = 5000
restart = no
```

Invocation for both: `python <script.py> -f <control_file>` (or `--ctrlfile`).

## Temperature quench (`temp_quench.py`)

Runs a **single** heating → production-style workflow with **temperature quench** stages and writes DCD/log files under `traj_dir`.

- **Replica count:** set **`n_copies`** in the control file (number of independent chains in one simulation). Example: `n_copies = 1` (see `temp_quench/control1.cntrl`).
- **Script location:** `benchmark/temp_quench.py` (copy or symlink it next to your control file if you run from another directory, e.g. `temp_quench/` with `job1.sh`).

```bash
cd benchmark
python temp_quench.py -f control.cntrl
```

## GPU benchmark (`bench.py`)

Measures **throughput vs. system size** by running the same template system at **many replica counts** in a loop, collecting steps/s, ns/day, and GPU stats (via `pynvml`), and writing **`bench.xlsx`** plus per-replica DCD/log files under `traj_dir`.

- **Replica counts:** edit **`copies_list`** at the **bottom of `bench.py`** (not the control file). Default:

  ```python
  copies_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  ```

  Change that list to choose which `n_copies` values are benchmarked.

- **Control file:** still supplies paths, `mdsteps`, `traj_dir`, `use_gpu`, etc. An `n_copies` line may be present in the file for parser compatibility, but **the sweep is entirely driven by `copies_list` in the script.**

```bash
cd benchmark
python bench.py -f control.cntrl
```

## Notebooks

Open `benchmark/combined_benchmark.ipynb` (and `combined_benchmark_extend.ipynb`) in JupyterLab or Jupyter Notebook after activating `bioenv`. They may need extra packages beyond `env.yml`.

## Cluster example (SLURM)

`temp_quench/job1.sh` is an **example** batch script (partition, account, node list, and GPU type are **site-specific**). Copy it, replace scheduler directives with your cluster’s settings, ensure `temp_quench.py` and the control file are on the job’s working path, then submit with `sbatch job1.sh` (or your local equivalent).


## License

This project is licensed under the [GNU General Public License v3.0](LICENSE) (GPL-3.0). See `LICENSE` for the full text.

## Citation

If this code supports a publication, cite OpenMM and any force fields or models you use, and describe your own modifications here as needed.
