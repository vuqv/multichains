# Temperature Quench Workflow

This folder contains scripts to run a two-stage coarse-grained OpenMM workflow on one or more non-interacting chain copies:

1. **Heating stage** at high temperature (`temp_heating`) for unfolding/equilibration.
2. **Quenching/production stage** at low temperature (`temp_prod`) for long folding dynamics.

All scripts read a plain-text control file (`key = value`).

## Files

- `equil.py`: single-stage equilibration at one target temperature (`temp_prod`).
- `temp_quench.py`: baseline temperature-quench workflow.
- `temp_quench_v2.py`: baseline workflow + optional positional restraints during heating only.
- `control.cfg`: runtime configuration.

## General Workflow (Both Scripts)

1. **Read and validate config**
   - Required keys: `psffile`, `corfile`, `prmfile`, `temp_prod`, `outname`, `mdsteps`.
   - Optional keys include `temp_heating`, `heating_steps`, `n_copies`, `traj_dir`, `nstxout`, `nstlog`, `restart`, `checkpoint_file`, `nstchk`, `use_gpu`, `ppn`.

2. **Build template system**
   - Load PSF/COR + force field.
   - Build OpenMM system (`createSystem`) and apply nonbonded switching settings.

3. **Replicate for multi-chain simulation**
   - Duplicate system/topology/positions to `n_copies`.
   - Keep copies intramolecular-only (no inter-copy nonbonded interaction).

4. **Choose run mode**
   - If `restart = yes` and checkpoint exists: skip heating, load checkpoint, continue quench to `mdsteps`.
   - Otherwise: run heating first, then start a fresh low-temperature quench.

5. **Reporting and checkpointing**
   - DCD trajectory: `nstxout`.
   - State log: `nstlog`.
   - Quench checkpoint reporter: `nstchk` (if `nstchk > 0`).
   - Final checkpoint saved again at run end.

## Restart and Checkpoint Behavior

- `restart = yes` enables checkpoint resume.
- `checkpoint_file` sets checkpoint path (default: `traj_dir/outname.chk`).
- `nstchk` controls checkpoint frequency during quench:
  - `nstchk > 0`: periodic writes via `CheckpointReporter`.
  - `nstchk = 0`: periodic checkpointing disabled.

This design protects long low-temperature runs from losing all progress after interruptions.

## `equil.py` Workflow

`equil.py` is for a simpler one-stage protocol (no heating/quenching split):

1. Read `control.cfg` and build replicated multi-copy system (`n_copies`).
2. Choose run mode:
   - If `restart = yes` and checkpoint exists: load checkpoint and continue to `mdsteps`.
   - Otherwise: initialize positions/velocities and start from step 0.
3. Run simulation at `temp_prod` for remaining steps.
4. Write outputs:
   - trajectory: `{outname}_equil.dcd`
   - log: `{outname}_equil.log`
   - checkpoint: default `{outname}.chk` (or `checkpoint_file` if provided)

Important behavior notes for `equil.py`:

- It **does** support restart-from-checkpoint flow (`restart = yes`).
- It **does** support periodic checkpoints via `nstchk` (if `nstchk > 0`).
- It always writes a final checkpoint at the end.
- It uses the same replicated intramolecular-only multi-copy setup as the quench scripts.

## Difference: `temp_quench.py` vs `temp_quench_v2.py`

- **Common behavior**
  - Same two-stage run logic (heating -> quench).
  - Same restart mode from checkpoint for quenching.
  - Same periodic checkpoint support via `nstchk`.

- **`temp_quench.py`**
  - No atom positional restraint feature.
  - Heating and quenching both use the same replicated force field model (except temperature/integrator/context changes).

- **`temp_quench_v2.py`**
  - Adds optional config:
    - `restraint_idx` (1-based atom indices/ranges like `1-117,166-214`)
    - `restraint_k` (kJ/mol/nm^2, default `1000.0`)
  - Applies harmonic positional restraints **only during heating**.
  - Removes restraints automatically by running quench on the unrestrained system.

In short: use `temp_quench.py` for the standard protocol, and `temp_quench_v2.py` when you need restrained heating before unrestrained quenching.

## When to Use Which Script

- `equil.py`: one-temperature equilibration/production only.
- `temp_quench.py`: two-stage heating then quenching, with quench restart + periodic quench checkpointing.
- `temp_quench_v2.py`: same as `temp_quench.py`, plus heating-only positional restraints (`restraint_idx`, `restraint_k`).

## Example Run

```bash
python equil.py -f control.cfg
```

or

```bash
python temp_quench.py -f control.cfg
```

or

```bash
python temp_quench_v2.py -f control.cfg
```

## Quick-Start `control.cfg`

Use this as a minimal starting point and adjust file paths and step counts for your system:

```ini
# Input files
psffile = setup/your_system.psf
corfile = setup/your_system.cor
prmfile = setup/your_forcefield.xml

# Output
outname = test_run
traj_dir = traj

# Simulation protocol
temp_heating = 600
heating_steps = 1000000
temp_prod = 300
mdsteps = 20000000

# Replicas
n_copies = 1

# Platform
use_gpu = yes
ppn = 1

# Output/checkpoint frequency
nstxout = 5000
nstlog = 5000
nstchk = 50000

# Restart options
restart = no
checkpoint_file = traj/test_run.chk

# v2-only (ignored by temp_quench.py and equil.py)
restraint_idx = 1-100
restraint_k = 1000.0
```

For restart after interruption, set `restart = yes` and keep the same `checkpoint_file`.

## Control File Notes

- `restraint_idx` and `restraint_k` are used only by `temp_quench_v2.py`.
- Unknown keys are ignored by the parser, so one `control.cfg` can be reused across all scripts in this folder.
