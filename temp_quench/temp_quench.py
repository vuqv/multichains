import copy
import getopt
import os
import sys
from distutils.util import strtobool

import numpy as np
import parmed as pmd
from openmm import *
from openmm.app import *
from openmm.unit import *


def read_control_file(path: str) -> dict[str, str]:
    """
    Parse control files as lines of ``key = value``.
    Blank lines and lines whose first non-whitespace character is ``#`` are skipped.
    Only the first ``=`` splits key from value; values may contain ``=``.
    Inline ``#`` after the value starts a comment (trimmed from the value).
    """
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if "#" in value:
                value = value.split("#", 1)[0].rstrip()
            cfg[key] = value
    return cfg


def replicate_cg_system_intra_only(template_system, n_copies):
    """
    Replicate an OpenMM template system n_copies times into one full system.

    Each copy is independent:
      - bonded/angle/torsion terms are duplicated within each copy
      - constraints are duplicated within each copy
      - CustomNonbondedForce uses interaction groups so copies do NOT see each other

    Supported forces:
      - HarmonicBondForce
      - CustomAngleForce
      - PeriodicTorsionForce
      - CustomTorsionForce
      - CustomNonbondedForce
    """

    n_particles = template_system.getNumParticles()
    full_system = System()

    # -----------------------------
    # 1. particles
    # -----------------------------
    for copy_idx in range(n_copies):
        for i in range(n_particles):
            full_system.addParticle(template_system.getParticleMass(i))

    # -----------------------------
    # 2. constraints
    # -----------------------------
    for copy_idx in range(n_copies):
        offset = copy_idx * n_particles
        for i in range(template_system.getNumConstraints()):
            p1, p2, dist = template_system.getConstraintParameters(i)
            full_system.addConstraint(p1 + offset, p2 + offset, dist)

    # -----------------------------
    # 3. forces
    # -----------------------------
    for force in template_system.getForces():

        # =========================================================
        # HarmonicBondForce
        # =========================================================
        if isinstance(force, HarmonicBondForce):
            new_force = HarmonicBondForce()
            new_force.setName(force.getName())
            new_force.setUsesPeriodicBoundaryConditions(
                force.usesPeriodicBoundaryConditions()
            )
            new_force.setForceGroup(force.getForceGroup())

            for copy_idx in range(n_copies):
                offset = copy_idx * n_particles
                for i in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(i)
                    new_force.addBond(p1 + offset, p2 + offset, length, k)

            full_system.addForce(new_force)

        # =========================================================
        # CustomAngleForce
        # =========================================================
        elif isinstance(force, CustomAngleForce):
            new_force = CustomAngleForce(force.getEnergyFunction())
            new_force.setName(force.getName())
            new_force.setForceGroup(force.getForceGroup())
            new_force.setUsesPeriodicBoundaryConditions(
                force.usesPeriodicBoundaryConditions()
            )

            # per-angle parameters
            for i in range(force.getNumPerAngleParameters()):
                new_force.addPerAngleParameter(force.getPerAngleParameterName(i))

            # global parameters
            for i in range(force.getNumGlobalParameters()):
                new_force.addGlobalParameter(
                    force.getGlobalParameterName(i),
                    force.getGlobalParameterDefaultValue(i)
                )

            # angles
            for copy_idx in range(n_copies):
                offset = copy_idx * n_particles
                for i in range(force.getNumAngles()):
                    p1, p2, p3, params = force.getAngleParameters(i)
                    new_force.addAngle(
                        p1 + offset,
                        p2 + offset,
                        p3 + offset,
                        params
                    )

            full_system.addForce(new_force)

        # =========================================================
        # PeriodicTorsionForce
        # =========================================================
        elif isinstance(force, PeriodicTorsionForce):
            new_force = PeriodicTorsionForce()
            new_force.setName(force.getName())
            new_force.setUsesPeriodicBoundaryConditions(
                force.usesPeriodicBoundaryConditions()
            )
            new_force.setForceGroup(force.getForceGroup())

            for copy_idx in range(n_copies):
                offset = copy_idx * n_particles
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    new_force.addTorsion(
                        p1 + offset,
                        p2 + offset,
                        p3 + offset,
                        p4 + offset,
                        periodicity,
                        phase,
                        k
                    )

            full_system.addForce(new_force)

        # =========================================================
        # CustomTorsionForce
        # =========================================================
        elif isinstance(force, CustomTorsionForce):
            new_force = CustomTorsionForce(force.getEnergyFunction())
            new_force.setName(force.getName())
            new_force.setForceGroup(force.getForceGroup())
            new_force.setUsesPeriodicBoundaryConditions(
                force.usesPeriodicBoundaryConditions()
            )

            # per-torsion parameters
            for i in range(force.getNumPerTorsionParameters()):
                new_force.addPerTorsionParameter(force.getPerTorsionParameterName(i))

            # global parameters
            for i in range(force.getNumGlobalParameters()):
                new_force.addGlobalParameter(
                    force.getGlobalParameterName(i),
                    force.getGlobalParameterDefaultValue(i)
                )

            # torsions
            for copy_idx in range(n_copies):
                offset = copy_idx * n_particles
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, params = force.getTorsionParameters(i)
                    new_force.addTorsion(
                        p1 + offset,
                        p2 + offset,
                        p3 + offset,
                        p4 + offset,
                        params
                    )

            full_system.addForce(new_force)

        # =========================================================
        # CustomNonbondedForce
        # =========================================================
        elif isinstance(force, CustomNonbondedForce):
            new_force = CustomNonbondedForce(force.getEnergyFunction())
            new_force.setName(force.getName())
            new_force.setForceGroup(force.getForceGroup())

            # settings
            new_force.setNonbondedMethod(force.getNonbondedMethod())
            new_force.setCutoffDistance(force.getCutoffDistance())
            new_force.setUseSwitchingFunction(force.getUseSwitchingFunction())
            if force.getUseSwitchingFunction():
                new_force.setSwitchingDistance(force.getSwitchingDistance())
            new_force.setUseLongRangeCorrection(force.getUseLongRangeCorrection())


            # global parameters
            for i in range(force.getNumGlobalParameters()):
                new_force.addGlobalParameter(
                    force.getGlobalParameterName(i),
                    force.getGlobalParameterDefaultValue(i)
                )

            # per-particle parameters
            for i in range(force.getNumPerParticleParameters()):
                new_force.addPerParticleParameter(
                    force.getPerParticleParameterName(i)
                )

            # tabulated functions, if any
            for i in range(force.getNumTabulatedFunctions()):
                fname = force.getTabulatedFunctionName(i)
                fobj = force.getTabulatedFunction(i)
                new_force.addTabulatedFunction(fname, copy.deepcopy(fobj))

            # add particles
            for copy_idx in range(n_copies):
                for i in range(n_particles):
                    params = force.getParticleParameters(i)
                    new_force.addParticle(params)

            # add exclusions for each copy
            for copy_idx in range(n_copies):
                offset = copy_idx * n_particles
                for i in range(force.getNumExclusions()):
                    p1, p2 = force.getExclusionParticles(i)
                    new_force.addExclusion(p1 + offset, p2 + offset)

            # IMPORTANT:
            # intramolecular only, each copy interacts only with itself
            for copy_idx in range(n_copies):
                start = copy_idx * n_particles
                stop = (copy_idx + 1) * n_particles
                group = list(range(start, stop))
                new_force.addInteractionGroup(group, group)

            full_system.addForce(new_force)

        else:
            raise NotImplementedError(
                f"Unsupported force type: {type(force)} with name '{force.getName()}'"
            )

    return full_system

def replicate_positions(template_positions, n_copies, shift=5.0 * nanometer):
    """
    Replicate positions n_copies times and shift each copy along x.

    Returns an OpenMM Quantity with the same distance unit as template_positions.
    """
    pos_unit = template_positions.unit
    pos_nm = template_positions.value_in_unit(nanometer)

    all_pos = []
    shift_nm = shift.value_in_unit(nanometer)

    for copy_idx in range(n_copies):
        new_pos = np.array(pos_nm, copy=True)
        new_pos[:, 0] += copy_idx * shift_nm
        all_pos.append(new_pos)

    full_pos_nm = np.vstack(all_pos)
    return Quantity(full_pos_nm, nanometer).value_in_unit(pos_unit) * pos_unit


def replicate_topology(template_topology, n_copies):
    """
    Replicate OpenMM topology n_copies times.
    """
    full_topology = Topology()
    box_vectors = template_topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        full_topology.setPeriodicBoxVectors(box_vectors)

    for _ in range(n_copies):
        atom_map = {}
        for chain in template_topology.chains():
            new_chain = full_topology.addChain(chain.id)
            for residue in chain.residues():
                new_residue = full_topology.addResidue(
                    residue.name,
                    new_chain,
                    id=residue.id,
                    insertionCode=residue.insertionCode,
                )
                for atom in residue.atoms():
                    atom_map[atom] = full_topology.addAtom(
                        atom.name,
                        atom.element,
                        new_residue,
                        id=atom.id,
                        formalCharge=atom.formalCharge,
                    )

        for bond in template_topology.bonds():
            full_topology.addBond(
                atom_map[bond.atom1],
                atom_map[bond.atom2],
                type=bond.type,
                order=bond.order,
            )

    return full_topology

usage = '\nUsage: python single_run.py -f control_file\n'
############## MAIN #################
ctrlfile = ''
if len(sys.argv) == 1:
    print(usage)
    sys.exit()

try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:", ["ctrlfile="])
except getopt.GetoptError:
    print(usage)
    sys.exit()
for opt, arg in opts:
    if opt == '-h':
        print(usage)
        sys.exit()
    elif opt in ("-f", "--ctrlfile"):
        ctrlfile = arg


if not os.path.exists(ctrlfile):
    print("Error: cannot find control file " + ctrlfile + ".")
    sys.exit()

cfg = read_control_file(ctrlfile)

_CONTROL_KEYS_REQUIRED = (
    "psffile",
    "corfile",
    "prmfile",
    "temp_heating",
    "temp_prod",
    "ppn",
    "outname",
    "traj_dir",
    "mdsteps",
    "heating_steps",
    "nstxout",
    "nstlog",
    "n_copies",
    "restart",
    "use_gpu",
)
_missing = [k for k in _CONTROL_KEYS_REQUIRED if k not in cfg]
if _missing:
    print("Error: control file missing keys: " + ", ".join(_missing))
    sys.exit(1)

psffile = cfg["psffile"]
corfile = cfg["corfile"]
prmfile = cfg["prmfile"]
temp_heating = float(cfg["temp_heating"]) * kelvin
temp_prod = float(cfg["temp_prod"]) * kelvin
ppn = str(cfg["ppn"])
outname = cfg["outname"]
traj_dir = cfg["traj_dir"]
mdsteps = int(cfg["mdsteps"])
heating_steps = int(cfg["heating_steps"])
nstxout = int(cfg["nstxout"])
nstlog = int(cfg["nstlog"])
n_copies = int(cfg["n_copies"])
restart = bool(strtobool(cfg["restart"]))
use_gpu = bool(strtobool(cfg["use_gpu"]))

# check if traj_dir exists, if not, create it
if not os.path.exists(traj_dir):
    os.makedirs(traj_dir, exist_ok=True)

timestep = 0.015 * picoseconds
fbsolu = 0.05 / picosecond

# Done for system initialization
psf = CharmmPsfFile(psffile)
psf_pmd = pmd.load_file(psffile)
cor = CharmmCrdFile(corfile)
forcefield = ForceField(prmfile)
top = psf.topology
# re-name residues that are changed by openmm
for resid, res in enumerate(top.residues()):
    if res.name != psf_pmd.residues[resid].name:
        res.name = psf_pmd.residues[resid].name
templete_map = {}
for chain in top.chains():
    for res in chain.residues():
        templete_map[res] = res.name

system = forcefield.createSystem(top, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0 * nanometer,
                                 constraints=AllBonds, removeCMMotion=False,
                                 ignoreExternalBonds=True, residueTemplates=templete_map)
# custom_nb_force = system.getForce(4)
for force in system.getForces():
    if force.getName() == 'CustomNonbondedForce':
        # custom_nb_force = force
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(1.8 * nanometer)


# prepare simulation
if use_gpu:
    print("Running simulation on CUDA device")
    dev_index = 0
    properties = {'CudaPrecision': 'mixed', "DeviceIndex": "%d" % dev_index}
    platform = Platform.getPlatformByName('CUDA')
else:
    print("Running simulation on CPU")
    properties = {'Threads': ppn}
    platform = Platform.getPlatformByName('CPU')

positions = cor.positions

print("Preparing temperature quenching of multiple chains...")
print(f"Number of chains in single simulations: {n_copies}")

# Prepare temperature quenching of multiple chains
full_system = replicate_cg_system_intra_only(system, n_copies)
full_topology = replicate_topology(top, n_copies)
full_positions = replicate_positions(positions, n_copies)
full_structure = pmd.openmm.load_topology(full_topology, system=full_system, xyz=full_positions)
full_structure.save(f"{traj_dir}/{outname}_n{n_copies}.psf", overwrite=True)

# Heating to high temperature
print(f"Heating to high temperature: {temp_heating} K for {heating_steps} steps")
integrator = LangevinIntegrator(temp_heating, fbsolu, timestep)
integrator.setConstraintTolerance(0.00001)
simulation = Simulation(full_topology, full_system, integrator, platform, properties)
simulation.context.setPositions(full_positions)
simulation.context.setVelocitiesToTemperature(temp_heating)
simulation.reporters = []
simulation.reporters.append(DCDReporter(f"{traj_dir}/{outname}_n{n_copies}_heating.dcd", nstxout, append=False))
simulation.reporters.append(
    StateDataReporter(f"{traj_dir}/{outname}_n{n_copies}_heating.log", nstlog, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
                        totalEnergy=True, temperature=True, speed=True, separator='\t'))
simulation.step(heating_steps)
# Get the final positions after heating
final_positions = simulation.context.getState(getPositions=True).getPositions()
# Quench to low temperature
print(f"Quenching to low temperature: {temp_prod} K for {mdsteps} steps")
integrator = LangevinIntegrator(temp_prod, fbsolu, timestep)
integrator.setConstraintTolerance(0.00001)
simulation = Simulation(full_topology, full_system, integrator, platform, properties)
simulation.context.setPositions(final_positions)
simulation.context.setVelocitiesToTemperature(temp_prod)
simulation.reporters = []
simulation.reporters.append(DCDReporter(f"{traj_dir}/{outname}_n{n_copies}_quench.dcd", nstxout, append=False))
simulation.reporters.append(
    StateDataReporter(f"{traj_dir}/{outname}_n{n_copies}_quench.log", nstlog, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
                        totalEnergy=True, temperature=True, speed=True, separator='\t'))
simulation.step(mdsteps)


