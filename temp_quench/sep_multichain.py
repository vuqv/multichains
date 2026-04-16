from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from mdtraj.formats import DCDTrajectoryFile
from openmm import Vec3, unit
from openmm.app import CharmmPsfFile, DCDFile


# Defaults for your CG workflow
DEFAULT_DT_PS = 0.015   # 15 fs
ANGSTROM_TO_NM = 0.1
DEFAULT_CENTER = True   # center each copy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split a large combined multi-copy DCD trajectory into one DCD per copy "
            "using streaming input and OpenMM DCD writers."
        )
    )

    parser.add_argument(
        "-tp", "--template-psf",
        type=Path,
        required=True,
        help="Single-chain PSF used for atoms-per-copy and output topology.",
    )
    parser.add_argument(
        "-cp", "--combined-psf",
        type=Path,
        required=True,
        help="Combined multi-copy PSF.",
    )
    parser.add_argument(
        "-f", "--combined-dcd",
        type=Path,
        required=True,
        help="Combined trajectory DCD file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("split_trajs"),
        help="Output directory (default: split_trajs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Frames per chunk (default: 1000).",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable centering (centering is ON by default).",
    )

    return parser.parse_args()


def validate_file(path: Path) -> None:
    """Ensure a file exists."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


def center_xyz_inplace(xyz: np.ndarray) -> None:
    """Center coordinates independently per frame."""
    xyz -= xyz.mean(axis=1, keepdims=True)


def frame_to_positions(frame_xyz_angstrom: np.ndarray):
    """
    Convert coordinates from Å (DCD input) to OpenMM positions in nm.

    IMPORTANT
    ---------
    MDTraj's low-level DCD reader (DCDTrajectoryFile) returns coordinates
    in angstrom (Å), while OpenMM's DCDFile expects positions in nanometers (nm).

    Therefore, a unit conversion is required:
        1 Å = 0.1 nm

    If this conversion is omitted, the output trajectory will be inflated
    by a factor of 10 (coordinates interpreted as nm instead of Å).

    Parameters
    ----------
    frame_xyz_angstrom : np.ndarray, shape (n_atoms, 3)
        Atomic coordinates in angstrom.

    Returns
    -------
    openmm.unit.Quantity
        Positions in nanometers, suitable for OpenMM DCD writing.
    """
    ANGSTROM_TO_NM = 0.1
    frame_xyz_nm = frame_xyz_angstrom * ANGSTROM_TO_NM

    return [
        Vec3(float(x), float(y), float(z))
        for x, y, z in frame_xyz_nm
    ] * unit.nanometer


def open_dcd_writers(n_copies, output_dir, topology, dt_ps):
    """Create OpenMM DCD writers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    handles = []
    writers = []
    dt = dt_ps * unit.picoseconds

    for i in range(n_copies):
        path = output_dir / f"copy_{i:03d}.dcd"
        fh = open(path, "wb")
        writer = DCDFile(fh, topology, dt)
        handles.append(fh)
        writers.append(writer)

    return handles, writers


def split_dcd_streaming(
    template_psf: Path,
    combined_psf: Path,
    combined_dcd: Path,
    output_dir: Path,
    dt_ps: float = DEFAULT_DT_PS,
    chunk_size: int = 1000,
    center: bool = DEFAULT_CENTER,
) -> None:
    """
    Stream and split combined trajectory into per-copy DCD files.
    """
    template = CharmmPsfFile(str(template_psf))
    combined = CharmmPsfFile(str(combined_psf))

    atoms_per_copy = template.topology.getNumAtoms()
    total_atoms = combined.topology.getNumAtoms()

    if total_atoms % atoms_per_copy != 0:
        raise ValueError(
            f"Total atoms ({total_atoms}) not divisible by atoms_per_copy ({atoms_per_copy})"
        )

    n_copies = total_atoms // atoms_per_copy

    print(f"atoms_per_copy = {atoms_per_copy}")
    print(f"total_atoms = {total_atoms}")
    print(f"n_copies = {n_copies}")
    print(f"dt_ps = {dt_ps}")
    print(f"center = {center}")

    handles, writers = open_dcd_writers(
        n_copies, output_dir, template.topology, dt_ps
    )

    total_frames = 0

    try:
        with DCDTrajectoryFile(str(combined_dcd), "r") as dcd:
            while True:
                xyz, _, _ = dcd.read(n_frames=chunk_size)

                if xyz.shape[0] == 0:
                    break

                n_frames_chunk = xyz.shape[0]
                print(f"Processing frames {total_frames} → {total_frames + n_frames_chunk - 1}")

                for i in range(n_copies):
                    start = i * atoms_per_copy
                    end = start + atoms_per_copy

                    sub_xyz = xyz[:, start:end, :].copy()

                    if center:
                        center_xyz_inplace(sub_xyz)

                    writer = writers[i]

                    for f in range(n_frames_chunk):
                        writer.writeModel(frame_to_positions(sub_xyz[f]))

                total_frames += n_frames_chunk

    finally:
        for fh in handles:
            fh.close()

    print(f"Done. Total frames: {total_frames}")


def main() -> None:
    args = parse_args()

    validate_file(args.template_psf)
    validate_file(args.combined_psf)
    validate_file(args.combined_dcd)

    split_dcd_streaming(
        template_psf=args.template_psf,
        combined_psf=args.combined_psf,
        combined_dcd=args.combined_dcd,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        center=not args.no_center,   # default True unless explicitly disabled
    )


if __name__ == "__main__":
    main()