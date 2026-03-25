#!/usr/bin/env python3
"""Output SLURM array spec for missing episode files. Empty output = all complete."""

import sys
from pathlib import Path


def indices_to_slurm_spec(indices: list[int]) -> str:
    """Convert sorted list of ints to compact SLURM array spec (e.g. '0-5,8,10-12')."""
    if not indices:
        return ""
    ranges = []
    start = indices[0]
    end = start
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start}-{end}" if end > start else str(start))
            start = end = i
    ranges.append(f"{start}-{end}" if end > start else str(start))
    return ",".join(ranges)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <n_episodes>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    n_episodes = int(sys.argv[2])
    episodes_dir = output_dir / "episodes"

    missing = []
    for i in range(n_episodes):
        if not (episodes_dir / f"episode_{i:04d}.json").exists():
            missing.append(i)

    spec = indices_to_slurm_spec(missing)
    if spec:
        print(spec)


if __name__ == "__main__":
    main()
