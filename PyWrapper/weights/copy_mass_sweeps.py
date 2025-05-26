#!/usr/bin/env python

from pathlib import Path
import shutil
import sys

def copy_weight_sweep_files(source_dir: Path, target_dir: Path):
	# Ensure the target directory exists
	target_dir.mkdir(parents=True, exist_ok=True)

	# Iterate through all subdirectories in the source directory
	for subdir in source_dir.iterdir():
		if subdir.is_dir():
			weight_file = subdir / "weight_sweep.csv"
			if weight_file.exists():
				new_filename = f"{subdir.name}_weight_sweep.csv"
				target_file = target_dir / new_filename
				shutil.copy(weight_file, target_file)
				print(f"Copied: {weight_file} -> {target_file}")
			else:
				print(f"No 'weight_sweep.csv' in {subdir}")

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python script.py <source_directory> <target_directory>")
	else:
		source = Path(sys.argv[1])
		target = Path(sys.argv[2])
		if not source.exists() or not source.is_dir():
			print(f"Source directory '{source}' does not exist or is not a directory.")
		else:
			copy_weight_sweep_files(source, target)

