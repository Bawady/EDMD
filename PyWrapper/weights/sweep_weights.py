#!/usr/bin/env python3
import subprocess
import itertools
import re
import numpy as np
import sys

# Parameters
sweep_range = np.arange(20, 32.1, 4)
dx_list = [f"{x} nm" for x in sweep_range]
dt_list = ["4 ps"] #[f"{x} ps" for x in sweep_range]
directory = sys.argv[1]
out_file = sys.argv[2]

with open(out_file, "a") as out_file:
	for dx, dt in itertools.product(dx_list, dt_list):
		try:
			# Run the command and capture output
			print(f"Running for {dx} {dt}")
			result = subprocess.run(
					f"python3 LatticeWeights.py --recompute --threads=8 '{dx}' '{dt}' {directory}",
					capture_output=True,
					text=True,
					check=True,
					shell=True
					)
			# Extract Avg. weights using regex
			match = re.search(r"Avg\. weights:\s*\[([^\]]+)\]", result.stdout)
			if match:
				weights_str = match.group(1)
			# Optionally round values
				weights = [f"{float(w):.4f}" for w in weights_str.strip().split()]
				weights_joined = ", ".join(weights)
			else:
				weights_joined = "N/A"
		except subprocess.CalledProcessError as e:
			weights_joined = f"Error ({e})"

		out_file.write(f"{dx},{dt},{weights_joined}\n")

