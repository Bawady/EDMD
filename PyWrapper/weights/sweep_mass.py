#!/usr/bin/env python3
import subprocess
import itertools
import re
import numpy as np
import sys
import pathlib

# Parameters
mass_scales = [0.1 ,0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0]
config = sys.argv[1]
out_dir = sys.argv[2]

for mass_scale in mass_scales:
	try:
		# Run the command and capture output
		print(f"Running EDMD for {mass_scale=}")
		result = subprocess.run(
				f"python3 PreProcess.py {config} {out_dir} {mass_scale}",
				capture_output=True,
				text=True,
				check=True,
				shell=True
				)
		sim_out_p = pathlib.Path(out_dir) / pathlib.Path(config).stem / f"{mass_scale}_mass"
		print(f"Computing weights: python3 sweep_weights.py {sim_out_p} {sim_out_p / 'weight_sweep.csv'}")
		result = subprocess.run(
				f"python3 sweep_weights.py {sim_out_p} {sim_out_p / 'weight_sweep.csv'}",
				text=True,
				check=True,
				shell=True
				)
	except subprocess.CalledProcessError as e:
		print(f"Error ({e})")

