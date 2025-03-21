#!/bin/python3

import sys
import os
import math
import subprocess
import pathlib
import yaml

from datetime import datetime

from Simulator import MicroSimulator
from UnitSystem import *
from Constants import *


log_file = None

def init_log(path: os.PathLike):
	global log_file
	if log_file is not None:
		log_file.close()
	log_file = open(path, 'w')

def close_log():
	global log_file
	if log_file is not None:
		log_file.close()
		log_file = None

def info(msg: str):
	log_file.write(f"{msg}\n")
	print(msg)

def detail(msg: str):
	log_file.write(f"{msg}\n")

def error(msg: str):
	log_file.write(f"ERROR: {msg}\n")
	print(msg)

def warning(msg: str):
	log_file.write(f"WARNING: {msg}\n")
	print(msg)


if __name__ == "__main__":
	default_yml = "single_spec_2d_units.yml"
	cfg_ymls = sys.argv[1:] if len(sys.argv) > 1 else [default_yml]

	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()
	out_p = pathlib.Path("out")

	for yml in cfg_ymls:
		sim_start = "{date:%d_%m_%H_%M_%S}".format(date=datetime.now())
		sim_out_p = out_p / pathlib.Path(yml).stem / sim_start
		sim_out_p.mkdir(parents=True, exist_ok=True)

		init_log(sim_out_p / "log")

		info(f"Loading simulation configuration {yml}")
		sim = MicroSimulator.from_yaml(yml)
		max_r = Q(0, "m")
		for spec in sim.species:
			if max_r < sim.species[spec].radius:
				max_r = sim.species[spec].radius
				biggest_spec = spec
		temperature = sim.tree.species_temperature(biggest_spec)
		m = sim.species[biggest_spec].mass
		sigma = 2 * max_r
		tau = np.sqrt(m * sigma**2 / (Constants.KB * temperature))
		info(f"Simulation characteristics: m={m} sigma={sigma} tau={tau.to('ps')}")
		characteristics(sigma, sim.species[biggest_spec].mass, tau, Constants.KB)
		info(f"Initial temperature: {temperature}")

		set_conversion_mode(ConversionMode.NON_DIM)
		init_file_p = sim_out_p / "particle_init.csv"
		sim.export_particles(init_file_p)

		info("Running EDMD simulation")

		max_sim_time = non_dim(sim.max_sim_time)
		dump_interval = non_dim(sim.dump_interval)

		with open(yml) as f:
			cfg = yaml.safe_load(f)
			# add sim params that got computed depending on the config for documentation / reproducibility
			cfg["sim"] = {}
			cfg["sim"]["temperature"] = str(temperature)
			# non dimensionalization quantities
			cfg["sim"]["sigma"] = str(sigma)
			cfg["sim"]["m"] = str(m)
			cfg["sim"]["tau_sim"] = str(tau.to('ps'))
			with open(sim_out_p / "config.yml", "w") as out_f:
				yaml.dump(cfg, out_f, default_flow_style=False)

		sim_cfg_yml = sim_out_p / "config.yml"

#		edmd_simulator_p = pathlib.Path(f"../Cell/{'2d' if cfg['setup']['dimensions'] == 2 else '3d'}")
		edmd_simulator_p = pathlib.Path("../Cell/3d")

		rel_init_file = os.path.relpath(init_file_p, edmd_simulator_p.parent)
		out_file_p = sim_out_p  / "particle_dump.csv"
		rel_out_file = os.path.relpath(out_file_p, edmd_simulator_p.parent)
		print(f"./{edmd_simulator_p} -f {rel_init_file} -o {rel_out_file} -m {max_sim_time}, -i {dump_interval} -s {cfg['setup']['seed']}")
		sim_exit_result = subprocess.run(f"./{edmd_simulator_p} -f {rel_init_file} -o {rel_out_file} -m {max_sim_time}, -i {dump_interval} -s {cfg['setup']['seed']}",
																	 shell=True, capture_output=True, text=True)

		detail(sim_exit_result.stdout)
		if sim_exit_result.returncode != 0:
			error("EDMD simulation failed with the following error:")
			error(sim_exit_result.stderr)
			raise SystemExit(sim_exit_result.returncode)
		info("EDMD simulation succeeded")

		close_log()