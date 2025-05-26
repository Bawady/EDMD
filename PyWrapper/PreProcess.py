#!/usr/bin/env python

import sys
import os
import subprocess
import pathlib
import yaml

from datetime import datetime
from joblib import Parallel, delayed

from util.Simulator import MicroSimulator
from util.Constants import *


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

def log(msg: str):
	now = datetime.now()
	log_file.write(f"{now}: {msg}\n")

def info(msg: str):
	log(msg)
	print(msg)

def detail(msg: str):
	log(msg)

def error(msg: str):
	log(f"ERROR: {msg}")
	print(f"ERROR: {msg}")

def warning(msg: str):
	log(f"WARNING: {msg}")
	print(f"WARNING: {msg}")

def flush_log():
	if log_file is not None:
		log_file.flush()
		os.fsync(log_file.fileno())


def run_simulation(cfg_yml: str | pathlib.Path, out_p: str | pathlib.Path, seed: int):
	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()

	sim_out_p = out_p
	if seed is not None:
		sim_out_p = sim_out_p / str(seed)
	sim_out_p.mkdir()

	init_log(sim_out_p / "log")

	chara_x, chara_t, chara_m = Q(1, "nm"), Q(1, "ns"), Q(1e6, "u")
	characteristics(chara_x, chara_t, chara_m, Constants.KB)
	set_conversion_mode(ConversionMode.NON_DIM)
	info(f"Loading simulation configuration {cfg_yml}")

	if seed is not None:
		sim = MicroSimulator.from_yaml(cfg_yml, seed=seed)
	else:
		sim = MicroSimulator.from_yaml(cfg_yml)

	set_conversion_mode(ConversionMode.DIM)
	max_r = Q(0, "m")
	for spec in sim.species:
		if max_r < Q(sim.species[spec].radius, unit(chara_x)):
			max_r = Q(sim.species[spec].radius, unit(chara_x))
			biggest_spec = spec
	set_conversion_mode(ConversionMode.NON_DIM)

	temperature = sim.tree.species_temperature(biggest_spec)
	set_conversion_mode(ConversionMode.DIM)
	temperature = (temperature * chara_m * (chara_x / chara_t)**2).to_base_units()
	m = sim.species[biggest_spec].mass * chara_m
	sigma = 2 * max_r
	tau = (np.sqrt(m * sigma**2 / (Constants.KB * temperature))).to_base_units()
	info(f"Simulation characteristics: m={m} sigma={sigma} tau={tau.to('ps')}")

	transform_characteristics(sigma, m, tau, Constants.KB)
	info(f"Initial temperature: {temperature}")

	set_conversion_mode(ConversionMode.NON_DIM)
	init_file_p = sim_out_p / "particle_init.csv"
	sim.export_particles(init_file_p)

	info("Running EDMD simulation")

	time_transform_factor = 1 / non_dim(chara_t)
	max_sim_time = sim.max_sim_time * time_transform_factor
	dump_interval = sim.dump_interval * time_transform_factor

	with open(cfg_yml) as f:
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

	edmd_simulator_p = pathlib.Path(f"../Cell/{'2d' if cfg['setup']['dimensions'] == 2 else '3d'}")

	rel_init_file = os.path.relpath(init_file_p, edmd_simulator_p.parent)
	rel_out_dir = os.path.relpath(sim_out_p, edmd_simulator_p.parent)
	detail(f"Simulation call: ./{edmd_simulator_p} -f {rel_init_file} -o {rel_out_dir} -m {max_sim_time} -i {dump_interval} -s {cfg['setup']['seed']}")
	flush_log()
	sim_exit_result = subprocess.run(f"./{edmd_simulator_p} -f {rel_init_file} -o {rel_out_dir} -m {max_sim_time}, -i {dump_interval} -s {cfg['setup']['seed']}",
																	 shell=True, capture_output=True, text=True)

	detail(sim_exit_result.stdout)
	if sim_exit_result.returncode != 0:
		error("EDMD simulation failed with the following error:")
		error(sim_exit_result.stderr)
		raise SystemExit(sim_exit_result.returncode)
	info("EDMD simulation succeeded")

	close_log()


if __name__ == "__main__":
	yml = "neon_2d_units_lbm_cfg.yml"
	yml = sys.argv[1] if len(sys.argv) > 1 else yml
	out_p = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("/data/out")

	sim_start = "{date:%d_%m_%H_%M_%S}".format(date=datetime.now())
	sim_out_p = out_p / pathlib.Path(yml).stem / sim_start
	sim_out_p.mkdir(parents=True, exist_ok=True)

	rng = np.random.default_rng(seed=42)
	seeds = rng.integers(0, 2**32-1, 16)
	Parallel(n_jobs=len(seeds), backend="multiprocessing")(delayed(run_simulation)(yml, sim_out_p, int(seed)) for seed in seeds)
