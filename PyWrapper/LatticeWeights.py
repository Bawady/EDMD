#!/usr/bin/env python
"""Lattice Weight Computation

Usage:
  LatticeWeights.py [options] <dx> <dt> <DIR>
  LatticeWeights.py (-h | --help)
  LatticeWeights.py --version

Options:
  -h --help    Show this screen.
  --version    Show version.
  --recompute  Recompute all weights.
  --plot       Plot the weights as bar graph.
"""
from docopt import docopt

import sys
import pathlib
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

sys.path.append("..")
from UnitSystem import *
from Constants import Constants


step_idx : int = 0


def compute_migration_kernel(pos_t, pos_t1, size):
	# D2Q9 -> 9 velocities
	migration_counts = np.zeros((size**2, 9))
	density = np.zeros(size**2)

	deltas = (pos_t1 - pos_t).astype(np.int16) % size
	half_size = size // 2
	deltas = (deltas + half_size) % size - half_size
	deltas = np.clip(deltas, -1, 1)  # ensure movement is max one cell in each direction -> minor inaccuracy if this doesn't happen too often

	for i in range(pos_t.shape[0]):
		from_x, from_y = pos_t[i]
		dx, dy = deltas[i]
		to_x, to_y = (from_x + dx) % size, (from_y + dy) % size
		cell_idx = int(to_x + to_y * size)
		# In D2Q9 the neighborhood is a 3x3 grid
		pop_idx = dx+1 + (dy+1) * 3
		migration_counts[cell_idx][pop_idx] += 1
		density[cell_idx] += 1

	density = density[:,np.newaxis]
	weights = np.zeros(9)
	for i in range(migration_counts.shape[0]):
		if density[i] > 0:
			weights += migration_counts[i]

	weights /= np.sum(density)
	# Reorder to rest weight, the "straight" weights and the diagonal weights
	return weights[[4, 5, 1, 3, 7, 2, 0, 6, 8]]


def compute_lattice_weigths(params: dict, sim_run_dir_p: pathlib.Path, out_dir_p):
	ppos = np.fromfile(sim_run_dir_p / "particle_positions.bin", dtype=np.float64)
	ppos = ppos.reshape(-1, params["particle_cnt"], params["dims"]) * params["pos_scale"]

	pids = np.fromfile(sim_run_dir_p / "pid.bin", dtype=np.uint32)
	pids = pids.reshape(-1, params["particle_cnt"])

	step_interval = params["step_interval"]

	iterations = (len(pids) - 2) // step_interval
	weights = np.zeros((iterations, 9))

	for t in range(iterations):
		pos = ppos[t*step_interval]
		cells = np.trunc(pos)# // params["sim_size"]

		pos_nxt = ppos[(t+1)*step_interval]
		cells_nxt = np.trunc(pos_nxt)# // params["sim_size"]
		weights_current_step = compute_migration_kernel(cells, cells_nxt, int(params["sim_size"]))
		weights[t] = weights_current_step

	np.save(out_dir_p / f"{sim_run_dir_p.name}.npy", weights)
	return np.mean(weights, axis=0)


def get_simulation_runs(dump_dir_p : pathlib.Path) -> list[str]:
	sim_runs = []
	# If the dump directory contains a config file this is not an ensemble simulation
	if (dump_dir_p / "config.yml").exists():
		sim_runs.append(dump_dir_p)
	else:
		for sub_dir in dump_dir_p.iterdir():
			# Every ensemble run contains an EDMD configuration file
			if sub_dir.is_dir() and (sub_dir / "config.yml").exists():
				sim_runs.append(sub_dir)
	if len(sim_runs) == 0:
		print(f"Neither the dump directory ({dump_dir_p}) nor any of its sub directories contains an EDMD configuration file")
		exit(1)
	return sim_runs


def plot_weights(weights: np.ndarray) -> None:
	fig, ax = plt.subplots()
	bars = ax.bar(range(9), weights[0])
	ax.set_title(f"Weights at 0")
	ax.set_ylim(ws.min(), ws.max())  # Optional: fix y-axis range

	def on_key(event):
		if not hasattr(on_key, "idx"):
			on_key.idx = 0

		if event.key == "right":
			on_key.idx = (on_key.idx + 1) % ws.shape[0]
		elif event.key == "left":
			on_key.idx = (on_key.idx - 1) % ws.shape[0]
		else:
			return

		for bar, height in zip(bars, ws[on_key.idx]):
			bar.set_height(height)
		ax.set_title(f"Weights at {on_key.idx}")
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect('key_press_event', on_key)
	plt.show()


def load_edmd_configuration(edmd_dump_dir_p: pathlib.Path) -> dict:
	cfg = {}
	try:
		with open(edmd_dump_dir_p / "config.yml", "r") as cfg_yml:
			cfg = yaml.safe_load(cfg_yml)
			species = cfg["species"]
			spec_dict = {}
			for s in species:
				spec_dict[s["name"]] = s
			cfg["species"] = spec_dict
	except FileNotFoundError:
		# This cannot happen if the edmd_dump_dir_p originates from get_simulation_runs, but who knows...
		print(f"Could not find config.yml in {edmd_dump_dir_p}")
		exit(1)
	return cfg


def compute_discretization_parameters(edmd_configuration: dict, dx: str, dt: str) -> dict:
	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()

	m = QParse(edmd_configuration["sim"]["m"])
	sigma = QParse(edmd_configuration["sim"]["sigma"])
	tau_sim = QParse(edmd_configuration["sim"]["tau_sim"])
	characteristics(sigma, m, tau_sim)

	particle_cnt: int = 0
	for species in edmd_configuration["species"]:
		particle_cnt += cfg["species"][species]["quantity"]

	dx_lbm = QParse(dx)
	dt_lbm = QParse(dt)
	step_interval = (dt_lbm / QParse(cfg["setup"]["dump_interval"])).to_base_units().magnitude
	pos_scale = 1 / non_dim(dx_lbm)
	sim_size = non_dim(QParse(cfg["setup"]["size"])) * pos_scale

	params = {}
	params["dims"] = cfg["setup"]["dimensions"]
	params["pos_scale"] = pos_scale
	params["sim_size"] = round(sim_size)
	params["particle_cnt"] = particle_cnt
	params["step_interval"] = int(step_interval)

	return params


def load_precomputed_weights(weight_dump_p: pathlib.Path, avg: bool=True) -> dict[str, np.ndarray]:
	ws = {}
	for x in weight_dump_p.iterdir():
		if x.is_file():
			ws[x.name] = np.load(x)
		if avg:
			ws[x.name] = ws[x.name].sum(axis=0) / ws[x.name].shape[0]
	return ws


if __name__ == "__main__":
	args = docopt(__doc__, version="1.0")
	dump_dir_p = pathlib.Path(args["<DIR>"])
	weight_dump_p = dump_dir_p / "weights"

	sim_runs : list[str] = get_simulation_runs(dump_dir_p)
	weights  : dict[str, np.ndarray] = load_precomputed_weights(weight_dump_p) if not args["--recompute"] else {}

	if len(sim_runs) != len(weights):
		# Compute weights that were not yet precomputed
		todo_sim_runs = [sim_run for sim_run in sim_runs if sim_run not in weights]
		cfg = load_edmd_configuration(pathlib.Path(todo_sim_runs[0]))
		if QParse(args["<dt>"]) < QParse(cfg["setup"]["dump_interval"]):
			print(f"Cannot compute weights for a time step ({args['<dt>']}) smaller than the EDMD dump interval ({cfg['setup']['dump_interval']})")
			exit(1)
		params = compute_discretization_parameters(cfg, args["<dx>"], args["<dt>"])

		weight_dump_p.mkdir(parents=True, exist_ok=True)
		print("Computing lattice weights")
		ws = Parallel(n_jobs=len(sim_runs), backend="multiprocessing")(delayed(compute_lattice_weigths)(params, run, weight_dump_p) for run in todo_sim_runs)
		for i, sim_run in enumerate(todo_sim_runs):
			weights[sim_run] = ws[i]

	avg_weights = np.zeros(9)
	for w in weights.values():
		avg_weights += w
	avg_weights /= len(sim_runs)
	print(f"Avg. weights: {avg_weights}, sum: {np.sum(avg_weights)}")

	if args["--plot"]:
		ws = load_precomputed_weights(weight_dump_p, avg=False)
		ensemble_ws_sum = None
		for w in ws.values():
			if ensemble_ws_sum is None:
				ensemble_ws_sum = w
			else:
				ensemble_ws_sum += w
		plot_weights(ensemble_ws_sum / len(ws))
