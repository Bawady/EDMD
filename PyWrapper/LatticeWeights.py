#!/usr/bin/env python

import csv
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import yaml
import math

from collections import defaultdict
from joblib import Parallel, delayed

sys.path.append("..")
from UnitSystem import *
from Constants import Constants

step_idx : int = 0


def read_csv(filename, delimiter=';', size=100):
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter=delimiter)
		data = list(reader)

	assert len(data) % 3 == 0, "CSV file does not contain triplets of rows."

	steps = len(data) // 3
	simulation_data = []

	for i in range(steps):
		x_coords = list(map(float, data[3 * i]))
		y_coords = list(map(float, data[3 * i + 1]))
		particle_ids = list(map(int, data[3 * i + 2]))

		simulation_data.append((x_coords, y_coords, particle_ids))

	return simulation_data


def determine_region(pos, size=100):
	region_size = size / 3
	col = int(pos[0] // region_size)
	row = int(pos[1] // region_size)
	return (row, col)


def track_migrations(ppos : np.ndarray, ids : np.ndarray, size : float = 100, step_interval : int = 1):
	migrations_per_step = []
	region_labels = {
		(-1, -1): "TL", (-1, 0): "T", (-1, 1): "TR",
		(0, -1): "L", (0, 0): "Stay", (0, 1): "R",
		(1, -1): "BL", (1, 0): "B", (1, 1): "BR"
	}

	for t in range(len(ids) // step_interval - 1):
		pos = ppos[t * step_interval]
		id = ids[t * step_interval]
		pos_nxt = ppos[(t+1) * step_interval]
		id_nxt = ids[(t+1) * step_interval]

		current_positions = {pid: determine_region(pos, size) for pos, pid in zip(pos, id)}
		next_positions = {pid: determine_region(pos, size) for pos, pid in zip(pos_nxt, id_nxt)}

		migrations = defaultdict(int)

		for pid in id:
			if pid in next_positions:
				from_region = current_positions[pid]
				to_region = next_positions[pid]
				if from_region == (1, 1):  # Only track migrations from (1,1)
					delta = (to_region[0] - 1, to_region[1] - 1)  # Offset relative to (1,1)
					label = region_labels.get(delta, "Unknown")
					migrations[label] += 1

		migrations_per_step.append(migrations)

	return migrations_per_step


def compute_migration_kernel(pos_t, pos_t1, size):
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
		pop_idx = dx+1 + (dy+1) * 3
		migration_counts[cell_idx][pop_idx] += 1
		density[cell_idx] += 1

	density = density[:,np.newaxis]
	weights = np.zeros(9)
	for i in range(migration_counts.shape[0]):
		if density[i] > 0:
			weights += migration_counts[i]

	weights /= np.sum(density)
	return weights[[4, 5, 1, 3, 7, 2, 0, 6, 8]]


def compute_lattice_weigths(params: dict, sim_run_dir_p: pathlib.Path, out_dir_p):

	print("Loading particle positions")
	ppos = np.fromfile(sim_run_dir_p / "particle_positions.bin", dtype=np.float64)
	ppos = ppos.reshape(-1, params["particle_cnt"], params["dims"]) * params["pos_scale"]

	print("Loading particle ids")
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

	np.save(out_dir_p / f"{sim_run_dir_p.name}_weights.npy", weights)
	return np.mean(weights, axis=0)


if __name__ == "__main__":
	default_dump_dir = ("out/18_04_12_21_43")
	dump_dir = sys.argv[1] if len(sys.argv) > 1 else default_dump_dir
	dump_dir_p = pathlib.Path(dump_dir)

	# This is supposed to be quick hack on not the final state of how drawing is triggered
	if len(sys.argv) > 2:
		ws = None
		sims = 0
		for x in (dump_dir_p / "weights").iterdir():
			if x.is_file():
				sims += 1
				if ws is None:
					ws = np.load(x)
				else:
					ws += np.load(x)
		ws /= sims
		print(ws.shape)

		fig, ax = plt.subplots()
		bars = ax.bar(range(9), ws[0])
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
		exit(0)


	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()

	sim_runs = []
	for sub_dir in dump_dir_p.iterdir():
		if sub_dir.is_dir() and sub_dir.name.isdigit():
			sim_runs.append(sub_dir)
	if len(sim_runs) == 0:
		sim_runs.append(dump_dir_p)


	cfg = {}
	with open(sim_runs[0] / "config.yml", "r") as cfg_yml:
		cfg = yaml.safe_load(cfg_yml)
		species = cfg["species"]
		spec_dict = {}
		for s in species:
			spec_dict[s["name"]] = s
		cfg["species"] = spec_dict

	print(f"Loading simulation dump and determining parameters")
	m = QParse(cfg["sim"]["m"])
	sigma = QParse(cfg["sim"]["sigma"])
	tau_sim = QParse(cfg["sim"]["tau_sim"])
	characteristics(sigma, m, tau_sim)

	particle_cnt: int = 0
	for s in cfg["species"]:
		particle_cnt += cfg["species"][s]["quantity"]

	max_iterations = math.ceil(mag(QParse(cfg["setup"]["max_sim_time"]) / QParse(cfg["setup"]["dump_interval"]))) + 2

	dx_lbm = Q(10, "nm")
	dt_lbm = Q(10, "ps")
	step_interval = (dt_lbm / QParse(cfg["setup"]["dump_interval"])).to_base_units().magnitude
	pos_scale = 1 / non_dim(dx_lbm)
	sim_size = non_dim(QParse(cfg["setup"]["size"])) * pos_scale

	sim_out_p = dump_dir_p / "weights"
	sim_out_p.mkdir(parents=True, exist_ok=True)

	params = {}
	params["dims"] = cfg["setup"]["dimensions"]
	params["pos_scale"] = pos_scale
	params["sim_size"] = round(sim_size)
	params["particle_cnt"] = particle_cnt
	params["step_interval"] = int(step_interval)

	weights = Parallel(n_jobs=len(sim_runs), backend="multiprocessing")(delayed(compute_lattice_weigths)(params, run, 	sim_out_p) for run in sim_runs)
	avg_weights = np.zeros(9)
	for w in weights:
		avg_weights += w
	avg_weights /= len(sim_runs)
	print(avg_weights)

