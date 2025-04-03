#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import yaml
import math

from collections import defaultdict

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
		pos = ppos[t]
		id = ids[t]
		pos_nxt = ppos[t * step_interval]
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


def plot_migrations(migrations_per_step, size):
	global step_idx
	fig, ax = plt.subplots()
	categories = ["TL", "T", "TR", "L", "Stay", "R", "BL", "B", "BR"]

	def update_plot():
		global step_idx
		ax.clear()
		counts = migrations_per_step[step_idx]
		values = [counts.get(cat, 0) for cat in categories]
		values = [v / sum(values) for v in values]
		ax.bar(categories, values)
		ax.set_title(f"Migrations from (1,1) - Step {step_idx * tau_lbm} -> {(step_idx+1) * tau_lbm}")
		ax.set_ylabel("Particle Count")
		plt.draw()

	def on_key(event):
		global step_idx
		if event.key == "right" and step_idx < len(migrations_per_step) - 1:
			step_idx += 1
			update_plot()
		elif event.key == "left" and step_idx > 0:
			step_idx -= 1
			update_plot()

	fig.canvas.mpl_connect("key_press_event", on_key)
	update_plot()
	plt.show(block=True)


if __name__ == "__main__":
	default_dump_dir = "out/neon_2d_units/28_03_10_39_40"
	dump_dir = sys.argv[1] if len(sys.argv) > 1 else default_dump_dir
	dump_dir_p = pathlib.Path(dump_dir)

	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()

	cfg = {}
	with open(dump_dir_p / "config.yml", "r") as cfg_yml:
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

	tau_lbm = Q(10, "ns")
	step_interval = int(mag(tau_lbm / QParse(cfg["setup"]["dump_interval"])))

	print(f"Loading particle data")
	i = 0
	iteration = 0
	pos_scale = dim(1, "bohr").magnitude
	sim_size = mag(non_dim(QParse(cfg["setup"]["size"]))) * pos_scale
	dims = cfg["setup"]["dimensions"]

	print("Loading particle velocities")
	ppos = np.fromfile(dump_dir_p / "particle_positions.bin", dtype=np.float64)
	ppos = ppos.reshape(-1, particle_cnt, dims) * pos_scale

	pids = np.fromfile(dump_dir_p / "dbg.bin", dtype=np.uint16)
	pids = pids.reshape(-1, particle_cnt)

	migrations_per_step = track_migrations(ppos, pids, size=sim_size, step_interval=step_interval)
	plot_migrations(migrations_per_step, sim_size)
