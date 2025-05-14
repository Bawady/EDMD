#!/usr/bin/env python

import csv

import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import yaml
import math

sys.path.append("..")
from UnitSystem import *
from Constants import Constants


ENSEMBLE = True


if __name__ == "__main__":
	default_dump_dir = "/data/out/2ps_step"
	dump_dir = sys.argv[1] if len(sys.argv) > 1 else default_dump_dir
	dump_dir_p = pathlib.Path(dump_dir)

	dump_dirs_p : list[pathlib.Path] = []
	if ENSEMBLE:
		for sim_dump in pathlib.Path(dump_dir).iterdir():
			if sim_dump.is_dir():
				dump_dirs_p.append(sim_dump)
	else:
		dump_dirs_p.append(pathlib.Path(dump_dir))

	set_conversion_mode(ConversionMode.DIM)
	Constants.prepare_constants()

	cfg = {}
	with open(dump_dirs_p[0] / "config.yml", "r") as cfg_yml:
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

	dx_lbm = mag(non_dim(Q(10, "nm")))
	sim_size = mag(non_dim(QParse(cfg["setup"]["size"])))
	grid_size = round(sim_size / dx_lbm)

	dump_interval = QParse(cfg["setup"]["dump_interval"])
	dump_count = math.ceil(mag(QParse(cfg["setup"]["max_sim_time"]) / dump_interval)) + 1
	dims = cfg["setup"]["dimensions"]

	if (dump_dir_p / "densities.npy").is_file():
		print("Found cached densities, loading")
		density_plots = np.load(dump_dir_p / "densities.npy")
#		np.save(pathlib.Path(dump_dir) / "edmd_dens_mask.npy", density_plots[0])
	else:
		density_plots = np.zeros((dump_count, grid_size, grid_size))
		cached = np.zeros((dump_count), dtype=np.bool)

		print("Loading particle positions")
		for dump_dir_p in dump_dirs_p:
			print(f"Loading dump {dump_dir_p}")
			ppos = np.fromfile(dump_dir_p / "particle_positions.bin", dtype=np.float64)
			ppos = ppos.reshape(-1, particle_cnt, dims)

			for iteration in range(dump_count):
				pos = ppos[iteration]
				cells = np.floor_divide(pos, mag(non_dim(dx_lbm))).astype(np.int32)		# Flatten (x, y) to single indices
				flat_ids = cells[:, 0] * grid_size + cells[:, 1]

				# Count occurrences of cells, i.e., discretize particle positions
				counts = np.bincount(flat_ids, minlength=grid_size * grid_size)

				# Reshape back to 2D grid
				plot = density_plots[iteration]
				plot[:, :] += counts.reshape((grid_size, grid_size))
	#			cached[iteration] = True

		density_plots[iteration] /= np.max(density_plots[0])
		np.save(pathlib.Path(dump_dir) / "densities.npy", density_plots)


fig, axs = plt.subplots(1, 2)
im = axs[0].imshow(density_plots[0], cmap='viridis')
axs[0].set_title(f"Density at {dump_interval * 0}")
axs[0].axes.get_xaxis().set_ticks([])
axs[0].axes.get_yaxis().set_ticks([])
center = density_plots[0].shape[0] // 2
line, = axs[1].plot(density_plots[0][center, :])
axs[1].set_title(f"Center Line Slice {dump_interval * 0}")

def on_key(event):
	if not hasattr(on_key, "idx"):
		on_key.idx = 0

	if event.key == "right":
		on_key.idx = (on_key.idx + 1) % dump_count
	elif event.key == "left":
		on_key.idx = (on_key.idx - 1) % dump_count
	else:
		return  # Ignore other keys

	im.set_data(density_plots[on_key.idx])
	line.set_data(range(density_plots[on_key.idx].shape[1]),
	              density_plots[on_key.idx][center, :])
	axs[0].set_title(f"Density at {on_key.idx * dump_interval}")
	axs[1].set_title(f"Center Line Slice at {on_key.idx * dump_interval}")
	fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
