#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml
import pathlib
import pint
import scipy.stats
import sys
import math

from scipy.optimize import curve_fit

sys.path.append("..")
from UnitSystem import *
from Constants import Constants
from SpeciesStats import SpeciesStats


# global vars
fit_mode_lin = True
current_index = -1


def fit(xs, ys, ax=None, col="bl", label: str="", species: str="", lin_fit:bool=False):
	if lin_fit:
		return fit_lin(xs, ys, ax, col, label, species)
	else:
		return fit_exp(xs, ys, ax, col, label, species)


def fit_exp(xs, ys, ax=None, col="bl", label: str="", species: str=""):
	def e_func(x, a, b, c):
		return a * np.exp(-b * x) + c

	try:
		popt, _ = curve_fit(e_func, xs, ys)
		tau = 1 / popt[1]
		print(f"Fit {label} for {species}: {popt[0]: .2f} * exp**(-{popt[1]: .2f}*x)+{popt[2]: .2f}, tau={tau: .3f}")
		if ax is not None:
			ax.plot(xs, e_func(xs, *popt), label=fr"{species} fit ($\tau$={tau: .3f})", color=col)
	except Exception:
		print(f"Could not fit exp curve for {label} {species}")


def fit_lin(xs, ys, ax=None, col="bl", label: str="", species: str=""):
	def lin_func(x, k, d):
		return -k * x + d

	try:
		popt, _ = curve_fit(lin_func, xs, ys)
		tau = popt[0]
		print(f"Fit {label} for {species}: -{popt[0]: .2f} * x + {popt[1]: .2f}, tau={tau: .3f}")
		if ax is not None:
			ax.plot(xs, lin_func(xs, *popt), label=fr"{species} fit ($\tau$={tau: .3f})", color=col)
	except Exception:
		print(f"Could not fit linear curve for {label} {species}")

if __name__ == '__main__':
	default_dump_dir = "out/single_spec_2d_units/21_03_11_43_48"
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
	species_stats: dict[str, SpeciesStats] = {}
	with open(dump_dir_p / "particle_dump.csv", "r") as f:
		particle_cnt = int(f.readline().strip())
		sim_size = list(map(float, f.readline().strip().split()))
		non_dim_temp = float(f.readline().strip())

		m = QParse(cfg["sim"]["m"])
		sigma = QParse(cfg["sim"]["sigma"])
		tau_sim = QParse(cfg["sim"]["tau_sim"])
		characteristics(sigma, m, tau_sim)
		temperature = mag(non_dim_temp * m * sigma**2 / (tau_sim**2 * Constants.KB))
		print(f"Simulation temperature: {temperature} K, time unit {tau_sim.to("ps")}")

		iterations = 10#math.ceil(mag(QParse(cfg["setup"]["max_sim_time"]) / QParse(cfg["setup"]["dump_interval"]))) + 2
		print(f"Simulation ran for {iterations} iterations")

		for spec in cfg["species"]:
			species_stats[spec] = SpeciesStats(particle_cnt=particle_cnt, iterations=iterations)

		print(f"Loading and processing particle data")
		i = 0
		iteration = 0
		pos_scale = dim(1, "bohr").magnitude
		vel_scale = dim(1, "m/s").magnitude
		dims = 3#cfg["setup"]["dimensions"]
		for l in f:
			if iteration == iterations:
				break
			if i < particle_cnt:
				pinfo = l.strip().split()
				species = pinfo[0]
				values = list(map(float, pinfo[1:]))
				# m/s
				pspeed = np.linalg.norm(np.array(values[dims:2*dims]) * vel_scale)
				# bohr
#				ppos = np.array([float(pinfo[1]), float(pinfo[2]), float(pinfo[3])]) * pos_scale
				species_stats[species].particle_speeds[iteration, i] = pspeed
			elif i == particle_cnt + 2:
				i = -1
				iteration += 1
			i += 1

	print("Computing statistics")
	for species in species_stats.keys():
		species_stats[species].compute_stats(temperature, species, cfg)

	# Plotting and fitting
	ts = (np.arange(iterations) * tau_sim).to("ps")
	plot_info = [
		((0, 0), "chi2", "Chi2", "Chi2 [1]"),
		((0, 1), "ks_test", "Kolmogorov-Smirnov", "KS [1]"),
		((0, 2), "ad_test", "Anderson-Darling", "AD [1]"),
		((1, 0), "cvm_test", "Cramer-Von Mises", "CvM [1]")
	]

	fig, axs = plt.subplots(2, 3, figsize=(8, 10), num=f"{dump_dir_p.name}" )

	for key in species_stats.keys():
		titles = ["Chi2", "Kolmogorov-Smirnov", "Anderson-Darling", "Cramer-Von Mises"]
		ylabels = ["Chi2 [1]", "KS [1]", "AD [1]", "CmV [1]"]
		for i, pi in enumerate(plot_info):
			idx = pi[0]
			vals = getattr(species_stats[key], pi[1])
			vals = np.log(vals) if fit_mode_lin else vals
			ap = axs[idx].plot(ts, vals, 'x', label=f"{key}")
			col = ap[0].get_color()  # get color to reuse it for fitted function plot
			fit(ts.magnitude, vals, axs[idx], "r", species=key, lin_fit=fit_mode_lin)

			axs[idx].set_title(pi[2])
			axs[idx].set_xlabel(f"Sim. Time [{ts.units}]")
			axs[idx].set_ylabel(pi[3])
			axs[idx].legend(loc="upper right")
			axs[idx].grid(True)

	# Initialize variables for switching data
	species_name = list(species_stats.keys())[-1]
	species = cfg["species"][species_name]

	def update_plot(index):
		""" Update the second plot based on the current index. """
		prob_plot_idx = (1, 1)
		mbd_ax_idx = (1, 2)
		axs[prob_plot_idx].clear()
		axs[mbd_ax_idx].clear()

		specstat = species_stats[species_name]
		last_bin_heights = specstat.bin_heights[index]
		last_bin_speeds = specstat.bin_speeds[index]
		temperature = specstat.temperature
		particle_speeds = specstat.particle_speeds[index]

		bin_width = (max(last_bin_speeds) - min(last_bin_speeds)) / len(last_bin_speeds)

		# plot histogram bars for particle speeds
		axs[mbd_ax_idx].bar(last_bin_speeds, [x for x in last_bin_heights], width=bin_width * 0.9, edgecolor='black',
						alpha=0.7, label=species_name)

		mass = QParse(species["mass"])
		# define a fixed range for MBD to avoid scaling while stepping through plots
		mbd_x = np.linspace(0, np.max(specstat.bin_speeds) * 1.2, 300)
		mbd_y = SpeciesStats.maxwell_boltzmann(temperature, mass, 3, mbd_x)
		mbd_y2 = SpeciesStats.maxwell_boltzmann(temperature, mass, 2, mbd_x)

		# plot the reference MBD curves
		axs[mbd_ax_idx].plot(mbd_x, mbd_y, "r-", linewidth=2, label="MBD 3D")
		axs[mbd_ax_idx].plot(mbd_x, mbd_y2, "g-", linewidth=2, label="MBD 2D")
#		axs[mbd_ax_idx].set_ylim(0, 1.5*np.max(specstat.bin_heights))
		axs[mbd_ax_idx].set_xlabel("Bin Speeds [m/s]")
		axs[mbd_ax_idx].set_ylabel(f"Prob. Density [{1 if cfg['setup']['dimensions'] == 2 else 's/m'}]")
		axs[mbd_ax_idx].set_title(f"Part. speeds at {(index * tau_sim).to('ps').magnitude: .2f} / {ts.magnitude[-1]: .2f} {ts.units}")
		axs[mbd_ax_idx].grid(True)
		axs[mbd_ax_idx].legend()

		# Probability Plot
		scipy.stats.probplot(particle_speeds, dist="maxwell", plot=axs[1,1])
		axs[prob_plot_idx].set_title("Probability Plot (MBD)")
		axs[prob_plot_idx].grid(True)

		plt.draw()

	current_index = iterations-1

	def on_key(event):
		""" Handle key press events to switch between iterations. """
		global current_index
		if event.key == "right":
			current_index = (current_index + 1) % iterations  # Loop forward
		elif event.key == "left":
			current_index = (current_index - 1) % iterations  # Loop backward
		update_plot(current_index)


	# Initial plot
	update_plot(current_index)
	# Connect the key press event
	fig.canvas.mpl_connect("key_press_event", on_key)
	plt.tight_layout()
plt.show()

