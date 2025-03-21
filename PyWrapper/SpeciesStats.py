import sys
import numpy as np
import scipy.stats

from dataclasses import dataclass, field
from typing import List, Tuple, Callable

sys.path.append("..")
from UnitSystem import *
from Constants import Constants


@dataclass
class SpeciesStats:
	particle_cnt : int
	iterations   : int

	particle_speeds : np.ndarray = field(init=False)
	chi2            : np.ndarray = field(init=False)
	tvd             : np.ndarray = field(init=False)
	ks_test         : np.ndarray = field(init=False)
	ad_test         : np.ndarray = field(init=False)
	cvm_test        : np.ndarray = field(init=False)
	bin_heights     : np.ndarray = field(init=False)
	bin_speeds      : np.ndarray = field(init=False)

	def __post_init__(self):
		self.particle_speeds = np.zeros((self.iterations, self.particle_cnt), dtype=np.float64)
		self.chi2        = np.zeros(self.iterations, dtype=np.float64)
		self.tvd         = np.zeros(self.iterations, dtype=np.float64)
		self.ad_test     = np.zeros(self.iterations, dtype=np.float64)
		self.ks_test     = np.zeros(self.iterations, dtype=np.float64)
		self.cvm_test    = np.zeros(self.iterations, dtype=np.float64)
		self.bin_speeds  = np.zeros((self.iterations, 80), dtype=np.float64)
		self.bin_heights = np.zeros((self.iterations, 80), dtype=np.float64)

	def compute_stats(self, temperature: float, species_name: str, sim_info: dict) -> None:
		self.temperature = temperature
		species = sim_info["species"][species_name]
		dims = sim_info["setup"]["dimensions"]
		mass = mag(QParse(species["mass"]))
		cdf = SpeciesStats.maxwell_boltzmann_cdf(temperature, mass, dims)

		for j, iter_speeds in enumerate(self.particle_speeds):
			bin_heights, bin_edges = np.histogram(iter_speeds, 80, density=True)
			bin_speeds = (bin_edges[:-1] + bin_edges[1:]) / 2
			self.bin_speeds[j] = bin_speeds
			self.bin_heights[j] = bin_heights

			expected_speeds = np.array([self.maxwell_boltzmann(temperature, mass, dims, s) for s in bin_speeds])

			self.chi2[j] = self.chi_square(bin_heights, expected_speeds)
			self.tvd[j] = self.total_variation_distance(bin_heights, expected_speeds)
			self.ks_test[j] = self.kolmogorov_smirnov(iter_speeds, cdf)
			self.ad_test[j] = self.anderson_darling(iter_speeds, cdf)
			self.cvm_test[j] = self.cramer_von_mises(iter_speeds, cdf)

	@staticmethod
	def chi_square(samples: np.ndarray, expected: np.ndarray) -> float :
		ks_results = scipy.stats.chisquare(samples, expected, sum_check=False)
		return ks_results.statistic#, ks_results.pvalue

	@staticmethod
	def total_variation_distance(samples: np.ndarray, expected: np.ndarray) -> float:
		return sum(abs(np.array(expected) - np.array(samples))) / 2.0

	@staticmethod
	def kolmogorov_smirnov(samples: np.ndarray, ref_cfd: Callable) -> float:
		ks_results = scipy.stats.ks_1samp(samples, ref_cfd)
		return ks_results.statistic#, ks_results.pvalue

	@staticmethod
	def cramer_von_mises(samples: np.ndarray, ref_cfd: Callable) -> float:
		cvm_results = scipy.stats.cramervonmises(samples, ref_cfd)
		return cvm_results.statistic#, cvm_results.pvalue

	@staticmethod
	def anderson_darling(samples: np.ndarray, ref_cdf: Callable) -> float:
		sorted_samples = np.sort(samples)
		cdfs = ref_cdf(sorted_samples)
		N = len(samples)
		S = np.sum((2 * np.arange(1, N + 1) - 1) * (np.log(cdfs) + np.log(1 - cdfs[::-1])))
		return -N - S / N

	@staticmethod
	def maxwell_boltzmann(temperature: float, mass: float, dims: int, speeds: np.ndarray) -> 	np.ndarray:
		scale = np.sqrt(mag(Constants.KB) * temperature / mag(mass))
		if dims == 2:
			return scipy.stats.maxwell.pdf(speeds, scale=scale)
		else:
			return scipy.stats.rayleigh.pdf(speeds, scale=scale)

	@staticmethod
	def maxwell_boltzmann_cdf(temperature, mass, dims):
		# Wrapper function that returns the CDF
		def cdf(x):
			scale = np.sqrt(Constants.KB * temperature / mass)
			if dims == 2:
				return scipy.stats.rayleigh.cdf(x, scale=scale)
			else:
				return scipy.stats.maxwell.cdf(x, scale=scale)
		return cdf  # Return the function to be used in kstest
