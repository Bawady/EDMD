import random
import math
import numpy as np
import pathlib
import yaml
import warnings
import pathlib

from os import PathLike

from Particle import *
from Species import Species
from ParticleTree import ParticleTree
from UnitSystem import *



class MicroSimulator:
	"""Molecular Dynamics simulator for multi-species particle collisions.
		 Supports elastic non-reactive collisions between two particles.
		 To run a simulation, register the required chemical species (register_species), 'populate*' the simulation and 'run' it.
		 Or load all from a yaml configuration file (from_yaml) before running the simulation.
	"""
	colls = 0
	time = 0
	def __init__(self, size: QFloat, dimensions: int, dump_interval: str, max_sim_time: str,
	             seed: int = 42, max_speed: QFloat = Q(0, "m/s")):
		if dimensions != 2 and dimensions !=3:
			raise Exception(f"Only 2- and 3-dimensional simulations supported, but passed dimensionality is {dimensions}")
		self.dims      : QArray = list_to_qarr([size for _ in range(dimensions)])
		self.species   : dict[str, Species] = {}
		self.tree      : ParticleTree = ParticleTree(np.zeros(len(self.dims)), self.dims[0], self.dims[0])

		self.dump_interval : QFloat = QParse(dump_interval)
		self.max_sim_time : QFloat = QParse(max_sim_time)

		self.seed       : int = seed
		self.max_speed  : QFloat = QParse(max_speed) if type(max_speed) is str else max_speed

		random.seed(seed)
		np.random.seed(seed)


	@classmethod
	def from_yaml(cls, yaml_file: pathlib.Path | str):
		"""Creates MicroSimulator based on configuration in 'yaml_file' (examples are in the sims folder)."""
		config = yaml.safe_load(open(yaml_file))

		setup = config["setup"]
		try:
			# strictly required parameters
			size = QParse(setup.pop("size"))
			dimensions = setup.pop("dimensions")
		except KeyError as e:
			raise ValueError(f"Simulation configuration misses required parameter '{e.args[0]}'")

		obj = cls(size, dimensions, **setup)

		quantities = {}
		try:
			for s in config["species"]:
				obj.register_species(Species.from_dict(s))
				# the particles dict stores the initial quantity and bins per species
				quantities[s["name"]] = s["quantity"] if "quantity" in s else 0
		except KeyError:
			raise Exception("Each entry of configuration's 'species' must have a 'quantity'")

		if not "population" in config:
			obj.populate_urandom_speed(quantities, QParse(setup["max_speed"]))
		else:
			obj.populate_from_list(config["population"])


		return obj


	def register_species(self, species: Species) -> None:
		"""Register a species to the simulation"""
		if species.name not in self.species:
			self.species[species.name] = species
			# the smallest particle determines the smallest quadrant / octant -> splitting below that insensible
			self.tree.min_size = min(self.tree.min_size, 2*species.radius)
		else:
			warnings.warn(f"Ignoring registration of already registered species: {species.name}")


	def populate_from_list(self, population: list) -> None:
		"""Populate the simulation domain with particles from the given list of particle speciciations.
			 Each particle is specified by its 'species' (str), 'position' (list[float]) and 'velocity' (list[float])
		"""
		for particle in population:
			try:
				species = particle["species"]
				pos = np.array(particle["position"])
				vel = np.array(particle["velocity"])
				self.tree.add_particle(Particle(pos, vel, self.species[species]))
			except KeyError as e:
				raise Exception(f"Invalid particle in population, missing required element {e.args[0]}")
		self.tree.redistribute()


	@staticmethod
	def _urandom_speed_particle(species: Species, max_speed: QFloat, dimensions: QArray) -> Particle:
		"""Returns a randomly generated particle with random position within 'dimensions and a random velocity.
			 The velocity's magnitude is uniform random within [0,max_speed[
		"""
		mid_coords = []
		for d in range(len(dimensions)):
			mid_coords.append(np.random.rand() * (dimensions[d] - 2.2 * species.radius) + 1.1 * species.radius)

		# Scale max vel given for unit mass to species' particle mass
		v_mag = np.random.rand() * max_speed * math.sqrt(Q(1, unit(species.mass)) / species.mass)

		# Random initial velocity vector
		if len(mag(dimensions)) == 2:
			v_ang = np.random.rand() * 2 * np.pi
			vel_vec = [v_mag * np.cos(v_ang), v_mag * np.sin(v_ang)]
		# Can otherwise only be 3 -> ensured by __init__
		else:
			polar = np.random.rand() * 2 * np.pi
			azimuth = np.random.rand() * 2 * np.pi
			vel_vec = [v_mag * np.sin(polar) * np.cos(azimuth), v_mag * np.sin(polar) * np.sin(azimuth),
								 v_mag * np.cos(polar)]

		return Particle(list_to_qarr(mid_coords), list_to_qarr(vel_vec), species)


	def populate_urandom_speed(self, quantities : dict, max_speed: float) -> None:
		"""Init the simulation according to the quantities dictionary.
			 The keys of this dictionary are the names of previously registered species and its value
			 is the amount of such particles.
			 max_vel is the maximum speed particles of mass 1 can have. Ultimately, this determines the simulation's
			 temperature.
		"""
		# sort by species radius in order to start ran init with the biggest ones as they are hardest to fit
		info_sorted = dict(sorted(quantities.items(), key=lambda item: self.species[item[0]].radius, reverse=True))

		for _, key in enumerate(info_sorted):
			species = self.species.get(key)
			i = 0
			while i < quantities[key]:
				particle = self._urandom_speed_particle(species, max_speed, self.dims)
				if not self.tree.does_collide(particle):
					self.tree.add_particle(particle)
					i += 1
		self.tree.redistribute()

	def export_particles(self, file: str | PathLike) -> None:
		with open(file, "w") as f:
			f.write(f"{len(self.tree)}\n")
			sim_size = non_dim(self.tree.size)
			if len(self.dims) == 2:
				depth = non_dim(2*self.species['A'].radius)
#				f.write(f"{sim_size} {sim_size} {depth}\n")
				f.write(f"{sim_size} {sim_size}\n")
			else:
				f.write(f"{sim_size} {sim_size} {sim_size}\n")
			for particle in self.tree:
				species = particle.species.name
				pos = non_dim(particle.pos)
				vel = non_dim(particle.vel)
				radius = non_dim(particle.species.radius)
				if len(self.dims) == 2:
					# placed in middle of z axis and no velocity in that direction
#					f.write(f"{species} {' '.join(map(str, pos))} {depth / 2} {' '.join(map(str, vel))} {0.0} {radius}\n")
					f.write(f"{species} {' '.join(map(str, pos))} {' '.join(map(str, vel))} {radius}\n")
				else:
					f.write(f"{species} {' '.join(map(str, pos))} {' '.join(map(str, vel))} {radius}\n")