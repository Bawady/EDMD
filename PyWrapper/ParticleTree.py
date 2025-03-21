import numpy as np

from typing import List, Tuple, Iterator
from collections.abc import Sized, Iterable

from Particle import *
from Species import Species
from UnitSystem import *
from Constants import *


def _would_fit_leaf(particle_pos: np.ndarray, particle_radius: float, leaf_tl : np.ndarray, size: float, min_size: float) -> bool:
	"""Returns True if p would completely fit a quadrant / octant at pos with size. If the
		quadrant / octant is smaller than the minimum size returns False."""
	if size < min_size:
		return False

	# bounding box for p
	min_bound = particle_pos - particle_radius
	max_bound = particle_pos + particle_radius

	# Check if the particle's bounding box fits in all dimensions
	return not (np.any(leaf_tl > min_bound) or np.any(max_bound > leaf_tl + size))


def _is_within_bounds(particle_pos: np.ndarray, particle_radius: float, leaf_tl: np.ndarray, size: float) -> bool:
	"""Returns True if p overlaps with the quadrant / octant associated to this tree node.
		Note: overlap does not mean that p has to *completely* fit into this node.
	"""
	min_bound = particle_pos - particle_radius
	max_bound = particle_pos + particle_radius

	# Check if the particle's bounding box fits in all dimensions
	return not (np.any(max_bound < leaf_tl) or np.any(min_bound > (leaf_tl + size)))


class ParticleTree(Sized, Iterable):
	"""A quad- / oct-tree for storing particles of different chemical species.
		 Supports collisions (reactive and non-reactive) and the movement of particles
	"""
	def __init__(self, pos: np.ndarray, size : float, min_size: float, root: "ParticleTree" = None):
		"""pos are the coordinates of the top-left corner of the respective quadrant.
		   size is the quadrants /octants dimension. min_size is the minimum size a quadrant / octant can have.
		"""
		self.pos       : np.ndarray = pos
		self.size      : int = size
		self.root      : "ParticleTree" = root
		self.min_size  : int = min_size
		self.leaves    : List["ParticleTree"] = []
		self.particles : List[Particle] = []


	def particle_fits(self, p: Particle) -> bool:
		"""Checks if a particle completely fits into this tree. Returns true if it does, otherwise false"""
		return _would_fit_leaf(p.pos, p.species.radius, self.pos, self.size, self.min_size)


	def add_particle(self, p: Particle) -> None:
		"""Add the particle p to this tree"""
		# Try fitting p into a leaf node
		for l in self.leaves:
			if l.particle_fits(p):
				l.add_particle(p)
				return

		# Try creating a new suitable leaf node if none exists
		s2 = self.size / 2
		dim = len(self.pos)
		# iterate over all possible leaf node starting positions
		for i in range(2**dim):
			# add s2 in the respective dimension for this leaf or not
			offset = np.array([(i >> d) & 1 for d in range(dim)])
			leaf_pos = self.pos + offset * s2
			# If there exists a possible leaf node to which the particle would fit, create it and add it to leaves and return
			if _would_fit_leaf(p.pos, p.species.radius, leaf_pos, s2, self.min_size):
				leaf = ParticleTree(leaf_pos, s2, self.min_size, self)
				leaf.add_particle(p)
				self.leaves.append(leaf)
				return

		# No suitable leaf was found / created -> add to own quadrant / octant
		self.particles.append(p)


	def is_within_bounds(self, p: Particle) -> bool:
		"""Returns True if p overlaps with the quadrant / octant associated to this tree node.
			Note: overlap does not mean that p has to *completely* fit into this node.
		"""
		return _is_within_bounds(p.pos, p.species.radius, self.pos, self.size)


	def does_collide(self, p: Particle) -> bool:
		"""Returns True if particle p collides with any other particle contained in the quadrant / octant
			 associated to this tree node, otherwise False.
		"""
		for q in self.particles:
			if does_collide_with(p.pos, q.pos, p.species.radius, q.species.radius):
				return True
		for l in self.leaves:
			# Only check for collision with particles in leaf quadrants / octants where p is actually present
			if l.is_within_bounds(p) and l.does_collide(p):
				return True
		return False


	def is_empty(self) -> bool:
		"""Returns true if this tree (thus also its leaves) has no particles, otherwise false"""
		return len(self.particles) == 0 and len(self.leaves) == 0


	def redistribute(self) -> None:
		"""Redistribute particles contained in this tree such that they are assigned to the correct tree level
			 and quadrant. This can become necessary after particles moved.
		"""
		for l in self.leaves:
			l.redistribute()

		for p in self.particles:
			# the four is a heuristic but seems to work quite well
			if not self.particle_fits(p) or len(self.particles) > 1:
				self.particles.remove(p)
				if self.root is None:
					self.add_particle(p)
				else:
					self.root.add_particle(p)

		for l in self.leaves:
			if l.is_empty():
				self.leaves.remove(l)


	def particle_energy(self, species_name: str) -> Tuple[float, int]:
		"""Returns the total energy of particles of the given species contained in this tree as well as the
			 amount of such particles (allowing to compute the average energy as well)
		"""
		energy = Q(0, "J")
		cnt = 0
		for p in self.particles:
			if p.species.name == species_name:
				energy += p.mass() * p.speed() ** 2 / 2
				cnt += 1

		for l in self.leaves:
			e, c = l.particle_energy(species_name)
			energy += e
			cnt += c

		return energy, cnt
	

	def species_temperature(self, species_name: str) -> float:
		"""Returns the temperature for the particles of the given species"""
		e_kin, cnt = self.particle_energy(species_name)
		# use the equipartition theorem (alright for an ideal monoatomic gas)
		return 2 * e_kin / (3 * cnt * Constants.KB)



	def particle_speeds(self, species: Species) -> List[float]:
		"""Returns the speeds of all particles of the given species contained in this tree as list"""
		speeds = []
		for p in self.particles:
			if p.species== species:
				speeds.append(p.speed())
		for l in self.leaves:
			s = l.particle_speeds(species)
			speeds += s
		return speeds


	def particle_pos(self, species: Species) -> List:
		"""Returns the positions of all particles of the given species contained in this tree as list"""
		pos = []
		for p in self.particles:
			if p.species== species:
				pos.append((p.pos[0], p.pos[1], p.id))
		for l in self.leaves:
			s = l.particle_pos(species)
			pos += s
		return pos


	def __len__(self):
		return len(self.particles) + sum(len(l) for l in self.leaves)


	def __iter__(self) -> Iterator[Particle]:
		"""Allows to iterate over particles contained in tree"""
		yield from self.particles
		for l in self.leaves:
			yield from l