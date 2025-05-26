import numpy as np
import math

from typing import Tuple

from Species import Species
from UnitSystem import *


def move(pos: np.ndarray, vel: np.ndarray, dt: float = 1) -> np.ndarray:
	return pos + vel * dt


def kinetic_energy(mass: float, vel: np.ndarray) -> float:
	"""Kinetic energy of this particle"""
	return mass * speed(vel) ** 2 / 2.0


def speed(vel: np.ndarray) -> float:
	"""Magnitude of the particle velocity"""
	return np.sqrt(vel.dot(vel))


def boundary_collision(pos : np.ndarray, vel : np.ndarray, radius: float, boundaries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Check and handle boundary collision."""
	for d in range(len(pos)):
		if pos[d] + radius >= boundaries[d]:
			vel[d] = -vel[d]
			pos[d] = boundaries[d] - radius
		elif pos[d] - radius <= 0:
			vel[d] = -vel[d]
			pos[d] = radius
	return pos, vel


def time_to_boundary(pos : np.ndarray, vel : np.ndarray, radius: float, boundaries: np.ndarray) -> float:
	"""Check and handle boundary collision."""
	vel_norm = vel / np.linalg.norm(vel)
	outer= pos + radius * vel_norm
	phi = math.atan2(-vel[1], vel[0])

	if 0 <= phi <= np.pi / 2:
		A = (500, 0)
		if phi <= np.pi / 2:
			B = (500, 500)
		else:
			B = (0,0)
	elif np.pi / 2 <= phi <= np.pi:
		A = (0, 0)
		if phi <= 3 * np.pi / 4:
			B = (500, 0)
		else:
			B = (0, 500)
	elif -np.pi <= phi <= -np.pi / 2:
		A = (0, 500)
		if phi <= -3 * np.pi / 4:
			B = (0, 0)
		else:
			B = (500, 500)
	else:
		A = (500, 500)
		if phi <= - np.pi / 4:
			B = (0, 500)
		else:
			B = (500, 0)

	A, B = np.array(A), np.array(B)
	AB = A-B
	det = np.linalg.det(np.column_stack((vel, -AB)))
	rhs = A - outer
	t = np.linalg.det(np.column_stack((rhs, -AB))) / det

	return t


def will_collide_in(pos1: np.ndarray, pos2: np.ndarray, radius1: float, radius2: float, vel1, vel2) -> float:
	dist = pos1 - pos2
	rel_vel = vel1 - vel2
	threshold = radius1 + radius2

	a = np.dot(rel_vel, rel_vel)
	b = 2 * np.dot(dist, rel_vel)
	c = np.dot(dist, rel_vel) - threshold**2

	discriminant = b ** 2 - 4 * a * c
	if discriminant < 0:
		return float('inf')
	sqrt_discriminant = math.sqrt(discriminant)

	t1 = (-b + sqrt_discriminant) / (2 * a)
	t2 = (-b - sqrt_discriminant) / (2 * a)

	if t1 < 0 and t2 < 0:
		return float('inf')
	elif t1 >= 0 and t2 >= 0:
		return min(t1, t2)
	elif t1 >= 0:
		return t1
	else:
		return t2


def particle_distance_sq(pos1 : np.ndarray, pos2: np.ndarray) -> float:
	"""Computes the squared distance between two points."""
	d = pos1 - pos2
	return np.dot(d, d)


def does_collide_with(pos1: np.ndarray, pos2: np.ndarray, radius1: float, radius2: float) -> bool:
	"""Returns True if two particles collide."""
	return particle_distance_sq(pos1, pos2) < (radius1 + radius2) ** 2


def particle_collision(pos1: np.ndarray, vel1: np.ndarray, mass1: float, radius1: float,
											 pos2: np.ndarray, vel2: np.ndarray, mass2: float, radius2: float,
											 dt: float) -> Tuple[bool, np.ndarray, np.ndarray]:
	"""Checks and performs elastic collision between two particles."""
	R_sq = (radius1 + radius2)**2
	d_pos = pos1 - pos2
	dist_pos_sq = particle_distance_sq(pos1, pos2)
	if dist_pos_sq < R_sq:
		return False, vel1, vel2

	tmp = 2. * np.dot(vel1 - vel2, d_pos) / dist_pos_sq * d_pos / (mass1 + mass2)
	vel1 -= mass2 * tmp
	vel2 += mass1 * tmp
	return True, vel1, vel2


class Particle:
	"""
	Encapsulates the behavior and properties of a single particle.
	A particle essentially consists of a position, velocity and chemical species.
	Includes collision (non-)reactive collision handling.
	"""

	_id_count = 0

	@classmethod
	def get_id(cls) -> int:
		cls._id_count += 1
		return cls._id_count - 1

	def __init__(self, pos: QArray, vel: QArray, species: "Species"):
		assert len(mag(pos)) == len(mag(vel)) and len(mag(pos)) in [2, 3], "Particle position and velocity must be of same dimensionality (2 or 3)"
		self.pos = pos
		self.vel = vel
		self.color = species.color
		self.species = species
		self.id = self.get_id()


	def set_color(self, color: int):
		self.color = color


	def move(self, dt: QFloat) -> None:
		"""Move this particle by the distance it travels in time dt"""
		self.pos = move(self.pos, self.vel, dt)


	def boundary_collision(self, boundaries: QArray) -> None:
		"""
				Checks if the particle would leave the simulation environment of given width and height
				within the next timestep of dt. If it does, it bounces of the wall.
		"""
		pos, vel = boundary_collision(self.pos, self.vel, self.species.radius, list_to_qarr(boundaries))
		self.pos = pos
		self.vel = vel


	def time_to_boundary(self, boundaries: QArray) -> float:
		"""
				Checks if the particle would leave the simulation environment of given width and height
				within the next timestep of dt. If it does, it bounces of the wall.
		"""
		return time_to_boundary(self.pos, self.vel, self.species.radius, list_to_qarr(boundaries))


	def kinetic_energy(self) -> QFloat:
		"""Kinetic energy of this particle"""
		return kinetic_energy(self.mass(), self.vel)


	def speed(self) -> QFloat:
		"""Magnitude of the particle velocity"""
		return speed(self.vel)


	def mass(self) -> QFloat:
		"""Particle mass"""
		return self.species.mass


	def particle_collision(self, p: "Particle", dt: QFloat) -> int:
		"""Checks if this particle and p collide and if they do perform the collision.
			 Returns None in case of a non-reactive collision or no collision and in case of
			 a reactive collision the new particle
		"""
		collision, v_self, v_p = particle_collision(self.pos, self.vel, self.species.mass, self.species.radius,
																		 p.pos, p.vel, p.species.mass, p.species.radius, dt=dt)
		if collision:
			self.vel = v_self
			p.vel = v_p
			return 1
		return 0


	def will_collide_in(self, p: "Particle") -> float:
		return will_collide_in(self.pos, p.pos, self.species.radius, p.species.radius, self.vel, p.vel)
