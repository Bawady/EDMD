import random

from dataclasses import dataclass, field

from UnitSystem import *


def _random_color() -> str:
	"""Returns random 24-bit color as hex-string."""
	return "#" + "".join([random.choice('0123456789ABCDEF') for _ in range(6)])


@dataclass
class Species:
	"""
	Encapsulates the properties of a chemical species.
	These include its name, the mass of a single particle of this species, its radius, and color
	(for graphical output).
	"""
	name: str
	mass: QFloat
	radius: QFloat
	color: str = field(default_factory=lambda: f"#{_random_color():06X}")

	@classmethod
	def from_dict(cls, config: dict):
		"""
		Create a Species instance from a dictionary.
		Required fields: 'name' (str), 'mass' (float), 'radius' (float)
		Optional fields: 'color' (24-bit hex string or integer)
		"""
		try:
				name = config["name"]
				mass = QParse(config["mass"])
				radius = QParse(config["radius"])
				if "color" in config:
					color = config["color"]
					# If color integer out of range
					if isinstance(color, int):
						color = f"#{color:06X}"
					# if hex-string invalid / too long
					elif isinstance(color, str) and (not color.startswith("#") or len(color) != 7):
						color = _random_color()
				else:
					color = _random_color()
				return cls(name=name, mass=mass, radius=radius, color=color)
		except KeyError as e:
				raise ValueError(f"Species configuration misses required field: {e.args[0]}")


	def __eq__(self, other):
		return self.name == other.name


	def __ne__(self, other):
		return not self.__eq__(other)