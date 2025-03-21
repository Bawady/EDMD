from UnitSystem import *

class Constants:
	constants_dict: dict[str, (float, str)] = {
		"KB": (1.380649E-23, "J/K")  # Boltzmann constant
	}

	@classmethod
	def prepare_constants(cls) -> None:
		for name, (mag, unit) in cls.constants_dict.items():
			setattr(cls, name, Q(mag, unit))

