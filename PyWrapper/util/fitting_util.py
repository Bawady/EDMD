import numpy as np

from scipy.optimize import curve_fit


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
