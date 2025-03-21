import pint
import numpy as np
from enum import Enum
import typing
import warnings

from typing import TypeAlias, Union


class ConversionMode(Enum):
    DIM, NON_DIM, NO_PINT = 0, 1, 2


chara_qs: tuple["pint.Quantity", ...] = ()
dim_active: bool = True
ureg = pint.UnitRegistry()
ureg.setup_matplotlib()
ureg.define("cfu = [population]")
conv_mode = ConversionMode.DIM
mode_user_override = False

QFloat: TypeAlias = Union[float, pint.Quantity]
QArray: TypeAlias = Union[np.ndarray, pint.Quantity]


def Q(mag: float | np.ndarray, unit: str) -> pint.Quantity | float | np.ndarray:
    match conv_mode:
        case ConversionMode.DIM:
            return ureg.Quantity(mag, unit)
        case ConversionMode.NON_DIM:
            return non_dim(ureg.Quantity(mag, unit))
        case ConversionMode.NO_PINT:
            return mag


def QParse(q: str) -> pint.Quantity | float | np.ndarray:
    try:
        mag, unit = q.split(" ")
    except:
        unit = "1"
        try:
            mag = float(q)
        except:
            raise Exception(f"Could not parse string {q} as quantity")
    return Q(float(mag), unit)


def list_to_qarr(l: list[QFloat]) -> QArray:
    if not isinstance(l[0], pint.Quantity):
        return np.array(l)

    mags = []
    units = []
    dim = l[0].dimensionality
    for q in l:
        mags.append(q.magnitude)
        units.append(q.units)
        if q.dimensionality != dim:
            raise Exception(f"Cannot convert quantity list with heterogenous units {dim} {q.dimensionality} to QArray")
    return Q(np.array(mags), units[0])



def mag(q: pint.Quantity | float | np.ndarray) -> float | np.ndarray:
    if isinstance(q, pint.Quantity):
        return q.to_base_units().magnitude
    else:
        return q


def mag_list(qs: list[QFloat]) -> list[float]:
    if not isinstance(qs[0], pint.Quantity):
        return qs
    mags = [mag(q) for q in qs]
    return mags


def unit(q: pint.Quantity | float | np.ndarray) -> str:
    if isinstance(q, pint.Quantity):
        return q.units
    else:
        return ""


def q_zeros_like(q: np.ndarray | pint.Quantity) -> np.ndarray | pint.Quantity:
    if isinstance(q, pint.Quantity):
        return Q(np.zeros_like(q.magnitude), q.units)
    else:
        return np.zeros_like(q)


def q_ones_like(q: np.ndarray | pint.Quantity) -> np.ndarray | pint.Quantity:
    if isinstance(q, pint.Quantity):
        return Q(np.ones_like(q.magnitude), q.units)
    else:
        return np.ones_like(q)


def set_conversion_mode(mode: ConversionMode) -> None:
    global conv_mode, mode_user_override
    conv_mode = mode
    mode_user_override = True


def characteristics(*quantities: pint.Quantity) -> tuple[pint.Quantity | float | np.ndarray, ...]:
    global chara_qs, conv_mode
    if conv_mode != ConversionMode.DIM:
        warnings.warn("Setting characteristic quantities with a conversion mode != DIM has no effect")
        return tuple([q for q in quantities])
    qs = {}
    for i in range(len(quantities)):
        if not isinstance(quantities[i], pint.Quantity):
            raise Exception(f"Encountered non-dimensional characteristic quantity: {quantities[i]}")
        qs[chr(ord('a') + i)] = quantities[i]
    coeffs = ureg.pi_theorem(qs)

    if len(coeffs) > 0:
        raise Exception("Characteristic quantities given to unit system are not independent!")
    elif len(quantities) < 3:
        raise Exception("At least three independent characteristic quantities must be specified")
    else:
        chara_qs = quantities
        conv_mode = ConversionMode.NON_DIM if not mode_user_override else conv_mode
    return tuple([Q(q.magnitude, q.units) for q in quantities])


def convert(x: float | np.ndarray | pint.Quantity, q: pint.Quantity) -> pint.Quantity | float | np.ndarray:
    qs = {}
    for i in range(len(chara_qs)):
        qs[chr(ord('a') + i)] = chara_qs[i]
    qs['q'] = q
    coeffs = ureg.pi_theorem(qs)
    scale = 1
    for key in coeffs[0]:
        if key != "q":
            p = int(coeffs[0][key])
            scale *= qs[key]**p
    d = 1 / coeffs[0]['q']
    return (x * scale**d).to_base_units()


def non_dim(q: pint.Quantity | float | np.ndarray) -> float | np.ndarray:
    if not isinstance(q, pint.Quantity):
        return q
    q_dimless = convert(q, q)
    assert len(q_dimless.dimensionality) == 0
    return q_dimless.magnitude


def dim(q: float | np.ndarray | pint.Quantity, unit: str) -> pint.Quantity:
    if isinstance(q, pint.Quantity) or len(chara_qs) < 3:
        return q
    target = ureg.Quantity(1, unit)
    x = convert(q, 1 / target)
    assert x.dimensionality == target.dimensionality
    return x.to(unit)


if __name__ == '__main__':
    characteristics(Q(10, 'm'), Q(2, 's'), Q(2, 'N'), Q(3, 'cfu'))
    set_conversion_mode(ConversionMode.DIM)
    vel_nondim = non_dim(Q(2, 'm/s'))
    print(vel_nondim)
    print(dim(vel_nondim, "m/s"))
    print(non_dim(Q(12, 'kg')))
    print(dim(non_dim(Q(12, 'kg')), 'kg'))
    print(non_dim(Q(12, 'cfu')))
    print(dim(non_dim(Q(12, 'cfu')), 'cfu'))
    print(non_dim(Q(12, 'cfu/(L*s)')))
