"""Pint unit registry. Internal storage is millimetre / newton / N·mm."""

from __future__ import annotations

from functools import lru_cache

from pint import UnitRegistry


@lru_cache(maxsize=1)
def registry() -> UnitRegistry:
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    ureg.define("N_mm = newton * millimeter = Nmm")
    return ureg


def format_quantity(value: float, unit: str, digits: int = 4) -> str:
    qty = float(value) * registry()(unit)
    return f"{qty.magnitude:.{digits}g} {qty.units:~}"
