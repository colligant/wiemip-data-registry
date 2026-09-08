"""LPJ-GUESS adapter.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from wiemip_registry import core
from wiemip_registry.const import DATA_ROOT

MODEL = "LPJ_GUESS"
_DIR = "LPJ-GUESS"
_OUTPUT = DATA_ROOT


def _sim_token(simulation: str) -> str:
    base, _, ndep = simulation.partition("_")
    return base.upper() + ("-Ndep" if ndep else "")


class LPJ_GUESS(core.WIEAdapter):
    model = MODEL
    LAT, LON = "lat", "lon"
    DECODE = False

    _DIR_STYLE_PREFIX = {"cLitterpft"}

    _PFT_STEMS = {
        "cVegpft": "cmass",
        "cLitterpft": "clitter",
        "nVegpft": "nmass",
        "nLitterpft": "nlitter",
    }

    def land_carbon_variables(self) -> list[str]:
        return ["cLitter", "cSoil", "cVeg"]

    def one_pct_path(self, simulation, forcing, factorial, variable) -> str:
        sim = _sim_token(simulation)
        run = f"{_DIR}_{forcing}_1pctCO2-{sim}"
        prefix = (
            run
            if variable in self._DIR_STYLE_PREFIX
            else f"{_DIR}_{forcing}_1pctco2_{sim}"
        )
        cad = "yr" if core.is_annual(variable) else "mon"
        return str(
            _OUTPUT
            / "1pctCO2"
            / "output"
            / _DIR
            / run
            / f"{prefix}_{variable}_{cad}_05deg.nc"
        )

    def _time(self, ds: xr.Dataset):
        tu = ds["time"].attrs.get("units", "")
        tv = np.asarray(ds["time"].values).astype("int64")
        base = core.cf_reference_month(tu)
        if "months since" in tu:
            return base + tv.astype("timedelta64[M]")
        return base + (tv * 12).astype("timedelta64[M]")

    def _stack_pfts(self, ds: xr.Dataset, variable: str) -> xr.DataArray:
        stem = self._PFT_STEMS[variable]
        names = [n for n in ds.data_vars if n.startswith(f"{stem}_")]
        if not names:
            raise core.MissingVariableError(
                f"{variable}: no '{stem}_*' fields in {ds.encoding.get('source', '?')}"
            )
        da = xr.concat([ds[n] for n in names], dim="pft")
        da = da.assign_coords(pft=[n[len(stem) + 1 :] for n in names])
        da.name = variable
        da.attrs = {"units": ds[names[0]].attrs.get("units", "")}
        return da

    def read(
        self, experiment, simulation, forcing, factorial, variable
    ) -> xr.DataArray:
        ds = xr.open_dataset(
            self.path(experiment, simulation, forcing, factorial, variable),
            decode_times=self.DECODE,
        )
        da = (
            self._stack_pfts(ds, variable)
            if variable in self._PFT_STEMS
            else ds[variable]
        )
        return core.standardize(core.mask_fill(da), self.LAT, self.LON, self._time(ds))

    def _compute_weights(self) -> xr.DataArray:
        ref = xr.open_dataset(
            self.path("1pctCO2", "bgc", "stable", "baseline", "cVeg"),
            decode_times=self.DECODE,
        )
        cell = core.spherical_area(ref, self.LAT, self.LON)
        ref.close()
        return cell
