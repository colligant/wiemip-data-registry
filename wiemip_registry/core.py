"""
Core of `wiemip_registry`: the `WIEAdapter` contract every per-model adapter
fills out, the `WIEFile` wrapper the dotted namespace resolves to, and the small
set of generic helpers the adapters share.

Per-model knowledge (paths, dims, time encoding, area recipe, fills) lives in
each `<MODEL>/convert.py` as a `WIEAdapter` subclass. This module only holds the
*generic* mechanics (spherical area, standardize, fill masking, weighted
aggregation), all seeded from the proven `extract.py`.
"""

from __future__ import annotations

import abc
import os
import functools
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import wiemip_registry.const as const


class MissingModelError(Exception):
    """Raised when a model is missing."""

    pass


class MissingForcingError(Exception):
    """Raised when a forcing is requested that doesn't exist."""

    pass


class MissingSimulationError(Exception):
    """Raised when a simulation is requested that doesn't exist."""

    pass


class MissingVariableError(Exception):
    """Raised when a variable is requested that doesn't exist."""

    pass


class MissingFactorialError(Exception):
    """Raised when a factorial is requested that doesn't exist."""

    pass


class InvalidSimulationError(Exception):
    """Raised when the combination of experiment, simulation, and factorial do not align with a known WIEMIP run."""

    pass


class InvalidExperimentError(Exception):
    """Raised when neither 1pctCO2 nor overshoot is requested."""

    pass


def ensure_valid(method):
    """
    Hand-deny any strange combination of arguments.
    Each adapter can have pathological combinations and decide whether to
    accept unconventional arguments (e.g., UKESM forcing a ctrl simulation),
    so we short-circuit that here.
    So:
     bgc/ctrl should only return True for stable climate
     cou should only return True for ukesm/gfdl/ipsl
     otherwise we should return false
    """

    @functools.wraps(method)
    def wrapper(self):

        if self.experiment == const.ONE_PERCENT_CO2_KEY:
            # cou and rad both include the radiative effect of CO2, so they are
            # driven with a transient GCM-pattern climate. bgc and ctrl hold the
            # climate constant, so they take the stable pattern.
            if self.simulation in (
                const.OnePctSimulation.cou.name,
                const.OnePctSimulation.cou_ndep.name,
                const.OnePctSimulation.rad.name,
                const.OnePctSimulation.rad_ndep.name,
            ):
                if self.forcing not in (
                    const.GCMPattern.ukesm.name,
                    const.GCMPattern.ipsl.name,
                    const.GCMPattern.gfdl.name,
                ):
                    self.invalid_combination = True
                    return False
            elif self.simulation in (
                const.OnePctSimulation.bgc.name,
                const.OnePctSimulation.bgc_ndep.name,
            ):
                if self.forcing != const.GCMPattern.stable.name:
                    self.invalid_combination = True
                    return False
            elif self.simulation == const.OnePctSimulation.ctrl.name:
                if self.forcing != const.GCMPattern.stable.name:
                    self.invalid_combination = True
                    return False

        return method(self)

    return wrapper


class Model(str):
    """A model name that also carries its adapter. Still a plain string
    (equality, hashing, joins, dict lookup all behave as the bare name), so it is
    a drop-in for the old string entries of `wr.models`; the extra attributes just
    save the `wr.adapters[name]` hop."""

    def __new__(cls, name: str, adapter: WIEAdapter) -> Model:
        obj = str.__new__(cls, name)
        obj._adapter = adapter
        return obj

    @property
    def adapter(self) -> WIEAdapter:
        """Return the `WIEAdapter` this model is part of. Implemented in wiemip_registry/MODEL_DIR."""
        return self._adapter

    @property
    def factorials(self) -> tuple[str, ...]:
        """Factorial names this model accepts (auto-ingested from its adapter)."""
        return self._adapter.factorials


class WIEAdapter(ABC):
    """
    Contract that each model must fill out. This converts whatever naming
    convention was uploaded to the WIE-MIP S3 bucket into something callable in
    Python.

    Subclasses implement the three abstract hooks (`path`, `read`,
    `_compute_weights`). There is no per-model "what exists" schema: the namespace
    lets a user select any (experiment, model, forcing, simulation, factorial,
    variable) by name, and a combination that wasn't uploaded simply fails when
    `read()` tries to open the file.
    """

    model: str

    _weights_cache: xr.DataArray | None = None

    # Per-model factorial vocabulary: canonical bucket -> however THIS model spells
    # the factorial
    # either overriden or set in the adapter subclass
    FACTORIALS: dict[str, str] = {"baseline": ""}

    # Same idea for the overshoot arm, empty unless the group ran more than one
    # overshoot configuration (only JULES did).
    OVERSHOOT_FACTORIALS: dict[str, str] = {}

    def land_carbon_variables(self) -> list[str]:
        """The variables that make up each model's land carbon stock. Usually some combination
        of cVeg, cSoil, cLitter, and cOther.
        """
        raise NotImplementedError()

    @property
    def factorials(self) -> tuple[str, ...]:
        """The factorial names this model accepts (drives namespace validation
        and tab-completion of the factorial axis)."""
        return tuple(self.FACTORIALS)

    @abc.abstractmethod
    def one_pct_path(
        self,
        simulation: str,
        forcing: str,
        factorial: str,
        variable: str,
    ) -> str:
        """Build the .nc path on the mounted S3 bucket by transforming the axis
        tokens into THIS model's upload naming convention."""

        raise NotImplementedError()

    def overshoot_path(
        self,
        simulation: str,
        forcing: str,
        variable: str,
        factorial: str | None = None,
    ) -> str:
        """Overshoot-experiment path. Overridden per model once that model's overshoot
        upload naming is known; until then asking for an overshoot path raises here
        rather than guessing a layout. Like `one_pct_path` it's a pure string transform
        — what exists is decided by `read()`.

        Most groups ran a single overshoot configuration, so `factorial` is None by
        default and those adapters ignore it. JULES is the exception: it repeated the
        whole scenario set under five fire configs, so the factorial picks the run."""
        raise NotImplementedError(f"overshoot paths not yet mapped for {self.model}")

    def paths(self, experiment, simulation, forcing, factorial, variable) -> list[str]:
        """Every file that makes up one variable's series, in time order. One file per
        variable for everyone except CLM's overshoot upload, which splits each run into
        time chunks, so the default just wraps `path`."""
        return [self.path(experiment, simulation, forcing, factorial, variable)]

    def path(self, experiment, simulation, forcing, factorial, variable):
        """Return the expected path of the file defined by the experiment, simulation, forcing, factorial, and variable.
        This is not guaranteed to exist - it just constructs the path.
        """

        if experiment == const.ONE_PERCENT_CO2_KEY:
            if factorial not in self.FACTORIALS:
                raise MissingFactorialError(
                    f"{self.model} has no '{factorial}' factorial (has: {sorted(self.FACTORIALS)})"
                )
            pth = self.one_pct_path(simulation, forcing, factorial, variable)
        elif experiment == "overshoot":
            pth = self.overshoot_path(simulation, forcing, variable, factorial)
        else:
            raise ValueError("Must specify either overshoot or one_percent_co2!")
        return pth

    @abc.abstractmethod
    def read(
        self,
        experiment: str,
        simulation: str,
        forcing: str,
        factorial: str,
        variable: str,
    ) -> xr.DataArray:
        """
        Open one variable and STANDARDIZE its layout: canonical dims
        ('time', 'lat', 'lon'[, level]), pd.DateTime `time` coord, sentinel fills
        masked to NaN. Units stay NATIVE — unit conversion happens in `WIEFile`,
        not here. Returns an unweighted DataArray.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _compute_weights(self) -> xr.DataArray:
        """Grid-cell weight [m2] for this model, per its README (provided raster
        OR computed spherical area), standardized to ('lat', 'lon')."""
        raise NotImplementedError()

    def weights(self) -> xr.DataArray:
        """Model weights [m2], materialized once and cached on the instance.
        Fills are zeroed: a land-only / ocean-masked area raster carries NaN over
        the cells it excludes, and `xarray.weighted()` rejects NaN weights — a
        zero weight drops the cell from the integral, which is what we want."""
        if self._weights_cache is None:
            self._weights_cache = self._compute_weights().fillna(0.0)
        return self._weights_cache

    def weight_dataarray(self, da: xr.DataArray) -> xr.core.weighted.DataArrayWeighted:
        """Wrap `da` in this model's documented weights via xarray `.weighted()`,
        so `.sum()`/`.mean()` over (lat, lon) are one call."""
        return da.weighted(self.weights())

    def to_pgc(self, total: xr.DataArray, variable: str) -> pd.Series:
        """Convert a per-timestep weighted (lat, lon) sum to a Pg series,
        PRESERVING the file's native cadence (monthly stays monthly — no annual
        collapse). MODEL-SPECIFIC: override when a model's upload units make the
        default wrong (e.g. its flux is not a per-second rate that SPY applies to).
        Default: stock `/PG`; flux rate `×SPY/PG` (a per-timestep annualized rate).

        Pg C for the carbon variables, Pg N for the nitrogen ones and Gt for the water
        pools — `PG` is a kg->Pg factor and carries no species, so the caller reads the
        species off the variable name."""
        s = total.to_series()
        s = s / const.PG if kind_of(variable) == "stock" else s * const.SPY / const.PG
        s.name = variable
        return s


def kind_of(variable: str) -> str:
    """'stock' or 'flux' — selects the global-integral unit conversion.

    'stock' means a per-m2 AMOUNT (no SPY multiply), 'flux' a per-second RATE. See
    `const.STOCKS`, whose membership comes from the uploads' `units` attributes: the
    carbon, nitrogen and water POOLS are stocks, everything else falls through to the
    rate branch — including the intensive variables, for which an area-weighted sum is
    the wrong reduction either way."""
    return "stock" if variable in const.STOCKS else "flux"


def is_annual(variable: str) -> bool:
    """Whether the variable is written at annual cadence (vs monthly) — selects
    the `yr`/`ann` vs `mon` filename token. Independent of stock/flux units (see
    const.ANNUAL): pools are annual, fluxes/states monthly."""
    return variable in const.ANNUAL


def spherical_area(
    obj: xr.Dataset | xr.DataArray, latn: str, lonn: str
) -> xr.DataArray:
    """
    Grid-cell area [m2] from lat/lon centres, assuming a regular spherical grid.

    Used by the models that do *not* ship an area raster (CLASSIC, JSBACH,
    JULES cell, VISIT-UT). Models with a provided raster (BiomeE `veg_area.nc`,
    LPX-Bern `gridcell_area.nc`, DLEM `LAND_AREA_DLEM.nc`) ignore this. Verbatim
    from extract.py.
    """
    lat, lon = obj[latn].values, obj[lonn].values
    R = 6.371e6
    dlat, dlon = np.abs(np.gradient(lat)), np.abs(np.gradient(lon))
    band = R**2 * (
        np.sin(np.deg2rad(lat + dlat / 2)) - np.sin(np.deg2rad(lat - dlat / 2))
    )
    return xr.DataArray(
        band[:, None] * np.deg2rad(dlon)[None, :],
        dims=(latn, lonn),
        coords={latn: obj[latn], lonn: obj[lonn]},
    ).astype("float32")


def mask_fill(da: xr.DataArray) -> xr.DataArray:
    """Mask sentinel fills not always declared as `_FillValue` (see FILL_FLOOR)."""
    return da.where(da > const.FILL_FLOOR)


def rename_latlon(da: xr.DataArray, latn: str, lonn: str) -> xr.DataArray:
    """Rename a model's native lat/lon dims to the canonical ('lat', 'lon')."""
    ren = {}
    if latn != "lat":
        ren[latn] = "lat"
    if lonn != "lon":
        ren[lonn] = "lon"
    return da.rename(ren) if ren else da


def standardize(
    da: xr.DataArray, latn: str, lonn: str, time: np.ndarray
) -> xr.DataArray:
    """
    Map a model's raw DataArray onto the canonical standardized form: lat/lon
    renamed to ('lat', 'lon'), a pandas-`datetime64` `time` coord attached
    (PRESERVING the file's native cadence — monthly stays monthly), and
    ('time', 'lat', 'lon') moved to the front while keeping any extra dims
    (e.g. PFT / soil levels). `da` is fill-masked already; `time` is the decoded
    datetime axis from the adapter's `_time(ds)` hook.
    """
    da = rename_latlon(da, latn, lonn)
    da = da.assign_coords(time=("time", to_datetime64(time)))
    front = [d for d in ("time", "lat", "lon") if d in da.dims]
    rest = [d for d in da.dims if d not in front]
    return da.transpose(*front, *rest)


def to_datetime64(values) -> np.ndarray:
    """A decoded time axis -> `datetime64[us]`.

    Microseconds, not nanoseconds: the overshoot runs end in 2300 and datetime64[ns]
    stops at 2262, so an ns cast silently wrapped every overshoot series past that year
    (2299 came back as 1714). Files that far out decode to cftime objects rather than
    datetime64, so pull the fields off them directly.
    """
    arr = np.asarray(values)
    if arr.dtype == object:  # cftime.datetime, one per timestep
        arr = np.array(
            [
                f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}"
                for t in arr
            ],
            dtype="datetime64[us]",
        )
    return arr.astype("datetime64[us]")


def cf_reference_month(units: str) -> np.datetime64:
    """The reference date of a CF `"<unit> since <date>"` string, as `datetime64[M]`.

    For the models read with `decode_times=False`, whose `_time` hook decodes the axis
    by hand. Those hooks used to hardcode a 1850 epoch, which is right for every
    1pctCO2 file and for the overshoot hist/ctrl runs but WRONG for the overshoot
    scenarios: DLEM and TEM both wrote those with a `since 2024-01-01` reference, so a
    hardcoded 1850 dated the 2024-2300 runs as 1850-2126 — a silent 174-year shift that
    also made a scenario un-joinable with its own historical run. Read the epoch off the
    file instead.
    """
    _, since, reference = units.partition("since")
    if not since:
        raise ValueError(f"not a CF 'unit since date' units string: {units!r}")
    return np.datetime64(reference.strip()[:7], "M")  # 'YYYY-MM'


def years_to_datetime(values) -> np.ndarray:
    """Numeric (possibly fractional) *calendar* years -> `datetime64[M]`, keeping
    sub-annual resolution: year = floor(v), month = round(frac * 12) clamped 0..11.
    Annual data (frac == 0) maps to January of each year. Used by the models whose
    time axis is a bare numeric year (LPX-Bern, VISIT-UT)."""
    v = np.asarray(values, dtype="float64")
    years = np.floor(v).astype("int64")
    months = np.clip(np.rint((v - years) * 12).astype("int64"), 0, 11)
    total_months = (years - 1970) * 12 + months  # months since the 1970 epoch
    return np.datetime64("1970-01", "M") + total_months.astype("timedelta64[M]")


def _csv_path(src: Path, start: float | None, end: float | None) -> Path:
    """Cache path for a source .nc + latitude band: the source path mirrored under
    `const.CSV_ROOT` (the `csv/` prefix), '.nc' -> '.csv', suffixed with the band
    ('global' for whole-globe, else '<start>_<end>')."""
    band = "global" if start is None or end is None else f"{start}_{end}"
    rel = src.relative_to(const.DATA_ROOT)
    return const.CSV_ROOT / rel.parent / f"{rel.stem}_{band}.csv"


def _pin_time_us(index: pd.Index) -> pd.Index:
    """Re-pin a CSV-read index's time level to `datetime64[us]`.

    read_csv picks a resolution per file, so a series that stops before 2262 comes
    back in ns and one running to 2300 in us; adding the two then overflows. Pin
    both to what standardize() hands out. Only the level actually named 'time' is
    a date — a `pft`/`pool`/`level` level is left alone.
    """
    if isinstance(index, pd.MultiIndex):
        if "time" not in index.names:
            return index
        i = index.names.index("time")
        return index.set_levels(
            pd.to_datetime(index.levels[i]).astype("datetime64[us]"), level=i
        )
    return pd.to_datetime(index).astype("datetime64[us]")


def _read_cached(out: Path) -> pd.Series:
    """Read back what `pd.Series.to_csv` wrote, for a flat OR MultiIndex series.

    `to_csv` writes one column per index level followed by the value column, so the
    values are the LAST column and everything before it is index. Reading a fixed
    `index_col=0` instead silently returned the second index level as the data for
    every variable that keeps a dim beyond (lat, lon) — the per-PFT pools on every
    model, CLM's `pool`-stacked `cSoilPools`/`rhPools`, the `*Layers` variables.
    That surfaced as strings (a PFT code) or, worse, as plausible floats (a year).
    """
    frame = pd.read_csv(out)
    *levels, value = frame.columns
    series = frame.set_index(levels)[value] if levels else frame[value]
    series.index = _pin_time_us(series.index)
    return series


def cache_csv(method):
    """Lazy CSV cache for a `WIEFile` (lat, lon)->time aggregation returning a
    `pd.Series`. Mirrors the result to a CSV under `const.CSV_ROOT` and recomputes
    only when that CSV is missing or older than the source variable file; on a hit
    it reads the CSV straight back.

    Use overwrite to recompute the sum. Useful when cache is invalid or methods change.
    """

    @functools.wraps(method)
    def wrapper(self, start=None, end=None, overwrite=False):
        srcs = [Path(p) for p in self.paths]  # pure transform == what read() opens
        out = _csv_path(srcs[0], start, end)
        if (
            not overwrite
            and out.exists()
            and all(src.exists() for src in srcs)
            and out.stat().st_mtime >= max(src.stat().st_mtime for src in srcs)
        ):
            return _read_cached(out)
        series = method(self, start, end)
        out.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(out)
        return series

    return wrapper


@dataclass
class WIEFile:
    """
    Thin, lazy wrapper over one variable's file(s) for one run. Holds identity +
    the model's adapter instance. No s3 access until a data method is called.
    """

    model: str  # canonical model name, e.g. "LPX-Bern"
    experiment: str  # on-disk experiment dir, "1pctCO2" | "overshoot"
    simulation: str  # per-experiment run name, e.g. "bgc", "hl"
    forcing: str  # GCM pattern name, e.g. "ukesm"
    variable: str  # CMIP name, e.g. "cVeg"
    _adapter: WIEAdapter
    factorial: str | None = None  # per-model factorial name, e.g. "baseline", "ndep"
    invalid_combination = False

    @property
    def kind(self) -> str:
        """Is the variable a stock or flux? Stocks don't carry a s-1 unit and as such are not multiplied by seconds."""
        return kind_of(self.variable)

    @property
    def units(self) -> str:
        """Native units string from the file header."""
        return self.read().attrs.get("units", "")

    @property
    def path(self) -> str:
        """Resolved bucket path, delegated to the model's adapter."""
        return str(
            self._adapter.path(
                self.experiment,
                self.simulation,
                self.forcing,
                self.factorial,
                self.variable,
            )
        )

    @property
    def paths(self) -> list[str]:
        """Every file behind this variable's series (one, unless the model chunked it)."""
        return [
            str(p)
            for p in self._adapter.paths(
                self.experiment,
                self.simulation,
                self.forcing,
                self.factorial,
                self.variable,
            )
        ]

    def read(self) -> xr.DataArray:
        """Standardized, *lazy* DataArray for this variable (canonical dims,
        pandas-datetime time coord at native cadence, NaN fills, native units).

        Raises whatever opening the file raises (FileNotFoundError for a combo
        that wasn't uploaded): read() is the single source of truth for what
        exists, so the caller can catch and report it. path() never pre-judges.
        """

        if not self.exists():
            if self.invalid_combination:
                raise InvalidSimulationError(
                    f"The combination of {self.experiment}, {self.simulation}, {self.forcing}, "
                    f"{self.factorial}, and {self.variable} does not exist in the WIEMIP protocol."
                )
            else:
                pth = self._adapter.path(
                    self.experiment,
                    self.simulation,
                    self.forcing,
                    self.factorial,
                    self.variable,
                )
                raise FileNotFoundError(
                    f"Missing file for {pth}. Requested: {self.experiment}, {self.simulation}, {self.forcing}, {self.factorial}, {self.variable}"
                )

        return self._adapter.read(
            self.experiment,
            self.simulation,
            self.forcing,
            self.factorial,
            self.variable,
        )

    def weighted_dataarray(
        self, da: xr.DataArray | None = None
    ) -> xr.core.weighted.DataArrayWeighted:
        """Wrap the data in this model's documented area weights, so a sum over
        (lat, lon) integrates the per-m2 quantity. Delegated to the adapter."""
        if da is None:
            da = self.read()
        return self._adapter.weight_dataarray(da)

    @ensure_valid
    def exists(self):
        """Does the file exist? Also returns false if a model hasn't been implemented yet in the registry."""
        try:
            pths = self.paths
        except (MissingFactorialError, NotImplementedError):
            # No such factorial for this model, or its overshoot naming isn't mapped
            # yet: either way there is nothing on disk to find. exists() is the
            # non-raising probe, so callers can sweep a request set without guarding
            # every combo; read() stays the gate that raises.
            return False
        return all(os.path.isfile(p) for p in pths)

    def zonal_mean(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> xr.DataArray:
        """Zonal mean of the variable at the file's native cadence and in native units. Land-only.
        The method returns a weighted mean across longitudes.
        Uses self.read() to open the netCDF then .mean("lon") to compute the mean over longitudes.
        NOT wrapped by `@cache_csv` since the result is 2-d.
        """
        da = self.read()

        if start is not None and end is not None:
            band = da.sel(lat=slice(start, end))
            if band.sizes.get("lat", 0) == 0:  # handle descending-lat grids
                band = da.sel(lat=slice(end, start))
            da = band

        total = self.weighted_dataarray(da).mean("lon")

        return total

    @cache_csv
    def latitudinal_sum(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> pd.Series:
        """Area-weighted total as a Pg C series at the file's native cadence
        (monthly stays monthly). With no band, sums the whole globe; pass
        (start, end) degrees to restrict to a latitude band. The unit conversion is
        delegated to the model's adapter (`to_pgc`).

        Wrapped by `@cache_csv`: the result is mirrored to a CSV under
        `const.CSV_ROOT` and reused until the source .nc is newer or overwrite is True.
        """
        da = self.read()
        if start is not None and end is not None:
            band = da.sel(lat=slice(start, end))
            if band.sizes.get("lat", 0) == 0:  # handle descending-lat grids
                band = da.sel(lat=slice(end, start))
            da = band
        total = self.weighted_dataarray(da).sum(("lat", "lon"))
        return self._adapter.to_pgc(total, self.variable)

    def __repr__(self) -> str:
        if self.factorial is None:
            return (
                f"WIEFile({self.experiment}.{self.simulation}.{self.model}."
                f"{self.forcing}.{self.variable})"
            )
        else:
            return (
                f"WIEFile({self.experiment}.{self.simulation}.{self.model}."
                f"{self.forcing}.{self.factorial}.{self.variable})"
            )
