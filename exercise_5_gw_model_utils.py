from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
from affine import Affine
import rasterio
from rasterio.transform import rowcol
from scipy.interpolate import griddata

try:
    import flopy
    HAS_FLOPY = True
except Exception:
    HAS_FLOPY = False

_MF6_EXE_PATH = None
_MF6_UNAVAILABLE = False


def read_raster(path: Path):
    """Read a single-band raster and return (data_array, affine_transform, crs, nodata)."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs, src.nodata


def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a 2-D array by averaging non-overlapping square blocks of size *factor*."""
    nr = (arr.shape[0] // factor) * factor
    nc = (arr.shape[1] // factor) * factor
    a = arr[:nr, :nc]
    view = a.reshape(nr // factor, factor, nc // factor, factor)
    return np.nanmean(view, axis=(1, 3))


def block_reduce_mode(arr: np.ndarray, factor: int, nodata=None) -> np.ndarray:
    """Coarsen a 2-D integer array by taking the most frequent value in each block."""
    nr = (arr.shape[0] // factor) * factor
    nc = (arr.shape[1] // factor) * factor
    a = arr[:nr, :nc]
    out = np.full((nr // factor, nc // factor), nodata if nodata is not None else 0, dtype=np.int16)

    for i in range(out.shape[0]):
        r0 = i * factor
        r1 = r0 + factor
        for j in range(out.shape[1]):
            c0 = j * factor
            c1 = c0 + factor
            block = a[r0:r1, c0:c1].ravel()
            if nodata is not None:
                block = block[block != nodata]
            if block.size == 0:
                continue
            vals, cnts = np.unique(block, return_counts=True)
            out[i, j] = vals[np.argmax(cnts)]

    return out


def load_and_coarsen_inputs(data_dir: Path, target_cellsize_m: float):
    """
    Load all raster inputs and coarsen them to the requested model cell size.

    Parameters
    ----------
    data_dir : Path
        Folder containing dem.tif, surface_water.tif, sea.tif, geology.tif,
        and recharge_m_s.tif.
    target_cellsize_m : float
        Target model cell size in metres.  Actual size may differ slightly
        because the coarsening factor is rounded to the nearest integer.

    Returns
    -------
    dict with keys: 'dem', 'sw', 'sea', 'geo', 'rch', 'active', 'transform',
        'delr', 'delc', 'nrow', 'ncol', 'factor', 'native_dx', 'crs', 'dem_nodata'.
    """
    dem_raw, transform_raw, crs, dem_nodata = read_raster(data_dir / "dem.tif")
    sw_raw, _, _, sw_nodata = read_raster(data_dir / "surface_water.tif")
    sea_raw, _, _, sea_nodata = read_raster(data_dir / "sea.tif")
    geo_raw, _, _, geo_nodata = read_raster(data_dir / "geology.tif")
    rch_raw, _, _, _ = read_raster(data_dir / "recharge_m_s.tif")

    native_dx = abs(transform_raw.a)
    factor = max(1, int(round(target_cellsize_m / native_dx)))
    factor = min(factor, min(dem_raw.shape) // 20)

    dem = block_reduce_mean(dem_raw, factor)
    sw = block_reduce_mode(sw_raw, factor, nodata=int(sw_nodata) if sw_nodata is not None else None)
    sea = block_reduce_mode(sea_raw, factor, nodata=int(sea_nodata) if sea_nodata is not None else None)
    geo = block_reduce_mode(geo_raw, factor, nodata=int(geo_nodata) if geo_nodata is not None else None)
    rch = block_reduce_mean(rch_raw, factor)

    transform = transform_raw * Affine.scale(factor, factor)
    delr = abs(transform.a)
    delc = abs(transform.e)
    nrow, ncol = dem.shape

    is_sea = sea == 1
    active = np.isfinite(dem) & ((geo > 0) | (sw > 0) | is_sea)

    return {
        "dem": dem,
        "sw": sw,
        "sea": sea,
        "geo": geo,
        "rch": rch,
        "active": active,
        "transform": transform,
        "delr": delr,
        "delc": delc,
        "nrow": nrow,
        "ncol": ncol,
        "factor": factor,
        "native_dx": native_dx,
        "crs": crs,
        "dem_nodata": dem_nodata,
    }


def drainage_conductance(cell_dx, cell_dy, k_bed=2e-6, b_bed=1.0):
    """
    Compute the MODFLOW drain conductance for a streambed / lake-bed cell.

    Used for surface-water (SW) cells only, where a distinct low-permeability
    clogging layer separates the aquifer from the channel or lake.

    C = K_bed * (cell_dx * cell_dy) / b_bed  [m2/s]

    Parameters
    ----------
    cell_dx, cell_dy : float
        Cell dimensions in metres.
    k_bed : float
        Streambed / lake-bed hydraulic conductivity (m/s).
        Represents the clogging-layer material, not the aquifer K.
    b_bed : float
        Streambed / lake-bed thickness (m).  Resistance layer thickness.
    """
    area = cell_dx * cell_dy
    return k_bed * area / b_bed


def build_boundary_arrays(dem_arr, sw_arr, sea_arr, active_arr, delr, delc,
                           sea_level_m=0.0, k_bed=2e-6, b_bed=1.0,
                           hk_arr=None, b_eff=10.0):
    """
    Build drain (DRN) and constant-head (CHD) boundary arrays for the model domain.

    Sea cells are assigned a constant head equal to sea_level_m.  All other
    active cells receive a drain boundary that activates when head exceeds the
    local drain elevation (at the DEM surface for upland; 0.15 m below DEM for
    surface-water cells, which also receive a higher streambed conductance).

    Two physically distinct drain conductance formulations are used:

    * **Surface-water (SW) cells** (rivers / lakes):  A discrete low-permeability
      clogging layer is assumed.  Conductance = k_bed × dx × dy / b_bed × 5,
      following McDonald & Harbaugh (1988) and Fleckenstein et al. (2006).

    * **Upland seepage-face cells** (all other active non-sea cells):  The flow
      resistance comes from within the aquifer itself, not a separate clogging
      layer.  Conductance therefore scales with the local aquifer hydraulic
      conductivity:  C = hk[r,c] × dx × dy / b_eff, where b_eff is the
      effective coupling depth (default 10 m — much larger than 1 cell depth to
      avoid over-draining; increase further for more resistive seepage faces).
      This follows Rushton (2003) and Beven (1981) for diffuse seepage faces.
      If hk_arr is not provided, falls back to the scalar k_bed / b_bed formula
      for backward compatibility.

    .. note::
        Setting b_eff too small (e.g. 1 m with 100 m cells) produces drain
        conductances comparable to the aquifer lateral transmissivity, causing
        the upland drain to remove more water than recharge can supply and
        driving spurious inflow from the sea CHD boundary.

    Parameters
    ----------
    dem_arr, sw_arr, sea_arr, active_arr : ndarray
        Model grid arrays (same shape).
    delr, delc : float
        Cell size in row / column direction (m).
    sea_level_m : float
        Fixed head at sea cells (m a.s.l.).
    k_bed : float
        Streambed / lake-bed K for SW drain cells (m/s).
    b_bed : float
        Streambed / lake-bed thickness for SW drain cells (m).
    hk_arr : ndarray or None
        2-D aquifer hydraulic conductivity (m/s), same shape as dem_arr.
        When provided, upland seepage-face conductance scales as K × area / b_eff.
    b_eff : float
        Effective coupling depth for upland seepage-face cells (m).  Controls
        how tightly the drain conductance tracks the aquifer K.  Default 10.0 m.
        Smaller values produce higher conductances; values much smaller than
        the cell size relative to aquifer transmissivity can cause over-drainage
        and spurious sea inflow.  Increase to make the seepage face more passive.

    Returns
    -------
    drn_mask, drn_elev, drn_cond, chd_mask, chd_head : ndarray
    anchor_description : str
        Human-readable description of the fixed-head anchor used.
    """
    chd_mask = np.zeros_like(active_arr, dtype=bool)
    chd_head = np.full_like(dem_arr, np.nan, dtype=float)

    sea_cells = (sea_arr == 1) & active_arr
    if sea_cells.any():
        chd_mask[sea_cells] = True
        chd_head[sea_cells] = sea_level_m
        anchor_description = "sea cells as fixed head (0 m)"
    else:
        # When no sea, use the lowest surface-water cell as the hydraulic anchor.
        # All other active cells get a seepage/drain boundary.
        sw_land = (sw_arr > 0) & active_arr & (sea_arr != 1)
        if sw_land.any():
            idx = np.nanargmin(np.where(sw_land, dem_arr, np.nan))
            r0, c0 = np.unravel_index(idx, dem_arr.shape)
            chd_mask[r0, c0] = True
            chd_head[r0, c0] = dem_arr[r0, c0]
            anchor_description = f"lowest surface-water cell at row={r0}, col={c0}"
        else:
            edge_mask = np.zeros_like(active_arr, dtype=bool)
            edge_mask[0, :] = True
            edge_mask[-1, :] = True
            edge_mask[:, 0] = True
            edge_mask[:, -1] = True
            candidate = edge_mask & active_arr
            idx = np.nanargmin(np.where(candidate, dem_arr, np.nan))
            r0, c0 = np.unravel_index(idx, dem_arr.shape)
            chd_mask[r0, c0] = True
            chd_head[r0, c0] = dem_arr[r0, c0]
            anchor_description = f"lowest active edge cell at row={r0}, col={c0}"

    # Drain (seepage face) on all active, non-sea cells that are not already CHD.
    # Drain elevation: at land surface for upland (true seepage face — activates
    # only when WT reaches the surface); 0.15 m below DEM for SW cells.
    # SW cells with sw_arr==3 are sea-coded in the surface-water raster; exclude them.
    land_sw = (sw_arr > 0) & (sw_arr < 3) & (sea_arr != 1)
    drn_mask = active_arr & (sea_arr != 1) & ~chd_mask
    drn_elev = np.where(land_sw, dem_arr - 0.15, dem_arr)

    # ── Drain conductance ─────────────────────────────────────────────────────
    # SW cells: streambed / lake-bed clogging layer (fixed K_bed material).
    # Factor of 5 accounts for more efficient connectivity to open water.
    c_sw = drainage_conductance(delr, delc, k_bed=k_bed, b_bed=b_bed) * 5.0

    # Upland seepage-face cells: conductance proportional to local aquifer K so
    # that the seepage boundary is physically consistent with the calibrated K
    # field.  When hk_arr is None (backward-compat path), falls back to k_bed.
    area = delr * delc
    if hk_arr is not None:
        c_upland = hk_arr * area / b_eff
    else:
        c_upland = np.full_like(dem_arr,
                                drainage_conductance(delr, delc, k_bed=k_bed, b_bed=b_bed),
                                dtype=float)

    drn_cond = np.where(land_sw, c_sw, c_upland)

    return drn_mask, drn_elev, drn_cond, chd_mask, chd_head, anchor_description


def ensure_mf6_executable(model_dir: Path):
    """
    Locate or download the MODFLOW 6 executable, caching the path for reuse.

    Tries in order: (1) system PATH, (2) downloaded by flopy into model_dir/bin.
    Returns None if FLOPY is not installed or download fails.
    """
    global _MF6_EXE_PATH, _MF6_UNAVAILABLE

    if _MF6_EXE_PATH is not None:
        return _MF6_EXE_PATH
    if _MF6_UNAVAILABLE:
        return None

    mf6 = shutil.which("mf6")
    if mf6 is not None:
        _MF6_EXE_PATH = mf6
        return _MF6_EXE_PATH

    if not HAS_FLOPY:
        _MF6_UNAVAILABLE = True
        return None

    bindir = model_dir / "bin"
    bindir.mkdir(exist_ok=True)
    try:
        flopy.utils.get_modflow(bindir=str(bindir), subset=["mf6"], quiet=True)
    except Exception as err:
        warnings.warn(f"Could not download mf6 executable: {err}")
        _MF6_UNAVAILABLE = True
        return None

    candidates = [bindir / "mf6", bindir / "mf6.exe"]
    for c in candidates:
        if c.exists():
            _MF6_EXE_PATH = str(c)
            return _MF6_EXE_PATH
    _MF6_UNAVAILABLE = True
    return None


def run_flopy_steady(
    hk_arr,
    rch_arr,
    drn_mask,
    drn_elev,
    drn_cond,
    chd_mask,
    chd_head,
    active_arr,
    top_arr,
    nrow,
    ncol,
    delr,
    delc,
    aquifer_thickness_m,
    model_dir: Path,
    wells=None,
):
    """
    Run a steady-state groundwater model using FLOPY / MODFLOW 6.

    The aquifer is treated as **confined** (``icelltype=0``, constant
    transmissivity T = K × aquifer_thickness_m).  This is deliberate: it
    makes the MF6 solution physically equivalent to the iterative fallback
    solver, which also uses a fixed T = K × b.  The simplification is
    justified because aquifer thickness is poorly constrained for Norwegian
    catchments, and K spans several orders of magnitude — K is therefore the
    dominant uncertainty, not the saturated thickness.

    Parameters
    ----------
    hk_arr : ndarray
        2-D array of hydraulic conductivity (m/s).
    rch_arr : ndarray
        2-D array of recharge rates (m/s).
    drn_mask, drn_elev, drn_cond : ndarray
        Drain location mask, drain elevation (m), and conductance (m2/s).
    chd_mask, chd_head : ndarray
        Constant-head location mask and head values (m).
    active_arr, top_arr : ndarray
        Active-cell mask and land-surface elevation (m).
    nrow, ncol, delr, delc : int/float
        Grid dimensions and cell sizes (m).
    aquifer_thickness_m : float
        Assumed uniform aquifer thickness (m).  Sets T = K × b (confined).
    model_dir : Path
        Directory for MODFLOW workspace sub-folders.
    wells : list of dict, optional
        Each dict must contain 'row', 'col', and 'rate_m3_s'.
        Negative rate = extraction (pumping).  Ignored if None.

    Returns
    -------
    head : ndarray
        2-D head array (m above sea level).  NaN outside active cells.
    drn_flux : ndarray
        2-D drain outflow array (m3/s, positive = drainage to surface).
    engine : str
        Label 'flopy-mf6'.
    gwf : flopy.mf6.ModflowGwf
        The MODFLOW 6 groundwater-flow model object.  Its ``modelgrid``
        attribute is a fully populated ``StructuredGrid`` that can be passed
        directly to ``flopy.plot.PlotMapView`` or ``PlotCrossSection``.
        The simulation workspace (``sim_ws``) is kept on disk so that
        the modelgrid geometry remains accessible after the function returns.
    """
    if not HAS_FLOPY:
        raise RuntimeError("FLOPY not installed")

    mf6_exe = ensure_mf6_executable(model_dir)
    if mf6_exe is None:
        raise RuntimeError("mf6 executable not available")

    sim_ws = tempfile.mkdtemp(prefix="ex5_mf6_", dir=str(model_dir))
    sim = flopy.mf6.MFSimulation(sim_name="ex5", version="mf6", exe_name=mf6_exe, sim_ws=sim_ws)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple", outer_dvclose=1e-4, inner_dvclose=1e-4)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf_ex5", save_flows=True)

    idomain = np.where(active_arr, 1, 0).astype(int)
    botm = top_arr - aquifer_thickness_m

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top_arr,
        botm=botm,
        idomain=idomain,
    )

    flopy.mf6.ModflowGwfic(gwf, strt=np.where(active_arr, top_arr - 1.0, np.nan))
    # icelltype=0 → confined: T = K × aquifer_thickness_m (constant, head-independent).
    # This matches the iterative fallback, which also uses fixed T = K × b.
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=hk_arr, save_specific_discharge=True)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=np.where(active_arr, rch_arr, 0.0))

    drn_rc = np.argwhere(drn_mask)
    drn_spd = [[(0, int(r), int(c)), float(drn_elev[r, c]), float(drn_cond[r, c])] for r, c in drn_rc]
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drn_spd, maxbound=len(drn_spd))

    chd_rc = np.argwhere(chd_mask)
    chd_spd = [[(0, int(r), int(c)), float(chd_head[r, c])] for r, c in chd_rc]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd, maxbound=len(chd_spd))

    if wells:
        well_spd = [
            [(0, int(w['row']), int(w['col'])), float(w['rate_m3_s'])]
            for w in wells
            if 0 <= int(w['row']) < nrow
            and 0 <= int(w['col']) < ncol
            and active_arr[int(w['row']), int(w['col'])]
        ]
        if well_spd:
            flopy.mf6.ModflowGwfwel(gwf, stress_period_data=well_spd, maxbound=len(well_spd))

    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf_ex5.hds", saverecord=[("HEAD", "ALL")])

    sim.write_simulation()
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MODFLOW 6 run failed: " + "\n".join(buff))

    hds = flopy.utils.HeadFile(Path(sim_ws) / "gwf_ex5.hds")
    head = hds.get_data(kstpkper=(0, 0))[0]
    head = np.where(active_arr, head, np.nan)

    drn_flux = np.where(drn_mask & (head > drn_elev), drn_cond * (head - drn_elev), 0.0)
    # sim_ws is intentionally NOT cleaned up so that gwf.modelgrid remains
    # accessible for plotting via flopy.plot.PlotMapView / PlotCrossSection.
    return head, drn_flux, "flopy-mf6", gwf


def run_iterative_fallback(
    hk_arr,
    rch_arr,
    drn_mask,
    drn_elev,
    drn_cond,
    chd_mask,
    chd_head,
    active_arr,
    top_arr,
    delr,
    delc,
    aquifer_thickness_m,
    max_iter=2000,
    tol=1e-4,
    wells=None,
):
    """
    Solve the steady-state groundwater flow equation using a simple iterative
    (Gauss-Seidel-like) scheme.  Used as a fallback when FLOPY/MODFLOW 6 is
    unavailable.

    Transmissivity is held **constant** at T = K × aquifer_thickness_m
    (confined assumption), identical to the MF6 path which uses
    ``icelltype=0``.  This ensures calibrated K values are comparable
    regardless of which solver is used.  The simplification is justified
    because aquifer thickness is poorly constrained and K is the dominant
    uncertainty.

    Parameters follow the same convention as run_flopy_steady.  Wells are
    represented as point sinks added to the recharge field
    (rate_m3_s / cell_area, m/s) before the iteration starts.

    Returns the same triple (head, drn_flux, engine) as run_flopy_steady,
    with engine = 'iterative-fallback'.
    """
    # Apply well extraction as negative recharge before iterating.
    if wells:
        rch_arr = rch_arr.copy()
        cell_area = delr * delc
        for w in wells:
            r, c = int(w['row']), int(w['col'])
            if 0 <= r < active_arr.shape[0] and 0 <= c < active_arr.shape[1] and active_arr[r, c]:
                rch_arr[r, c] += float(w['rate_m3_s']) / cell_area

    transmissivity = np.maximum(hk_arr * aquifer_thickness_m, 1e-12)
    h = np.where(active_arr, top_arr - 1.0, np.nan)
    h = np.where(chd_mask, chd_head, h)

    cell_factor = (delr * delc) / np.maximum(transmissivity, 1e-12)

    # Pre-compute active-neighbour masks and counts (constant between iterations).
    # Using no-flow boundaries: inactive cells are excluded from the head average
    # so they do not create a spurious constant-head = 0 sink at catchment divides.
    a = active_arr.astype(float)
    north_a = np.roll(a,  1, axis=0); north_a[0,  :] = 0.0
    south_a = np.roll(a, -1, axis=0); south_a[-1, :] = 0.0
    west_a  = np.roll(a,  1, axis=1); west_a[:,  0]  = 0.0
    east_a  = np.roll(a, -1, axis=1); east_a[:, -1]  = 0.0
    # Number of active neighbours (1 at minimum to avoid 0-division for isolated cells).
    n_nb = np.maximum(north_a + south_a + west_a + east_a, 1.0)

    for _ in range(max_iter):
        h_old = h.copy()
        # Fill inactive cells with 0 so rolling produces numeric values;
        # the *_a masks then zero out any contribution from inactive neighbours.
        h_fill = np.where(active_arr, h, 0.0)

        north = np.roll(h_fill,  1, axis=0); north[0,  :] = 0.0
        south = np.roll(h_fill, -1, axis=0); south[-1, :] = 0.0
        west  = np.roll(h_fill,  1, axis=1); west[:,  0]  = 0.0
        east  = np.roll(h_fill, -1, axis=1); east[:, -1]  = 0.0

        # Sum only active neighbours (no-flow boundary: inactive sides add 0).
        sum_nb = north * north_a + south * south_a + west * west_a + east * east_a

        # Recharge source term (full, not divided by 4; divided by n_nb below).
        recharge_term = rch_arr * cell_factor

        # Drain removal (explicit: uses current h, stable for small conductances).
        drn_active = drn_mask & (h > drn_elev)
        drn_term = np.where(
            drn_active,
            (drn_cond / np.maximum(transmissivity, 1e-12)) * (h - drn_elev),
            0.0)

        # Gauss-Seidel-like update: solve for h given n_nb active neighbours.
        # For interior cells n_nb = 4; for no-flow boundaries n_nb < 4.
        h_new = (sum_nb + recharge_term - drn_term) / n_nb
        h = np.where(active_arr, 0.7 * h + 0.3 * h_new, np.nan)
        h = np.where(chd_mask, chd_head, h)

        maxdiff = np.nanmax(np.abs(h - h_old))
        if maxdiff < tol:
            break
    else:
        # Loop completed without meeting the tolerance — warn the user.
        warnings.warn(
            f"Iterative fallback solver did not converge after {max_iter} iterations "
            f"(max head change = {maxdiff:.4g} m, tolerance = {tol} m). "
            "Results may be inaccurate. Consider increasing max_iter or checking "
            "that drain conductances are not orders of magnitude larger than "
            "aquifer transmissivity.",
            UserWarning, stacklevel=2,
        )

    drn_flux = np.where(drn_mask & (h > drn_elev), drn_cond * (h - drn_elev), 0.0)
    return h, drn_flux, "iterative-fallback"


def simulate(
    hk_arr,
    dem,
    sw,
    sea,
    active,
    rch,
    nrow,
    ncol,
    delr,
    delc,
    model_dir,
    aquifer_thickness_m=60.0,
    recharge_multiplier=1.0,
    sea_level_m=0.0,
    k_bed=2e-6,
    b_bed=1.0,
    b_eff=10.0,
    wells=None,
):
    """
    Set up and run one steady-state groundwater model simulation.

    Tries FLOPY/MODFLOW 6 first; falls back to a simple iterative solver if
    FLOPY or the mf6 executable is unavailable.

    Parameters
    ----------
    hk_arr : ndarray
        2-D hydraulic conductivity field (m/s), same shape as dem.
    dem : ndarray
        Land-surface elevation (m above sea level).
    sw : ndarray
        Surface-water mask (0 = none, 1 = lake, 2 = river).
    sea : ndarray
        Sea mask (0 = land, 1 = sea, 2 = coastal).
    active : ndarray
        Boolean mask of active model cells.
    rch : ndarray
        Recharge rate (m/s) before multiplier is applied.
    nrow, ncol : int
        Grid dimensions.
    delr, delc : float
        Cell size in row and column directions (m).
    model_dir : Path
        Directory for MODFLOW workspace sub-folders.
    aquifer_thickness_m : float
        Uniform aquifer thickness used for T = K * b  (m). Default 60.0.
    recharge_multiplier : float
        Scale factor applied to rch before the run.  Use values > 1 for
        wet-event sensitivity tests.  Default 1.0.
    sea_level_m : float
        Fixed head assigned to sea cells (m).  Default 0.0.
    k_bed : float
        Streambed / lake-bed hydraulic conductivity for SW drain cells (m/s).
    b_bed : float
        Streambed / lake-bed thickness for SW drain cells (m).
    b_eff : float
        Effective coupling depth for upland seepage-face drain cells (m).
        Upland conductance = K[r,c] × dx × dy / b_eff.  Default 10.0.
    wells : list of dict, optional
        List of pumping/injection wells.  Each dict must have:
          'row'        : int  – row index (0-based)
          'col'        : int  – column index (0-based)
          'rate_m3_s'  : float – volumetric rate (m3/s).
                         Negative = extraction (pumping).
        Default None (no wells).

    Returns
    -------
    head : ndarray
        2-D hydraulic head field (m above sea level). NaN outside active cells.
    diagnostics : dict
        Contains 'engine', 'anchor', 'drn_mask', 'drn_elev', 'drn_cond',
        'chd_mask', 'chd_head', 'drn_flux', and 'modelgrid'.
        'modelgrid' is a ``flopy.discretization.StructuredGrid`` when the
        flopy-mf6 engine was used, or ``None`` when the iterative fallback
        was used (in which case the plotting utilities reconstruct an
        equivalent grid from the ``grid`` dict).
    """
    drn_mask, drn_elev, drn_cond, chd_mask, chd_head, anchor_desc = build_boundary_arrays(
        dem,
        sw,
        sea,
        active,
        delr,
        delc,
        sea_level_m=sea_level_m,
        k_bed=k_bed,
        b_bed=b_bed,
        hk_arr=hk_arr,
        b_eff=b_eff,
    )
    rch_use = rch * recharge_multiplier

    try:
        head, drn_flux, engine, gwf = run_flopy_steady(
            hk_arr,
            rch_use,
            drn_mask,
            drn_elev,
            drn_cond,
            chd_mask,
            chd_head,
            active,
            dem,
            nrow,
            ncol,
            delr,
            delc,
            aquifer_thickness_m,
            model_dir,
            wells=wells,
        )
        modelgrid = gwf.modelgrid
    except Exception as err:
        warnings.warn(f"Using fallback solver because FLOPY/MF6 was unavailable: {err}")
        head, drn_flux, engine = run_iterative_fallback(
            hk_arr,
            rch_use,
            drn_mask,
            drn_elev,
            drn_cond,
            chd_mask,
            chd_head,
            active,
            dem,
            delr,
            delc,
            aquifer_thickness_m,
            wells=wells,
        )
        modelgrid = None

    diagnostics = {
        "engine": engine,
        "anchor": anchor_desc,
        "drn_mask": drn_mask,
        "drn_elev": drn_elev,
        "drn_cond": drn_cond,
        "chd_mask": chd_mask,
        "chd_head": chd_head,
        "drn_flux": drn_flux,
        "modelgrid": modelgrid,
    }
    return head, diagnostics


def map_obs_to_grid(obs_df, transform_use, active_arr, dem_arr):
    """
    Map observation-point coordinates onto the model grid.

    Observation heads can be provided as 'water_level_masl' (directly in m
    above sea level) or as 'depth_to_water_m' (depth below DEM), or both.
    Points outside the active model domain are silently dropped.

    Returns
    -------
    pd.DataFrame with columns: station_id, r, c, obs_head_m, source.
    """
    rows = []
    for _, row in obs_df.iterrows():
        x = float(row["x"])
        y = float(row["y"])

        r_idx, c_idx = rowcol(transform_use, x, y)
        if r_idx < 0 or r_idx >= active_arr.shape[0] or c_idx < 0 or c_idx >= active_arr.shape[1]:
            continue
        if not active_arr[r_idx, c_idx]:
            continue

        obs_head = row.get("water_level_masl", np.nan)
        if pd.isna(obs_head):
            dtw = row.get("depth_to_water_m", np.nan)
            if pd.notna(dtw):
                obs_head = dem_arr[r_idx, c_idx] - float(dtw)

        if pd.isna(obs_head):
            continue

        rows.append(
            {
                "station_id": row["station_id"],
                "r": int(r_idx),
                "c": int(c_idx),
                "obs_head_m": float(obs_head),
                "source": row.get("source", ""),
            }
        )

    return pd.DataFrame(rows)


def evaluate_vs_obs(head_arr, obs_grid_df):
    """
    Compare modelled head values to observed water-table measurements.

    Returns
    -------
    out : pd.DataFrame
        obs_grid_df with extra columns 'model_head_m' and 'residual_m'
        (residual = modelled − observed).
    stats : dict
        Keys: 'n', 'rmse' (m), 'mae' (m), 'bias' (m), 'r2'.
    """
    if obs_grid_df.empty:
        return pd.DataFrame(), {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}

    out = obs_grid_df.copy()
    out["model_head_m"] = [head_arr[r, c] for r, c in zip(out["r"], out["c"])]
    out["residual_m"] = out["model_head_m"] - out["obs_head_m"]

    obs = out["obs_head_m"].to_numpy()
    mod = out["model_head_m"].to_numpy()
    ss_res = np.nansum((obs - mod) ** 2)
    ss_tot = np.nansum((obs - np.nanmean(obs)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    stats = {
        "n": int(len(out)),
        "rmse": float(np.sqrt(np.nanmean(out["residual_m"] ** 2))),
        "mae": float(np.nanmean(np.abs(out["residual_m"]))),
        "bias": float(np.nanmean(out["residual_m"])),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }
    return out, stats


def seepage_surfacewater_targets(head_arr, diagnostics, sw_arr, active_arr):
    """
    Evaluate how well the model reproduces surface-water / seepage locations.

    Three metrics are computed:
    - seepage_match_fraction  : fraction of mapped SW cells where the drain is
      active (head > drain elevation), i.e. groundwater discharges here.
    - surfacewater_stage_rmse_m : RMSE of (head − drain_elev) at SW cells.
    - seepage_surfacewater_jaccard  : Jaccard overlap of modelled seepage area
      and mapped SW area.
    - below_wt_fraction : fraction of mapped SW cells where head < drain
      elevation (water table is BELOW the river/lake bed – losing reaches).
      A high value indicates the model over-drains the stream network and
      should receive a calibration penalty.

    Returns
    -------
    dict with keys listed above plus 'n_surface_water_cells' and 'n_sw_below_wt'.
    """
    drn_mask = diagnostics["drn_mask"]
    drn_elev = diagnostics["drn_elev"]
    drn_flux = diagnostics["drn_flux"]

    drn_active = drn_mask & (head_arr > drn_elev)
    # Exclude sea-coded cells in the surface-water raster (sw_arr==3) from SW metrics.
    sw_cells = (sw_arr > 0) & (sw_arr < 3) & active_arr

    n_sw = int(sw_cells.sum())
    seepage_match = np.nan
    stage_rmse = np.nan
    below_wt_fraction = np.nan
    n_below_wt = 0
    if n_sw > 0:
        seepage_match = float((drn_active & sw_cells).sum() / n_sw)
        stage_rmse = float(np.sqrt(np.nanmean((head_arr[sw_cells] - drn_elev[sw_cells]) ** 2)))
        # Cells where the water table is below the river/lake bed (losing reach).
        below_wt_arr = sw_cells & np.isfinite(head_arr) & (head_arr < drn_elev)
        n_below_wt = int(below_wt_arr.sum())
        below_wt_fraction = n_below_wt / n_sw

    seepage_cells = (drn_flux > 0.0) & active_arr
    jaccard = np.nan
    if seepage_cells.any() or sw_cells.any():
        inter = float((seepage_cells & sw_cells).sum())
        union = float((seepage_cells | sw_cells).sum())
        jaccard = np.nan if union == 0 else inter / union

    return {
        "n_surface_water_cells": n_sw,
        "seepage_match_fraction": seepage_match,
        "surfacewater_stage_rmse_m": stage_rmse,
        "seepage_surfacewater_jaccard": jaccard,
        "n_sw_below_wt": n_below_wt,
        "below_wt_fraction": below_wt_fraction,
    }


def check_drain_activation(head_arr, diagnostics, sw, sea, active):
    """
    Summarise drain activation for rivers, lakes, and sea cells.

    Returns a DataFrame with columns: feature, n_cells, n_drn_active,
    active_fraction.  Useful for quickly checking whether surface-water
    bodies are receiving groundwater discharge in the model.
    """
    drn_mask = diagnostics["drn_mask"]
    drn_elev = diagnostics["drn_elev"]
    chd_mask = diagnostics["chd_mask"]

    drn_active = drn_mask & (head_arr > drn_elev)

    river = (sw == 2) & active & (sea != 1)
    lake = (sw == 1) & active & (sea != 1)
    sea_cells = (sea == 1) & active

    rows = []
    for name, mask in [("river", river), ("lake", lake)]:
        n_total = int(mask.sum())
        if n_total == 0:
            frac = np.nan
            n_act = 0
        else:
            n_act = int((drn_active & mask).sum())
            frac = n_act / n_total
        rows.append({"feature": name, "n_cells": n_total, "n_drn_active": n_act, "active_fraction": frac})

    n_sea = int(sea_cells.sum())
    n_sea_drn = int((sea_cells & drn_mask).sum())
    n_sea_chd = int((sea_cells & chd_mask).sum())
    rows.append(
        {
            "feature": "sea",
            "n_cells": n_sea,
            "n_drn_active": n_sea_drn,
            "active_fraction": (n_sea_chd / n_sea) if n_sea > 0 else np.nan,
        }
    )

    return pd.DataFrame(rows)


def infer_base_k(row):
    """
    Infer a base hydraulic conductivity (m/s) from a geology legend row.

    Two look-up paths are used:

    1. **Løsmasse (Quaternary) rows** — matched against Norwegian groundwater /
       infiltration potential keywords in the ``gw_potential`` and
       ``inf_potential`` columns (e.g. 'betydelig', 'mulig', 'lite', 'ikke').

    2. **Bedrock rows** (source = 'berggrunn_n50') — matched against common
       Norwegian lithology names in the ``deposit_type`` column.
       K values follow Gleeson et al. (2011), Table 1.

    Returns a default of 3e-6 m/s when no keyword matches.
    """
    # ── Path 1: løsmasse potential text ──────────────────────────────────────
    txt = f"{row.get('gw_potential', '')} {row.get('inf_potential', '')}".lower()
    if "betydelig" in txt or "godt egnet" in txt:
        return 8e-5   # high potential: gravel, coarse sand
    if "mulig" in txt or "egnet" in txt:
        return 2e-5   # moderate potential: fine sand, silty deposits
    if "lite" in txt:
        return 5e-6   # low potential: till, moraine
    if "ikke" in txt or "uegnet" in txt:
        return 8e-7   # very low potential: compact till, peat

    # ── Path 2: bedrock lithology name ───────────────────────────────────────
    # Reached when gw_potential / inf_potential are empty (e.g. Berggrunn N50).
    dep = str(row.get('deposit_type', '')).lower()

    # Carbonate rocks: slightly elevated K due to fracturing / dissolution
    if any(kw in dep for kw in ['kalkstein', 'dolomitt', 'marmor', 'kalk']):
        return 2e-7

    # Porous sedimentary rocks: sandstone, conglomerate
    if any(kw in dep for kw in ['sandstein', 'konglomerat', 'arkose', 'vake']):
        return 5e-7

    # Crystalline and metamorphic bedrock — typical Norwegian basement
    _bedrock_kws = [
        'gneis', 'granitt', 'granodior', 'monzon', 'syenitt', 'dioritt',
        'gabro', 'tonalit', 'larvikitt', 'anortositt',        # igneous
        'skifer', 'fylitt', 'kvartsitt', 'amfibo', 'glimmer', 'migmatitt',  # metamorphic
        'basalt', 'diabas', 'ryolitt', 'porfyr', 'grønnstein',              # volcanic
    ]
    if any(kw in dep for kw in _bedrock_kws):
        return 5e-8   # fractured crystalline rock (Gleeson 2011: 10^-8 to 10^-6)

    return 3e-6  # fallback for unrecognised unit types


def make_base_k_by_geology(geo, legend_df, active):
    """
    Build a 2-D hydraulic conductivity array from a geology raster and legend.

    Each geology unit ID in *geo* is looked up in *legend_df* (must have 'id'
    column) and assigned a base K via infer_base_k().  Active cells whose
    geology ID is missing from the legend receive the median K as a fallback.

    Returns
    -------
    hk_geo_base : ndarray  (m/s)
    legend : pd.DataFrame  (legend_df with added 'k_base_m_s' column)
    """
    legend = legend_df.copy()
    legend["k_base_m_s"] = legend.apply(infer_base_k, axis=1)
    id_to_kbase = dict(zip(legend["id"].astype(int), legend["k_base_m_s"]))

    hk_geo_base = np.zeros_like(geo, dtype=float)
    for gid, kval in id_to_kbase.items():
        hk_geo_base[geo == gid] = kval

    fallback = np.nanmedian(list(id_to_kbase.values()))
    hk_geo_base = np.where(active & (hk_geo_base <= 0), fallback, hk_geo_base)

    return hk_geo_base, legend


def build_k_zone_labels(hk_geo_base, active):
    """
    Assign each active cell to one of three K zones (0 = low, 1 = mid, 2 = high)
    based on tertile thresholds of the base K distribution.

    Returns a 2-D integer array (inactive cells = −1).
    """
    vals = hk_geo_base[active]
    q1 = np.nanquantile(vals, 1 / 3)
    q2 = np.nanquantile(vals, 2 / 3)
    zone = np.full_like(hk_geo_base, -1, dtype=int)
    zone[(hk_geo_base <= q1) & active] = 0
    zone[(hk_geo_base > q1) & (hk_geo_base <= q2) & active] = 1
    zone[(hk_geo_base > q2) & active] = 2
    return zone


def apply_spatial_k_control_points(base_hk, control_points, transform, active):
    """
    Blend user-defined K control-point values into a background K field.

    Each control point can be specified by (row, col) grid indices or by
    (x, y) projected coordinates.  Linear interpolation fills the domain
    between points; nearest-neighbour fills any remaining gaps.  The result
    is blended with *base_hk* as a geometric mean to smooth the transition
    and avoid abrupt artefacts.

    Returns
    -------
    blended : ndarray  (m/s)
    interp : ndarray or None
        The raw interpolated control-point field (None if < 2 points).
    """
    if not control_points:
        return base_hk.copy(), None

    pts_rc = []
    vals = []
    for cp in control_points:
        if ("row" in cp) and ("col" in cp):
            rr = int(cp["row"])
            cc = int(cp["col"])
        elif ("x" in cp) and ("y" in cp):
            rr, cc = rowcol(transform, float(cp["x"]), float(cp["y"]))
        else:
            continue

        if rr < 0 or rr >= base_hk.shape[0] or cc < 0 or cc >= base_hk.shape[1]:
            continue
        if not active[rr, cc]:
            continue

        k_val = cp.get("k_value_m_s", cp.get("k", None))
        if k_val is None:
            continue

        pts_rc.append([rr, cc])
        vals.append(float(k_val))

    if len(pts_rc) < 2:
        return base_hk.copy(), None

    pts_rc = np.asarray(pts_rc, dtype=float)
    vals = np.asarray(vals, dtype=float)

    rr, cc = np.indices(base_hk.shape)
    sample_points = np.column_stack([rr.ravel(), cc.ravel()])

    interp_lin = griddata(pts_rc, vals, sample_points, method="linear").reshape(base_hk.shape)
    interp_near = griddata(pts_rc, vals, sample_points, method="nearest").reshape(base_hk.shape)
    interp = np.where(np.isfinite(interp_lin), interp_lin, interp_near)

    # Blend control-point field with geology background to avoid abrupt artifacts.
    blended = np.where(active, np.sqrt(np.maximum(base_hk, 1e-12) * np.maximum(interp, 1e-12)), base_hk)
    return blended, interp


def combined_calibration_loss(
    obs_stats,
    target_stats,
    w_rmse=1.0,
    w_r2=0.75,
    w_seepage=0.5,
    w_sw_stage=0.3,
    w_below_wt=0.5,
):
    """
    Compute a scalar calibration loss that combines multiple goodness-of-fit
    criteria into a single value to minimise.

    Terms are scaled to comparable magnitudes (but not all constrained to
    [0, 1]) so that weights can be interpreted consistently. A higher loss
    means a worse calibration.

    Scaling used:
    - rmse_term     = rmse / 10
    - r2_term       = 1 - clamp(r2, -1, 1)
    - seep_term     = 1 - seepage_match_fraction
    - sw_stage_term = min(surfacewater_stage_rmse_m / 3, 5)
    - below_wt_term = below_wt_fraction

    Parameters
    ----------
    obs_stats : dict
        Output of evaluate_vs_obs (keys: 'rmse', 'r2').
    target_stats : dict
        Output of seepage_surfacewater_targets (keys: 'seepage_match_fraction',
        'surfacewater_stage_rmse_m', 'below_wt_fraction').
    w_rmse : float
        Weight for the RMSE term (head observations). Default 1.0.
    w_r2 : float
        Weight for the R² term. Default 0.75.
    w_seepage : float
        Weight for the seepage-match term. Default 0.5.
    w_sw_stage : float
        Weight for the surface-water stage RMSE term. Default 0.3.
    w_below_wt : float
        Weight for the fraction of SW cells where water table is below
        river/lake bed (losing-reach penalty). Default 0.5.

    Returns
    -------
    float  – combined loss value.
    """
    rmse     = obs_stats.get("rmse", np.nan)
    r2       = obs_stats.get("r2", np.nan)
    seep     = target_stats.get("seepage_match_fraction", np.nan)
    sw_stage = target_stats.get("surfacewater_stage_rmse_m", np.nan)
    below_wt = target_stats.get("below_wt_fraction", np.nan)

    rmse_term     = 1e6 if not np.isfinite(rmse)     else rmse / 10.0
    r2_term       = 2.0 if not np.isfinite(r2)       else (1.0 - max(min(r2, 1.0), -1.0))
    seep_term     = 1.0 if not np.isfinite(seep)     else (1.0 - seep)
    sw_stage_term = 1.0 if not np.isfinite(sw_stage) else min(sw_stage / 3.0, 5.0)
    below_wt_term = 1.0 if not np.isfinite(below_wt) else below_wt

    return (
        w_rmse     * rmse_term
        + w_r2     * r2_term
        + w_seepage * seep_term
        + w_sw_stage * sw_stage_term
        + w_below_wt * below_wt_term
    )


def compute_darcy_flux(head_arr, hk_arr, delr, delc, active_arr):
    """
    Compute cell-centred Darcy flux components from the modelled head field.

    Uses finite differences (np.gradient, central-difference scheme) on the
    head field, then multiplies by hydraulic conductivity::

        q = -K * ∇h

    NaN values in inactive cells are replaced with 0 before differencing so
    that gradients at the active-domain boundary are not contaminated, then
    masked back to NaN outside the active area.

    Parameters
    ----------
    head_arr : ndarray
        2-D hydraulic head (m above sea level). NaN outside active cells.
    hk_arr : ndarray
        2-D hydraulic conductivity (m/s).
    delr : float
        Cell size in the row direction, north–south (m).
    delc : float
        Cell size in the column direction, east–west (m).
    active_arr : ndarray
        Boolean mask of active model cells.

    Returns
    -------
    qx : ndarray
        Darcy flux in the east–west (column) direction (m/s).
        Positive = flow eastward.
    qy : ndarray
        Darcy flux in the north–south (row) direction (m/s).
        Positive = flow southward (increasing row index).
    q_mag : ndarray
        Magnitude of the Darcy flux vector, sqrt(qx² + qy²)  (m/s).
    """
    h = np.where(np.isfinite(head_arr), head_arr, 0.0)

    dh_dy = np.gradient(h, delr, axis=0)   # row direction (N–S)
    dh_dx = np.gradient(h, delc, axis=1)   # column direction (E–W)

    qx = np.where(active_arr, -hk_arr * dh_dx, np.nan)
    qy = np.where(active_arr, -hk_arr * dh_dy, np.nan)
    q_mag = np.sqrt(np.nan_to_num(qx, nan=0.0) ** 2 + np.nan_to_num(qy, nan=0.0) ** 2)
    q_mag = np.where(active_arr, q_mag, np.nan)

    return qx, qy, q_mag


def apply_slope_correction(hk_arr, dem_arr, delr, delc, active_arr, max_slope_m_m=5.0):
    """
    Scale hydraulic conductivity to account for longer flow paths in steep terrain.

    In a plan-view 2D groundwater model, the flow equation assumes a flat
    horizontal aquifer.  Where topography is steep the actual saturated flow
    path between two model cells is longer than the cell spacing, and the
    true hydraulic gradient along that path is smaller than the plan-view
    gradient.  The effective plan-view transmissivity is therefore reduced
    by the factor cos²(α), where α is the terrain slope angle::

        T_eff = T / (1 + s²)

    where ``s`` is the terrain slope in m/m (rise over run).

    Parameters
    ----------
    hk_arr : ndarray
        2-D hydraulic conductivity field (m/s).
    dem_arr : ndarray
        Land-surface elevation (m above sea level).
    delr, delc : float
        Cell size in row and column directions (m).
    active_arr : ndarray
        Boolean mask of active cells.
    max_slope_m_m : float
        Clip the computed slope to this value before applying the correction.
        Prevents extreme corrections in cliff-like cells where DEM artefacts
        may be present.  Default 5.0 m/m (about 79° slope).

    Returns
    -------
    hk_corrected : ndarray
        K field with slope correction applied (m/s).
    slope_factor : ndarray
        The multiplicative correction factor 1/(1+s²); values in (0, 1].
    """
    h = np.where(np.isfinite(dem_arr), dem_arr, 0.0)
    dz_dy = np.gradient(h, delr, axis=0)   # N-S gradient (row direction)
    dz_dx = np.gradient(h, delc, axis=1)   # E-W gradient (col direction)
    slope = np.sqrt(dz_dy**2 + dz_dx**2)  # slope magnitude (m/m)
    slope = np.clip(slope, 0.0, max_slope_m_m)
    slope_factor = 1.0 / (1.0 + slope**2)
    hk_corrected = np.where(active_arr, hk_arr * slope_factor, hk_arr)
    return hk_corrected, slope_factor


def list_geology_units_in_domain(geo_arr, legend_df, active_arr):
    """
    Return a DataFrame of geology units that are present in the active model domain.

    The table includes the number of active cells per unit, the unit legend
    attributes, and the default K value derived by ``infer_base_k``.

    Parameters
    ----------
    geo_arr : ndarray
        2-D geology ID raster (integer).
    legend_df : pd.DataFrame
        Legend table with at least an 'id' column.
    active_arr : ndarray
        Boolean mask of active cells.

    Returns
    -------
    pd.DataFrame
        Columns: id, n_cells, deposit_type, gw_potential, inf_potential,
        source, k_default_m_s.
        Sorted by n_cells descending.
    """
    ids_in_domain = np.unique(geo_arr[active_arr])
    legend = legend_df.copy()
    legend["id"] = legend["id"].astype(int)
    legend["k_default_m_s"] = legend.apply(infer_base_k, axis=1)

    rows = []
    for uid in ids_in_domain:
        n_cells = int((geo_arr == uid).sum())
        match = legend[legend["id"] == uid]
        if not match.empty:
            r = match.iloc[0]
            rows.append({
                "id":            int(uid),
                "n_cells":       n_cells,
                "deposit_type":  r.get("deposit_type", ""),
                "gw_potential":  r.get("gw_potential", ""),
                "inf_potential": r.get("inf_potential", ""),
                "source":        r.get("source", ""),
                "k_default_m_s": float(r["k_default_m_s"]),
            })
        else:
            rows.append({
                "id":            int(uid),
                "n_cells":       n_cells,
                "deposit_type":  "(not in legend)",
                "gw_potential":  "",
                "inf_potential": "",
                "source":        "",
                "k_default_m_s": 3e-6,
            })
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def make_base_k_by_geology_custom(geo_arr, legend_df, active_arr, custom_k_by_id=None):
    """
    Build a 2-D K array with optional per-unit custom K values.

    Identical to ``make_base_k_by_geology`` but accepts a dict
    ``custom_k_by_id`` that maps geology unit IDs to K values (m/s).
    Units not listed in the dict fall back to ``infer_base_k``.

    Parameters
    ----------
    geo_arr : ndarray
        2-D geology raster (integer IDs).
    legend_df : pd.DataFrame
        Geology legend (must have 'id' column).
    active_arr : ndarray
        Boolean mask of active cells.
    custom_k_by_id : dict or None
        Mapping {int unit_id: float K_m_s}.  None means use defaults only.

    Returns
    -------
    hk_geo_base : ndarray  (m/s)
    legend_out : pd.DataFrame  (legend with 'k_base_m_s' column)
    """
    legend = legend_df.copy()
    legend["id"] = legend["id"].astype(int)
    legend["k_base_m_s"] = legend.apply(infer_base_k, axis=1)

    if custom_k_by_id:
        for uid, kval in custom_k_by_id.items():
            mask = legend["id"] == int(uid)
            if mask.any():
                legend.loc[mask, "k_base_m_s"] = float(kval)

    id_to_k = dict(zip(legend["id"], legend["k_base_m_s"]))
    hk_geo_base = np.zeros_like(geo_arr, dtype=float)
    for gid, kval in id_to_k.items():
        hk_geo_base[geo_arr == gid] = kval

    fallback = np.nanmedian(list(id_to_k.values()))
    hk_geo_base = np.where(active_arr & (hk_geo_base <= 0), fallback, hk_geo_base)
    return hk_geo_base, legend


def seepage_flux_stats(drn_flux, active_arr, sw_arr, sea_arr, delr, delc):
    """
    Summarise seepage and drainage fluxes across the model domain.

    Parameters
    ----------
    drn_flux : ndarray
        2-D drain flux (m³/s per cell, positive = discharge to land surface).
    active_arr : ndarray
        Boolean mask of active cells.
    sw_arr : ndarray
        Surface-water mask (0 = none, 1 = lake, 2 = river, 3 = sea).
    sea_arr : ndarray
        Sea mask (0 = land, 1 = sea).
    delr, delc : float
        Cell dimensions (m).

    Returns
    -------
    dict with keys:
        'total_seepage_m3_s'      – total seepage discharge (m³/s).
        'total_seepage_mm_yr'     – domain-average seepage rate (mm/yr).
        'sw_seepage_m3_s'         – seepage at mapped SW cells (m³/s).
        'upland_seepage_m3_s'     – seepage at non-SW upland cells (m³/s).
        'seepage_map_mm_yr'       – 2-D array of per-cell seepage (mm/yr).
    """
    cell_area = delr * delc
    sec_per_yr = 365.25 * 86400.0

    seepage = np.where(active_arr & (sea_arr != 1), np.maximum(drn_flux, 0.0), 0.0)
    seepage_mm_yr = (seepage / cell_area) * 1000 * sec_per_yr

    land_sw = (sw_arr > 0) & (sw_arr < 3) & (sea_arr != 1) & active_arr
    upland = active_arr & (sea_arr != 1) & ~land_sw

    return {
        "total_seepage_m3_s":  float(seepage.sum()),
        "total_seepage_mm_yr": float(seepage_mm_yr[active_arr & (sea_arr != 1)].mean()),
        "sw_seepage_m3_s":     float(seepage[land_sw].sum()),
        "upland_seepage_m3_s": float(seepage[upland].sum()),
        "seepage_map_mm_yr":   seepage_mm_yr,
    }


def sea_flux_from_darcy(head_arr, hk_arr, sea_arr, active_arr, delr, delc,
                        aquifer_thickness_m):
    """
    Estimate the total groundwater flux into the sea using face-centred Darcy
    fluxes at the land/sea interface.

    For every active land cell that borders a sea cell (horizontally or
    vertically), the flux through the shared face is:

        Q_face = K_land * T_land * (h_land - h_sea) / (0.5 * delr_or_c)
                 * face_width * aquifer_thickness

    where ``T_land`` is the transmissivity of the land cell (K * b), the
    hydraulic gradient uses a half-cell distance (face is at the cell edge,
    shared boundary head is the sea-cell head), and ``face_width`` is the
    perpendicular cell dimension.

    Only net seaward flow (positive, i.e. land head > sea head) is summed;
    net landward flow contributes negative values (inflow from the sea).

    Parameters
    ----------
    head_arr : ndarray      – Modelled hydraulic head (m a.s.l.).  NaN outside active.
    hk_arr   : ndarray      – Hydraulic conductivity field (m/s).
    sea_arr  : ndarray      – Sea mask (1 = sea, 0 = land).
    active_arr : ndarray    – Boolean mask of active model cells.
    delr, delc : float      – Cell dimensions (m); delr = N-S, delc = E-W.
    aquifer_thickness_m : float – Uniform aquifer thickness (m).

    Returns
    -------
    sea_flux_m3s : float
        Net groundwater flux towards the sea (m³/s).
        Positive = water leaving the aquifer to the sea.
        Negative = water entering the aquifer from the sea (reversed gradient).
    sea_flux_map : ndarray
        Per-cell contribution to sea flux (m³/s), NaN outside interface cells.
        Positive = cell discharges to sea; negative = cell receives from sea.
    """
    nrow, ncol = head_arr.shape
    is_sea = (sea_arr == 1)

    # Fill NaN heads for sea cells with sea-level head (0 m, or as-is if present)
    h = np.where(np.isfinite(head_arr), head_arr, 0.0)
    # For sea cells the effective head is 0 m (sea level)
    h = np.where(is_sea, 0.0, h)

    sea_flux_map = np.full_like(head_arr, np.nan)
    total = 0.0

    # Iterate over the four cardinal neighbours of each active land cell.
    # Check right (east), left (west), down (south), up (north).
    land = active_arr & ~is_sea

    # Right face: land cell (r,c) — sea cell (r, c+1)
    mask_e = land[:, :-1] & is_sea[:, 1:]
    if mask_e.any():
        dh = h[:, :-1] - h[:, 1:]          # head difference (land minus sea=0)
        dist = 0.5 * delc                   # half-cell distance to shared face
        k_l  = hk_arr[:, :-1]
        q_face = k_l * aquifer_thickness_m * dh / dist * delr  # m³/s per cell
        contrib = np.where(mask_e, q_face, 0.0)
        sea_flux_map[:, :-1] = np.where(
            mask_e,
            np.nan_to_num(sea_flux_map[:, :-1], nan=0.0) + contrib,
            sea_flux_map[:, :-1],
        )
        total += float(contrib.sum())

    # Left face: land cell (r,c) — sea cell (r, c-1)
    mask_w = land[:, 1:] & is_sea[:, :-1]
    if mask_w.any():
        dh = h[:, 1:] - h[:, :-1]
        dist = 0.5 * delc
        k_l  = hk_arr[:, 1:]
        q_face = k_l * aquifer_thickness_m * dh / dist * delr
        contrib = np.where(mask_w, q_face, 0.0)
        sea_flux_map[:, 1:] = np.where(
            mask_w,
            np.nan_to_num(sea_flux_map[:, 1:], nan=0.0) + contrib,
            sea_flux_map[:, 1:],
        )
        total += float(contrib.sum())

    # Down face: land cell (r,c) — sea cell (r+1, c)
    mask_s = land[:-1, :] & is_sea[1:, :]
    if mask_s.any():
        dh = h[:-1, :] - h[1:, :]
        dist = 0.5 * delr
        k_l  = hk_arr[:-1, :]
        q_face = k_l * aquifer_thickness_m * dh / dist * delc
        contrib = np.where(mask_s, q_face, 0.0)
        sea_flux_map[:-1, :] = np.where(
            mask_s,
            np.nan_to_num(sea_flux_map[:-1, :], nan=0.0) + contrib,
            sea_flux_map[:-1, :],
        )
        total += float(contrib.sum())

    # Up face: land cell (r,c) — sea cell (r-1, c)
    mask_n = land[1:, :] & is_sea[:-1, :]
    if mask_n.any():
        dh = h[1:, :] - h[:-1, :]
        dist = 0.5 * delr
        k_l  = hk_arr[1:, :]
        q_face = k_l * aquifer_thickness_m * dh / dist * delc
        contrib = np.where(mask_n, q_face, 0.0)
        sea_flux_map[1:, :] = np.where(
            mask_n,
            np.nan_to_num(sea_flux_map[1:, :], nan=0.0) + contrib,
            sea_flux_map[1:, :],
        )
        total += float(contrib.sum())

    return total, sea_flux_map


def water_budget(drn_flux, rch, active_arr, sw_arr, sea_arr, delr, delc,
                 rch_multiplier=1.0,
                 head_arr=None, hk_arr=None, aquifer_thickness_m=None):
    """
    Compute a catchment-wide water budget with components in m³/s and mm/yr.

    At steady state the budget must close:
        Recharge = Lake discharge + River discharge + Upland seepage + Sea discharge

    Surface-water discharge is split into **lake** (sw==1) and **river** (sw==2)
    components.

    Sea discharge is estimated in two ways:

    1. **Darcy face-flux** (preferred): when ``head_arr``, ``hk_arr``, and
       ``aquifer_thickness_m`` are provided, the flux is computed explicitly at
       the land/sea interface using face-centred finite differences (see
       ``sea_flux_from_darcy``).  This is the physically correct estimate.

    2. **Mass-balance residual** (fallback / cross-check): computed as
       Recharge − SW discharge − Upland seepage.  Returned as
       ``sea_discharge_residual_mm_yr`` for comparison.

    All mm/yr values are normalised by the **active land area** (active cells
    excluding sea cells).

    Parameters
    ----------
    drn_flux : ndarray
        2-D drain flux (m³/s per cell, positive = discharge to land surface).
    rch : ndarray
        2-D recharge rate (m/s), before applying the multiplier.
    active_arr : ndarray
        Boolean mask of active model cells.
    sw_arr : ndarray
        Surface-water mask (0 = none, 1 = lake, 2 = river, 3 = sea).
    sea_arr : ndarray
        Sea mask (0 = land, 1 = sea).
    delr, delc : float
        Cell dimensions (m).
    rch_multiplier : float
        Recharge multiplier applied to *rch* (default 1.0 = present-day).
    head_arr : ndarray or None
        Modelled hydraulic head (m a.s.l.).  Required for Darcy sea flux.
    hk_arr : ndarray or None
        Hydraulic conductivity field (m/s).  Required for Darcy sea flux.
    aquifer_thickness_m : float or None
        Aquifer thickness (m).  Required for Darcy sea flux.

    Returns
    -------
    dict with keys:
        'land_area_m2'                – active land area (m²).
        'n_land_cells'                – number of active non-sea cells.
        'rch_multiplier'              – the multiplier used.
        'recharge_m3_s'               – total recharge (m³/s).
        'lake_discharge_m3_s'         – discharge to lakes (m³/s).
        'river_discharge_m3_s'        – discharge to rivers (m³/s).
        'sw_discharge_m3_s'           – total SW discharge (lake + river, m³/s).
        'upland_seepage_m3_s'         – seepage outside SW cells (m³/s).
        'sea_discharge_darcy_m3_s'    – sea flux via Darcy face method (m³/s);
                                        NaN if head_arr / hk_arr not provided.
        'sea_discharge_residual_m3_s' – sea flux as mass-balance residual (m³/s).
        'sea_flux_map'                – per-cell sea-interface flux (m³/s) or None.
        'recharge_mm_yr'
        'lake_discharge_mm_yr'
        'river_discharge_mm_yr'
        'sw_discharge_mm_yr'
        'upland_seepage_mm_yr'
        'sea_discharge_darcy_mm_yr'
        'sea_discharge_residual_mm_yr'
        'budget_imbalance_darcy_mm_yr'   – Recharge − all outputs (Darcy sea).
        'budget_imbalance_residual_mm_yr'– always ≈ 0 by construction.
    """
    cell_area = delr * delc
    sec_per_yr = 365.25 * 86400.0

    # Boolean helpers
    is_sea = (sea_arr == 1) & active_arr
    land_active = active_arr & ~is_sea
    lake_sw  = (sw_arr == 1) & land_active
    river_sw = (sw_arr == 2) & land_active
    land_sw  = lake_sw | river_sw
    upland   = land_active & ~land_sw

    # Active land area used as denominator for mm/yr conversion
    n_land = int(land_active.sum())
    land_area_m2 = n_land * cell_area

    def _to_mm_yr(q_m3s):
        if q_m3s is None or (isinstance(q_m3s, float) and np.isnan(q_m3s)):
            return np.nan
        return q_m3s * sec_per_yr * 1000.0 / land_area_m2

    # IN: total recharge over all active land cells
    rch_eff = rch * rch_multiplier
    recharge_m3s = float(np.where(land_active, rch_eff * cell_area, 0.0).sum())

    # OUT: discharge split by SW type
    lake_discharge_m3s  = float(np.where(lake_sw,  np.maximum(drn_flux, 0.0), 0.0).sum())
    river_discharge_m3s = float(np.where(river_sw, np.maximum(drn_flux, 0.0), 0.0).sum())
    sw_discharge_m3s    = lake_discharge_m3s + river_discharge_m3s

    # OUT: upland seepage
    upland_seepage_m3s = float(np.where(upland, np.maximum(drn_flux, 0.0), 0.0).sum())

    # OUT: sea discharge — direct Darcy face-flux estimate
    sea_flux_map = None
    if head_arr is not None and hk_arr is not None and aquifer_thickness_m is not None:
        sea_darcy_m3s, sea_flux_map = sea_flux_from_darcy(
            head_arr, hk_arr, sea_arr, active_arr, delr, delc, aquifer_thickness_m)
    else:
        sea_darcy_m3s = np.nan

    # OUT (residual): sea discharge as steady-state mass balance
    sea_residual_m3s = recharge_m3s - sw_discharge_m3s - upland_seepage_m3s

    # Warn if residual is negative (overactive drains)
    if sea_residual_m3s < 0:
        import warnings as _w
        _w.warn(
            f"Water budget: sea_discharge residual is NEGATIVE "
            f"({_to_mm_yr(sea_residual_m3s):.1f} mm/yr). "
            "SW/upland drains are removing more water than recharge supplies — "
            "the CHD sea boundary is acting as a net source. "
            "Check drain_coupling_depth_m (b_eff) and drain elevation settings.",
            UserWarning, stacklevel=2,
        )

    # Imbalance when using Darcy sea estimate
    if not np.isnan(sea_darcy_m3s):
        imbalance_darcy_m3s = (recharge_m3s
                               - lake_discharge_m3s - river_discharge_m3s
                               - upland_seepage_m3s - sea_darcy_m3s)
    else:
        imbalance_darcy_m3s = np.nan

    # Residual imbalance is 0 by construction
    imbalance_residual_m3s = 0.0

    return {
        'land_area_m2':                    land_area_m2,
        'n_land_cells':                    n_land,
        'rch_multiplier':                  rch_multiplier,
        'recharge_m3_s':                   recharge_m3s,
        'lake_discharge_m3_s':             lake_discharge_m3s,
        'river_discharge_m3_s':            river_discharge_m3s,
        'sw_discharge_m3_s':               sw_discharge_m3s,
        'upland_seepage_m3_s':             upland_seepage_m3s,
        'sea_discharge_darcy_m3_s':        sea_darcy_m3s,
        'sea_discharge_residual_m3_s':     sea_residual_m3s,
        'sea_flux_map':                    sea_flux_map,
        'recharge_mm_yr':                  _to_mm_yr(recharge_m3s),
        'lake_discharge_mm_yr':            _to_mm_yr(lake_discharge_m3s),
        'river_discharge_mm_yr':           _to_mm_yr(river_discharge_m3s),
        'sw_discharge_mm_yr':              _to_mm_yr(sw_discharge_m3s),
        'upland_seepage_mm_yr':            _to_mm_yr(upland_seepage_m3s),
        'sea_discharge_darcy_mm_yr':       _to_mm_yr(sea_darcy_m3s),
        'sea_discharge_residual_mm_yr':    _to_mm_yr(sea_residual_m3s),
        'budget_imbalance_darcy_mm_yr':    _to_mm_yr(imbalance_darcy_m3s),
        'budget_imbalance_residual_mm_yr': _to_mm_yr(imbalance_residual_m3s),
    }


def make_k_by_geo_groups(geo_arr, active_arr, groups, allow_fallback=True):
    """
    Assign each active cell to a user-defined geology calibration group.

    Students define groups by listing the geology unit IDs that belong to each
    group.  scipy.optimize then calibrates one K multiplier per group.

    Parameters
    ----------
    geo_arr : ndarray
        2-D geology unit ID raster (integer values).
    active_arr : ndarray
        Boolean mask of active model cells.
    groups : dict
        Ordered mapping  ``{group_name: [list of geology IDs]}``.
        Example::

            groups = {
                'quaternary': [3, 7, 12],     # Quaternary deposits
                'bedrock':    [1, 5, 9, 14],  # Older bedrock
            }

        Active cells whose geology ID does not appear in any group are, by
        default, assigned to the **last** group as a fallback.
    allow_fallback : bool, optional
        If True (default), active cells not covered by any provided unit list
        are assigned to the last group. If False, a ValueError is raised when
        unassigned active cells are detected.

    Returns
    -------
    labels : ndarray of int
        Same shape as geo_arr.  -1 = inactive cell;
        0, 1, … = group index (matching order of *groups* dict).
    group_names : list of str
        Group names in the same order as the label indices.
    """
    group_names = list(groups.keys())
    labels = np.full(geo_arr.shape, -1, dtype=int)

    for i, (name, ids) in enumerate(groups.items()):
        for uid in ids:
            labels[(geo_arr == uid) & active_arr] = i

    unassigned = active_arr & (labels == -1)
    if unassigned.any():
        if allow_fallback:
            last = len(group_names) - 1
            labels[unassigned] = last
        else:
            missing_ids = np.unique(geo_arr[unassigned]).astype(int).tolist()
            raise ValueError(
                "Unassigned active geology unit IDs: "
                f"{missing_ids}. Add them to groups or enable fallback."
            )

    return labels, group_names


def find_cross_section_transects(head, active, nrow, ncol, delr, delc,
                                  n_sections=3):
    """
    Automatically place cross-section lines perpendicular to water-table contours.

    The dominant groundwater flow direction is estimated from the mean head
    gradient.  Cross-sections are placed **perpendicular** to that direction
    (i.e., they run *along* the flow path, crossing from high to low head).
    Three sections are evenly spaced across the active domain.

    Parameters
    ----------
    head : ndarray
        Modelled hydraulic head (m a.s.l.).
    active : ndarray
        Boolean active-cell mask.
    nrow, ncol : int
        Grid dimensions.
    delr, delc : float
        Cell size in row / column direction (m).
    n_sections : int
        Number of transects (default 3).

    Returns
    -------
    list of dict, each containing:
      ``'label'``  : str – e.g. ``'Section 1'``.
      ``'rows'``   : ndarray (int) – row indices along the transect.
      ``'cols'``   : ndarray (int) – column indices along the transect.
      ``'dist_m'`` : ndarray (float) – distance along transect in metres,
                     starting at 0.
    """
    head_nan = np.where(active, head, np.nan)
    h_filled = np.nan_to_num(head_nan, nan=0.0)

    # Mean gradient of the head field over active cells.
    # dh_dc : change in head per column step (E-W gradient).
    # dh_dr : change in head per row step (N-S gradient; row 0 = top of domain).
    dh_dc = float(np.nanmean(np.gradient(h_filled, axis=1)[active]))
    dh_dr = float(np.nanmean(np.gradient(h_filled, axis=0)[active]))

    # Dominant gradient axis:
    # If |dh_dc| ≥ |dh_dr|  → flow is mostly E-W → transects run N-S (fix a column).
    # If |dh_dr| >  |dh_dc| → flow is mostly N-S → transects run E-W (fix a row).
    if abs(dh_dc) >= abs(dh_dr):
        axis = 'col'
        # Active columns: columns that contain at least one active cell.
        active_positions = np.where(active.any(axis=0))[0]
    else:
        axis = 'row'
        # Active rows: rows that contain at least one active cell.
        active_positions = np.where(active.any(axis=1))[0]

    # Trim the outermost 15 % of positions to avoid near-edge artefacts.
    n_pos = len(active_positions)
    trim  = max(1, int(0.15 * n_pos))
    inner = (active_positions[trim:-trim]
             if n_pos > 2 * trim else active_positions)

    # Evenly-spaced positions within the trimmed range.
    if len(inner) < n_sections:
        inner = active_positions  # fallback: use all positions
    indices   = np.round(
        np.linspace(0, len(inner) - 1, n_sections + 2)[1:-1]
    ).astype(int)
    positions = inner[indices]

    transects = []
    for i, pos in enumerate(positions):
        if axis == 'row':
            # Horizontal (E-W) transect at row = pos.
            cols_t = np.where(active[pos, :])[0]
            if len(cols_t) < 3:
                continue
            rows_t = np.full(len(cols_t), pos, dtype=int)
            dist_t = cols_t * delc
        else:
            # Vertical (N-S) transect at col = pos.
            rows_t = np.where(active[:, pos])[0]
            if len(rows_t) < 3:
                continue
            cols_t = np.full(len(rows_t), pos, dtype=int)
            dist_t = rows_t * delr

        # Normalise distance to start at 0.
        dist_t = dist_t - dist_t.min()

        transects.append({
            'label':  f'Section {i + 1}',
            'rows':   rows_t,
            'cols':   cols_t,
            'dist_m': dist_t,
        })

    return transects


def make_transect_from_endpoints(start_xy, end_xy, transform, delr, delc,
                                  active, label='Section'):
    """
    Build a cross-section transect dict from two projected (x, y) coordinates.

    The transect is sampled along the straight line between ``start_xy`` and
    ``end_xy``, keeping only cells that fall inside the active model domain.

    Parameters
    ----------
    start_xy, end_xy : tuple of float
        Start and end projected coordinates (easting, northing) in the model
        CRS (same units as the Affine transform, typically metres EPSG:25833).
    transform : Affine
        Rasterio Affine transform for the model grid.
    delr, delc : float
        Cell size in the row (N-S) and column (E-W) directions (m).
    active : ndarray
        Boolean active-cell mask, shape (nrow, ncol).
    label : str
        Human-readable label for the transect.

    Returns
    -------
    dict with keys:
        'label'  : str
        'rows'   : ndarray (int) – row indices along the transect.
        'cols'   : ndarray (int) – column indices along the transect.
        'dist_m' : ndarray (float) – cumulative distance from start (m).
    """
    nrow, ncol = active.shape

    # Convert projected coordinates to (row, col) pixel indices.
    r0, c0 = rowcol(transform, float(start_xy[0]), float(start_xy[1]))
    r1, c1 = rowcol(transform, float(end_xy[0]),   float(end_xy[1]))

    # Clip to grid extent.
    r0, r1 = int(np.clip(r0, 0, nrow - 1)), int(np.clip(r1, 0, nrow - 1))
    c0, c1 = int(np.clip(c0, 0, ncol - 1)), int(np.clip(c1, 0, ncol - 1))

    # Sample cells along the straight line.
    n_pts  = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rows_f = np.linspace(r0, r1, n_pts)
    cols_f = np.linspace(c0, c1, n_pts)
    rows_t = np.round(rows_f).astype(int)
    cols_t = np.round(cols_f).astype(int)

    # Remove duplicate adjacent cells while preserving order.
    seen, unique = set(), []
    for r, c in zip(rows_t, cols_t):
        if (r, c) not in seen:
            seen.add((r, c))
            unique.append((r, c))
    rows_t = np.array([p[0] for p in unique])
    cols_t = np.array([p[1] for p in unique])

    # Keep only active cells.
    mask = active[rows_t, cols_t]
    rows_t, cols_t = rows_t[mask], cols_t[mask]
    if len(rows_t) < 2:
        raise ValueError(
            f"Transect '{label}' contains fewer than 2 active cells. "
            "Check that the start/end coordinates lie within the active domain.")

    # Cumulative distance from start (m).
    dr     = np.diff(rows_t, prepend=rows_t[0]) * delr
    dc     = np.diff(cols_t, prepend=cols_t[0]) * delc
    dist_t = np.cumsum(np.sqrt(dr ** 2 + dc ** 2))
    dist_t[0] = 0.0

    return {'label': label, 'rows': rows_t, 'cols': cols_t, 'dist_m': dist_t}


def limit_k_contrast(hk_arr, active_arr, max_log10_range=4.0):
    """
    Clip hydraulic conductivity so the log₁₀(K) range of active cells does not
    exceed *max_log10_range* (symmetric around the geometric mean).

    Extreme K contrasts between adjacent cells cause numerical instability in
    the groundwater solver (non-convergence, oscillating heads, NaN results).
    This function centres the log₁₀(K) distribution on the geometric mean and
    clips values outside ±(max_log10_range / 2) to reduce such artefacts.

    Parameters
    ----------
    hk_arr : ndarray
        2-D hydraulic conductivity (m/s).
    active_arr : ndarray
        Boolean active-cell mask.
    max_log10_range : float
        Maximum allowed total log₁₀(K) range for active cells.
        Default 4 (factor of 10⁴ between lowest and highest active K).

    Returns
    -------
    ndarray – K field with contrast clipped (inactive cells unchanged).
    """
    log_k = np.where(active_arr & (hk_arr > 0),
                     np.log10(np.maximum(hk_arr, 1e-20)), np.nan)
    log_k_mean = float(np.nanmean(log_k[active_arr]))
    half       = max_log10_range / 2.0
    log_k_clip = np.clip(log_k, log_k_mean - half, log_k_mean + half)
    return np.where(active_arr, 10.0 ** log_k_clip, hk_arr)

