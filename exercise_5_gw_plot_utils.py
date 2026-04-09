"""
Plotting utilities for the GEOV212 Exercise 5 groundwater model.

These functions are imported from exercise_5a_gw_model.ipynb so that the
notebook cells stay short and readable.  All functions that need spatial grid
information accept a ``grid`` dict with these keys::

    grid = {
        'dem':                 dem,
        'active':              active,
        'sw_cells':            sw_cells,
        'is_sea':              is_sea,
        'nrow':                nrow,
        'ncol':                ncol,
        'delr':                delr,
        'delc':                delc,
        'transform':           transform,
        'aquifer_thickness_m': aquifer_thickness_m,
    }

Functions
---------
_cbar                       – Matched colorbar helper.
_panel_h                    – Map panel height preserving aspect ratio.
add_map_ticks               – Easting/northing tick labels on map axes.
add_map_overlays            – Domain boundary + sea overlay on map axes.
plot_model_output           – 5-panel spatial map for one calibration.
plot_calibration_comparison – 4-panel calibration quality figure.
plot_cross_sections         – 2-D hydrogeological cross-sections.
plot_water_budget           – Water-budget bar chart + printed table.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmcrameri.cm as cmc

import exercise_5_gw_model_utils as gwu

_S_PER_YR = 365.25 * 86400.0   # seconds per year


# ── Low-level map helpers ─────────────────────────────────────────────────────

def _cbar(im, ax, label='', **kwargs):
    """Add a colorbar that exactly matches the height of the map axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.08)
    return plt.colorbar(im, cax=cax, label=label, **kwargs)


def _panel_h(nrow, ncol):
    """Return figure height (inches) for one map panel, preserving aspect ratio."""
    return 6.5 * nrow / ncol


def add_map_ticks(ax, grid, interval_km=5):
    """
    Add easting/northing tick labels (km, EPSG:25833) to a map axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    grid : dict  – Contains 'transform', 'nrow', 'ncol', 'delr', 'delc'.
    interval_km : float  – Tick spacing in km (default 5).
    """
    ax.grid(False)
    transform = grid['transform']
    nrow, ncol = grid['nrow'], grid['ncol']
    delr, delc = grid['delr'], grid['delc']

    step = interval_km * 1000.0
    x_min = transform.c
    y_top = transform.f

    x_ticks_m = np.arange(np.ceil(x_min / step) * step,
                           x_min + ncol * delr, step)
    y_ticks_m = np.arange(np.ceil((y_top - nrow * delc) / step) * step,
                           y_top, step)

    x_px = [(m - x_min) / delr - 0.5  for m in x_ticks_m]
    y_px = [(y_top - m) / delc - 0.5  for m in y_ticks_m]

    xv = [(px, m) for px, m in zip(x_px, x_ticks_m) if -0.5 <= px <= ncol - 0.5]
    yv = [(px, m) for px, m in zip(y_px, y_ticks_m) if -0.5 <= px <= nrow - 0.5]

    if xv:
        pxs, ms = zip(*xv)
        ax.set_xticks(list(pxs))
        ax.set_xticklabels([f'{m / 1000:.0f}' for m in ms], fontsize=7)
    if yv:
        pxs, ms = zip(*yv)
        ax.set_yticks(list(pxs))
        ax.set_yticklabels([f'{m / 1000:.0f}' for m in ms], fontsize=7)

    ax.set_xlabel('Easting (km, EPSG:25833)', fontsize=8)
    ax.set_ylabel('Northing (km, EPSG:25833)', fontsize=8)


def add_map_overlays(ax, grid):
    """
    Overlay the model-domain boundary and sea mask on a map axes.

    The domain boundary is drawn as a solid black contour.  Sea cells are
    shown with a light blue fill and a dashed dark-blue outline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    grid : dict  – Contains 'active', 'is_sea'.
    """
    active = grid['active']
    is_sea = grid['is_sea']

    ax.contour(active.astype(float), levels=[0.5],
               colors='k', linewidths=1.0, zorder=6)
    if is_sea.any():
        ax.contourf(is_sea.astype(float), levels=[0.5, 1.5],
                    colors=['#cce8f4'], alpha=0.28, zorder=5)
        ax.contour(is_sea.astype(float), levels=[0.5],
                   colors=['#1565C0'], linewidths=0.7, linestyles='--', zorder=6)


# ── Main plotting functions ───────────────────────────────────────────────────

def plot_model_output(head, diagnostics, hk_arr, label, grid, show_obs=None):
    """
    6-panel single-column spatial overview + water-budget figure for one scenario.

    Panels
    ------
    A  Water-table elevation (m a.s.l.) — filled contour + contour lines
    B  Depth to water table (m; diverging colormap, blue = above surface)
    C  Darcy flux magnitude (m/yr) + black flow arrows
    D  Seepage/drainage flux (mm/yr, all land) with surface-water overlay
    E  Flux to surface water (mm/yr, at river/lake cells only)
    F  log₁₀ Transmissivity (m²/s)

    After the 6 map panels a separate water-budget figure is shown if ``grid``
    contains 'rch', 'sw_raw', 'sea_raw', and 'aquifer_thickness_m'.

    Parameters
    ----------
    head : ndarray        – Modelled hydraulic head (m a.s.l.).
    diagnostics : dict    – Output dict from gwu.simulate().
    hk_arr : ndarray      – Hydraulic conductivity field (m/s).
    label : str           – Scenario name (used in titles).
    grid : dict           – Spatial grid information (see module docstring).
                            Optional extra keys for water budget:
                            'rch'      – recharge array (m/s)
                            'sw_raw'   – raw SW raster (0/1/2/3)
                            'sea_raw'  – raw sea mask (0/1)
    show_obs : DataFrame or None
        Observation DataFrame with columns 'r', 'c', 'well_no'.
        If provided, numbered markers are drawn on panel A.
    """
    dem    = grid['dem']
    active = grid['active']
    sw_cells = grid['sw_cells']
    is_sea   = grid['is_sea']
    nrow, ncol = grid['nrow'], grid['ncol']
    delr, delc = grid['delr'], grid['delc']
    aquifer_thickness_m = grid['aquifer_thickness_m']

    wt_depth = np.where(active, dem - head, np.nan)
    qx_l, qy_l, q_mag_l = gwu.compute_darcy_flux(head, hk_arr, delr, delc, active)
    sw_arr  = np.where(sw_cells, 1, 0)
    sea_int = is_sea.astype(int)
    seep_s  = gwu.seepage_flux_stats(diagnostics['drn_flux'], active,
                                     sw_arr, sea_int, delr, delc)
    seep_map = seep_s['seepage_map_mm_yr']

    T_log = np.where(active & (hk_arr > 0),
                     np.log10(np.maximum(hk_arr * aquifer_thickness_m, 1e-12)), np.nan)
    q_myr_plot  = np.where(active & np.isfinite(q_mag_l),
                           q_mag_l * _S_PER_YR, np.nan)
    seep_plot   = np.where(active & ~is_sea & (seep_map > 0), seep_map, np.nan)
    sw_flux_plot = np.where(sw_cells & ~is_sea & (seep_map > 0), seep_map, np.nan)

    ph = _panel_h(nrow, ncol)
    fig, axes = plt.subplots(6, 1, figsize=(7, 6 * ph))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.97, bottom=0.01, hspace=0.55)

    # A: water-table elevation
    vlo = np.nanpercentile(head[active], 2)
    vhi = np.nanpercentile(head[active], 98)
    _x, _y = np.arange(ncol), np.arange(nrow)
    _hm = np.ma.masked_invalid(np.where(active, head, np.nan))
    _levels = np.linspace(vlo, vhi, 21)
    cf0 = axes[0].contourf(_x, _y, _hm, levels=_levels, cmap=cmc.lapaz, extend='both')
    axes[0].contour(_x, _y, _hm, levels=_levels, colors='k', linewidths=0.25, alpha=0.4)
    axes[0].set_xlim(-0.5, ncol - 0.5)
    axes[0].set_ylim(nrow - 0.5, -0.5)
    axes[0].set_title(f'A) Water-table elevation – {label} (m a.s.l.)')
    _cbar(cf0, axes[0], 'Head (m a.s.l.)')
    if show_obs is not None and not show_obs.empty:
        _obs_cmap = plt.cm.plasma
        if 'obs_depth_m' in show_obs.columns and show_obs['obs_depth_m'].notna().any():
            _obs_d  = show_obs['obs_depth_m'].to_numpy(dtype=float)
            _finite = _obs_d[np.isfinite(_obs_d)]
            _obs_vmax = max(float(np.nanpercentile(_finite, 98)), 1.0) if len(_finite) > 0 else 1.0
            _obs_n  = Normalize(vmin=0.0, vmax=_obs_vmax)
            sc_obs0 = axes[0].scatter(show_obs['c'], show_obs['r'],
                                      c=_obs_d, cmap=_obs_cmap, norm=_obs_n,
                                      edgecolors='0.4', s=90, linewidths=1.2, zorder=8)
            _cax_d = axes[0].inset_axes([0.52, 0.02, 0.40, 0.04])
            _cb_d  = plt.colorbar(sc_obs0, cax=_cax_d, orientation='horizontal')
            _cb_d.set_label('Obs WT depth (m)', fontsize=6)
            _cax_d.tick_params(labelsize=6)
        else:
            axes[0].scatter(show_obs['c'], show_obs['r'],
                            c='yellow', edgecolors='0.4', s=90, linewidths=1.2, zorder=8)
        for _, row in show_obs.iterrows():
            axes[0].text(row['c'] + 1.0, row['r'] - 1.0,
                         str(int(row['well_no'])), fontsize=7, color='k',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                   ec='none', alpha=0.6), zorder=9)

    # B: depth to water table
    wt_valid = wt_depth[active & np.isfinite(wt_depth)]
    wt_p2  = max(np.nanpercentile(wt_valid, 2), -5.0)
    wt_p98 = min(np.nanpercentile(wt_valid, 98), 80.0)
    wt_norm = TwoSlopeNorm(vmin=min(wt_p2, -0.5), vcenter=0.0,
                           vmax=max(wt_p98, 0.5))
    im1 = axes[1].imshow(wt_depth, cmap=cmc.vik, norm=wt_norm)
    axes[1].set_title(
        f'B) Depth to water table – {label}\n'
        'Blue = above surface (springs); white = at surface; red = deep')
    _cbar(im1, axes[1], 'WT depth (m below surface)')

    # C: Darcy flux magnitude + black flow arrows
    _valid_q = q_myr_plot[np.isfinite(q_myr_plot)]
    _q_hi = float(np.nanpercentile(_valid_q, 98)) if len(_valid_q) > 0 else 1.0
    im2 = axes[2].imshow(q_myr_plot, cmap=cmc.roma, vmin=0, vmax=_q_hi)
    axes[2].set_title(f'C) Darcy flux magnitude – {label} (m/yr)')
    _cbar(im2, axes[2], '|q| (m/yr)')
    u_p = np.nan_to_num(qx_l, nan=0.0)
    v_p = np.nan_to_num(qy_l, nan=0.0)
    try:
        axes[2].streamplot(np.arange(ncol), np.arange(nrow), u_p, v_p,
                           color='k', linewidth=0.9, arrowsize=1.3,
                           arrowstyle='-|>', density=0.7, zorder=5)
    except Exception:
        pass

    # D: seepage / drainage (all land cells)
    _valid_s = seep_plot[np.isfinite(seep_plot)]
    _seep_hi = float(np.nanpercentile(_valid_s, 98)) if len(_valid_s) > 0 else 1.0
    im3 = axes[3].imshow(seep_plot, cmap=cmc.vik, vmin=0, vmax=_seep_hi)
    axes[3].set_title(f'D) Seepage / drainage flux – {label} (mm/yr)')
    _cbar(im3, axes[3], 'Seepage (mm/yr)')
    sw_ov = np.ma.masked_where(~sw_cells, np.ones(dem.shape))
    axes[3].imshow(sw_ov, cmap=cmc.lapaz, alpha=0.35, vmin=0, vmax=1)

    # E: flux to surface water (rivers/lakes only)
    _valid_sw = sw_flux_plot[np.isfinite(sw_flux_plot)]
    _sw_hi = float(np.nanpercentile(_valid_sw, 98)) if len(_valid_sw) > 0 else 1.0
    im4 = axes[4].imshow(sw_flux_plot, cmap=cmc.lapaz, vmin=0, vmax=_sw_hi)
    axes[4].set_title(f'E) Flux to surface water – {label} (mm/yr)')
    _cbar(im4, axes[4], 'SW flux (mm/yr)')

    # F: transmissivity
    im5 = axes[5].imshow(T_log, cmap=cmc.batlow)
    axes[5].set_title(f'F) log₁₀ Transmissivity – {label} (m²/s)')
    _cbar(im5, axes[5], 'log₁₀(T [m²/s])')

    for ax in axes:
        add_map_ticks(ax, grid)
        add_map_overlays(ax, grid)

    fig.suptitle(f'Model output maps – {label}', fontsize=11, y=0.98)
    plt.show()

    # ── Water budget (separate figure) ─────────────────────────────────────────
    # Requires 'rch', 'sw_raw', 'sea_raw', and 'aquifer_thickness_m' in grid.
    _rch     = grid.get('rch')
    _sw_raw  = grid.get('sw_raw')
    _sea_raw = grid.get('sea_raw')
    _b       = grid.get('aquifer_thickness_m')
    if _rch is not None and _sw_raw is not None and _sea_raw is not None:
        wb = gwu.water_budget(
            diagnostics['drn_flux'], _rch, active, _sw_raw, _sea_raw,
            delr, delc,
            head_arr=head, hk_arr=hk_arr, aquifer_thickness_m=_b,
        )
        plot_water_budget(wb, label)
    else:
        _missing = [k for k, v in
                    {'rch': _rch, 'sw_raw': _sw_raw, 'sea_raw': _sea_raw}.items()
                    if v is None]
        import warnings as _w
        _w.warn(
            f"Water budget skipped: key(s) {_missing} not found in grid dict. "
            "Add rch=rch, sw_raw=sw, sea_raw=sea when building the grid dict "
            "to enable the automatic water-budget figure.",
            UserWarning, stacklevel=2,
        )


def plot_calibration_comparison(head, diagnostics, eval_df, obs_stats, targets,
                                 label, grid):
    """
    3-panel calibration quality figure for one scenario.

    Panels
    ------
    A  Modelled water-table elevation (filled contour); observation wells coloured
       by observed depth to water table (plasma scale).
    B  Observed vs modelled depth-to-WT scatter with ME / RMSE / R²
       (same colour coding as panel A).
    C  Gaining / losing stream and lake cells map.

    Parameters
    ----------
    head : ndarray    – Modelled hydraulic head (m a.s.l.).
    diagnostics : dict – Output from gwu.simulate().
    eval_df : DataFrame – Per-well model vs observed comparison table.
    obs_stats : dict  – Summary statistics (rmse, r2, bias, …).
    targets : dict    – Surface-water / seepage target stats.
    label : str       – Scenario name.
    grid : dict       – Spatial grid information (see module docstring).
    """
    dem      = grid['dem']
    active   = grid['active']
    sw_cells = grid['sw_cells']
    nrow, ncol = grid['nrow'], grid['ncol']

    drn_elev = diagnostics['drn_elev']
    below_wt = sw_cells & np.isfinite(head) & (head < drn_elev)
    losing_map_l = np.full(dem.shape, np.nan)
    losing_map_l[sw_cells] = 0.0
    losing_map_l[below_wt] = 1.0

    ef = eval_df.copy()
    ef['dem_m']         = [dem[r, c] for r, c in zip(ef['r'], ef['c'])]
    ef['obs_depth_m']   = ef['dem_m'] - ef['obs_head_m']
    ef['model_depth_m'] = ef['dem_m'] - ef['model_head_m']
    ef['depth_resid_m'] = ef['model_depth_m'] - ef['obs_depth_m']

    _dr = ef['depth_resid_m'].dropna()
    _od = ef['obs_depth_m'].dropna()
    _md = ef['model_depth_m'].dropna()
    if len(_dr) > 1:
        _me_d   = float(_dr.mean())
        _rmse_d = float(np.sqrt((_dr ** 2).mean()))
        _ss_res = float(((_md - _od) ** 2).sum())
        _ss_tot = float(((_od - _od.mean()) ** 2).sum())
        _r2_d   = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else np.nan
    else:
        _me_d = _rmse_d = _r2_d = np.nan
    _stats_str = (f'ME = {_me_d:+.2f} m   RMSE = {_rmse_d:.2f} m   '
                  f'R² = {_r2_d:.2f}')

    # Shared colormap for observed depth-to-WT (panels A and B)
    _obs_cmap = plt.cm.plasma
    if not ef.empty and ef['obs_depth_m'].notna().any():
        _obs_arr  = ef['obs_depth_m'].to_numpy(dtype=float)
        _finite   = _obs_arr[np.isfinite(_obs_arr)]
        _obs_vmax = max(float(np.nanpercentile(_finite, 98)), 1.0) if len(_finite) > 0 else 1.0
        _obs_norm = Normalize(vmin=0.0, vmax=_obs_vmax)
    else:
        _obs_arr  = None
        _obs_norm = None

    ph = _panel_h(nrow, ncol)
    fig, axes = plt.subplots(3, 1, figsize=(7, 1.5 * ph + 8.0))
    fig.subplots_adjust(left=0.12, right=0.92, top=0.96, bottom=0.04, hspace=0.6)

    # A: modelled head map with obs wells coloured by observed depth-to-WT
    vlo = np.nanpercentile(head[active], 2)
    vhi = np.nanpercentile(head[active], 98)
    _x, _y = np.arange(ncol), np.arange(nrow)
    _hm = np.ma.masked_invalid(np.where(active, head, np.nan))
    _levels = np.linspace(vlo, vhi, 21)
    cf0 = axes[0].contourf(_x, _y, _hm, levels=_levels, cmap=cmc.lapaz, extend='both')
    axes[0].contour(_x, _y, _hm, levels=_levels, colors='k', linewidths=0.25, alpha=0.4)
    axes[0].set_xlim(-0.5, ncol - 0.5)
    axes[0].set_ylim(nrow - 0.5, -0.5)
    axes[0].set_title(f'A) Modelled water-table elevation – {label} (m a.s.l.)')
    _cbar(cf0, axes[0], 'Head (m a.s.l.)')
    if not ef.empty and 'c' in ef.columns:
        _sc_kw = dict(edgecolors='0.4', s=100, linewidths=1.2, zorder=8)
        if _obs_norm is not None:
            sc_obs_a = axes[0].scatter(ef['c'], ef['r'],
                                       c=_obs_arr, cmap=_obs_cmap, norm=_obs_norm,
                                       **_sc_kw)
            _cax_a = axes[0].inset_axes([0.52, 0.02, 0.40, 0.04])
            _cb_a  = plt.colorbar(sc_obs_a, cax=_cax_a, orientation='horizontal')
            _cb_a.set_label('Obs WT depth (m)', fontsize=6)
            _cax_a.tick_params(labelsize=6)
        else:
            axes[0].scatter(ef['c'], ef['r'], c='yellow', **_sc_kw)
        for _, row in ef.iterrows():
            axes[0].text(row['c'] + 1.5, row['r'] - 1.5,
                         str(int(row['well_no'])), fontsize=7, color='k',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                   ec='none', alpha=0.7), zorder=9)
    add_map_ticks(axes[0], grid)
    add_map_overlays(axes[0], grid)

    # B: scatter observed vs modelled depth to WT, coloured by observed depth
    if not ef.empty:
        all_d = np.concatenate([ef['obs_depth_m'].dropna(),
                                 ef['model_depth_m'].dropna()])
        lo, hi = all_d.min(), all_d.max()
        _sc_kw_b = dict(s=60, alpha=0.9, edgecolors='0.4', linewidths=0.8)
        if _obs_norm is not None:
            axes[1].scatter(ef['obs_depth_m'], ef['model_depth_m'],
                            c=_obs_arr, cmap=_obs_cmap, norm=_obs_norm, **_sc_kw_b)
        else:
            axes[1].scatter(ef['obs_depth_m'], ef['model_depth_m'],
                            color='tab:blue', **_sc_kw_b)
        for _, row in ef.iterrows():
            axes[1].annotate(str(int(row['well_no'])),
                             (row['obs_depth_m'], row['model_depth_m']),
                             fontsize=7, ha='left', va='bottom',
                             xytext=(3, 2), textcoords='offset points')
        axes[1].plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='1:1 line')
        axes[1].set_xlabel('Observed depth to water table (m below surface)')
        axes[1].set_ylabel('Modelled depth to water table (m below surface)')
        axes[1].set_title(
            f'B) Observed vs modelled depth-to-WT – {label}\n{_stats_str}',
            fontsize=8)
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].text(0.1, 0.5, 'No valid observations',
                     transform=axes[1].transAxes)
        axes[1].set_axis_off()

    # C: gaining / losing map (was D)
    im2 = axes[2].imshow(losing_map_l, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title(
        f'C) Gaining / losing reaches – {label}\n'
        f'{below_wt.sum()} of {sw_cells.sum()} SW cells losing '
        f'({targets["below_wt_fraction"]:.1%})')
    _cbar(im2, axes[2], '0 = gaining  |  1 = losing', ticks=[0, 1])
    add_map_ticks(axes[2], grid)
    add_map_overlays(axes[2], grid)

    fig.suptitle(f'Calibration quality – {label}', fontsize=11, y=0.98)
    plt.show()


def plot_cross_sections(transects, head, dem, sw, drn_flux, active,
                         delr, delc, aquifer_thickness_m, label):
    """
    Plot an overview map with cross-section locations, followed by one panel
    per cross-section perpendicular to the water-table contours.

    Overview panel (top)
    --------------------
    Water-table elevation map with coloured lines showing where each section
    is located.

    Cross-section panels
    --------------------
    - Brown fill  : below assumed aquifer base.
    - Blue fill   : saturated zone (aquifer base → water table).
    - Green fill  : unsaturated zone (water table → land surface).
    - Black line  : land surface elevation.
    - Blue line   : water-table elevation (hydraulic head).
    - Dashed grey : assumed aquifer base (land surface − aquifer_thickness_m).
    - Blue markers: surface water bodies (rivers/lakes intersecting the section).
    - Red bars    : seepage / drainage flux (mm/yr) on secondary axis.

    Parameters
    ----------
    transects : list of dict
        Output from gwu.find_cross_section_transects().
    head : ndarray
        Modelled hydraulic head (m a.s.l.).
    dem : ndarray
        Land-surface elevation (m a.s.l.).
    sw : ndarray
        Surface-water mask (0 = none, 1 = lake, 2 = river, 3 = sea).
    drn_flux : ndarray
        Drain/seepage flux (m³/s per cell; from diagnostics['drn_flux']).
    active : ndarray
        Boolean active-cell mask.
    delr, delc : float
        Cell dimensions (m).
    aquifer_thickness_m : float
        Assumed uniform aquifer thickness (m).
    label : str
        Calibration scenario name used in the figure title.
    """
    n = len(transects)
    if n == 0:
        print('No cross-sections to plot.')
        return

    nrow, ncol   = head.shape
    cell_area    = delr * delc
    _sec_colours = ['#e74c3c', '#2980b9', '#27ae60', '#e67e22', '#8e44ad']

    # Figure: 1 overview map + per section: seepage panel (1/3 height) + main section
    fig, axes = plt.subplots(
        1 + 2 * n, 1,
        figsize=(9, 3.2 + 5.1 * n),
        gridspec_kw={'height_ratios': [3.2] + [1.3, 3.8] * n},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.93, top=0.94, bottom=0.04, hspace=0.60)
    axes = axes.ravel()

    # ── Overview map: water-table elevation + section locations ──────────────
    ov = axes[0]
    head_map = np.where(active, head, np.nan)
    vlo = np.nanpercentile(head_map, 2)
    vhi = np.nanpercentile(head_map, 98)
    ov_im = ov.imshow(head_map, cmap=cmc.lapaz, vmin=vlo, vmax=vhi,
                      interpolation='nearest')
    _cbar(ov_im, ov, 'Head\n(m a.s.l.)')

    # Domain boundary
    ov.contour(active.astype(float), levels=[0.5],
               colors='k', linewidths=0.8, zorder=6)

    # Surface-water / sea overlay
    sw_ov = np.ma.masked_where(sw < 1, sw.astype(float))
    ov.imshow(sw_ov, cmap='Blues', vmin=0, vmax=4, alpha=0.40, zorder=5,
              interpolation='nearest')

    # Draw each cross-section as a thick coloured line with A/B, C/D … endpoint labels
    _endpoint_labels = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'H'), ('I', 'J')]
    for i_t, tr in enumerate(transects):
        clr = _sec_colours[i_t % len(_sec_colours)]
        ov.plot(tr['cols'], tr['rows'], color=clr, linewidth=2.5, zorder=8,
                label=tr['label'])
        mid = len(tr['rows']) // 2
        ov.text(tr['cols'][mid] + 1.5, tr['rows'][mid] - 1.5,
                tr['label'], color=clr, fontsize=8, fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        # Endpoint markers (A/B for section 1, C/D for section 2, …)
        lbl_start, lbl_end = _endpoint_labels[i_t % len(_endpoint_labels)]
        _ep_kw = dict(fontsize=9, fontweight='bold', color=clr, zorder=11,
                      ha='center', va='center',
                      bbox=dict(boxstyle='circle,pad=0.25', fc='white',
                                ec=clr, linewidth=1.2))
        ov.text(tr['cols'][0],  tr['rows'][0],  lbl_start, **_ep_kw)
        ov.text(tr['cols'][-1], tr['rows'][-1], lbl_end,   **_ep_kw)

    ov.set_xlim(-0.5, ncol - 0.5)
    ov.set_ylim(nrow - 0.5, -0.5)
    ov.set_title(
        f'Cross-section locations – water-table elevation (m a.s.l.) – {label}')
    ov.set_xlabel('Grid column', fontsize=8)
    ov.set_ylabel('Grid row', fontsize=8)
    ov.tick_params(labelsize=7)

    # ── Cross-section panels: seepage panel (above) + hydrogeological section ─
    for i_sec, (tr, clr) in enumerate(zip(transects, _sec_colours)):
        ax_seep = axes[1 + 2 * i_sec]   # small seepage panel (~1/3 height)
        ax      = axes[2 + 2 * i_sec]   # main hydrogeological cross-section

        rows_t  = tr['rows']
        cols_t  = tr['cols']
        dist_km = tr['dist_m'] / 1000.0

        elev     = dem[rows_t, cols_t]
        wt       = head[rows_t, cols_t]
        sw_t     = sw[rows_t, cols_t]
        flux     = drn_flux[rows_t, cols_t]
        aq_base  = elev - aquifer_thickness_m
        seep_myr = np.maximum(flux / cell_area, 0.0) * 1000.0 * _S_PER_YR

        ymin = float(np.nanmin(aq_base)) - 5.0
        ymax = float(np.nanmax(elev)) + 8.0

        # ── Seepage panel (1/3 height, above cross-section) ──────────────────
        bar_w = (dist_km[-1] - dist_km[0]) / max(len(dist_km), 1) * 0.8
        ax_seep.bar(dist_km, seep_myr, width=bar_w,
                    color='steelblue', alpha=0.7, edgecolor='none')
        ax_seep.set_xlim(dist_km[0], dist_km[-1])
        seep_max = max(float(seep_myr.max()), 1.0)
        ax_seep.set_ylim(0, seep_max * 1.25)
        ax_seep.set_ylabel('Seepage\n(mm/yr)', fontsize=7)
        ax_seep.tick_params(labelsize=7)
        ax_seep.set_title(
            f'{tr["label"]}  —  seepage flux  |  {label}',
            fontsize=8, color=clr)
        ax_seep.set_xticklabels([])   # x labels shown in main panel below

        # ── Main hydrogeological cross-section ────────────────────────────────
        # Fill areas
        ax.fill_between(dist_km, ymin, np.maximum(aq_base, ymin),
                        color='#a07040', alpha=0.35, label='Below aquifer base',
                        zorder=1)
        ax.fill_between(dist_km,
                        np.maximum(aq_base, ymin),
                        np.minimum(wt, elev),
                        color='#4a90d9', alpha=0.45, label='Saturated zone (aquifer)',
                        zorder=2)
        ax.fill_between(dist_km, np.minimum(wt, elev), elev,
                        color='#78c670', alpha=0.45, label='Unsaturated zone',
                        zorder=3)

        # Profile lines
        ax.plot(dist_km, elev,    'k-',  linewidth=1.5, label='Land surface',   zorder=5)
        ax.plot(dist_km, wt,      'b-',  linewidth=1.5, label='Water table',    zorder=6)
        ax.plot(dist_km, aq_base, '--',  color='#666', linewidth=0.9,
                label=f'Aquifer base (DEM − {aquifer_thickness_m:.0f} m)',       zorder=4)

        # Surface water markers
        sw_mask = (sw_t > 0) & (sw_t < 3)
        if sw_mask.any():
            ax.scatter(dist_km[sw_mask], wt[sw_mask] + 2.0,
                       marker='v', color='#1565C0', s=45, zorder=7,
                       label='Surface water (river/lake)')

        ax.set_xlim(dist_km[0], dist_km[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Distance along section (km)')
        ax.set_ylabel('Elevation (m a.s.l.)')
        ax.set_title(f'{tr["label"]} — {label}', color=clr)

        # Endpoint labels (A/B, C/D, …) at the left and right margins
        lbl_start, lbl_end = _endpoint_labels[i_sec % len(_endpoint_labels)]
        _ep_ax_kw = dict(fontsize=10, fontweight='bold', color=clr, zorder=11,
                         va='bottom', transform=ax.get_xaxis_transform(),
                         bbox=dict(boxstyle='circle,pad=0.30', fc='white',
                                   ec=clr, linewidth=1.2))
        ax.text(dist_km[0],  -0.06, lbl_start, ha='right', **_ep_ax_kw)
        ax.text(dist_km[-1], -0.06, lbl_end,   ha='left',  **_ep_ax_kw)
        ax.legend(loc='best', fontsize=6, ncol=2, framealpha=0.85)

    fig.suptitle(f'Hydrogeological cross-sections — {label}', fontsize=11, y=0.99)
    plt.show()


def plot_water_budget(wb, label):
    """
    Print a formatted water-budget table and show a bar chart for one scenario.

    Budget components are split into:
      - Lake discharge   (SW cells with sw == 1)
      - River discharge  (SW cells with sw == 2)
      - Upland seepage   (non-SW land cells)
      - Sea discharge    (Darcy face-flux estimate, falling back to residual)

    All values are shown in mm/yr normalised by active land area.

    Parameters
    ----------
    wb : dict
        Output of ``gwu.water_budget()``.
    label : str
        Scenario name used in the figure title.

    Returns
    -------
    fig : matplotlib Figure
    """
    # ── Choose sea discharge value to display ──────────────────────────────────
    sea_darcy  = wb.get('sea_discharge_darcy_mm_yr', np.nan)
    sea_resid  = wb.get('sea_discharge_residual_mm_yr', np.nan)
    has_darcy  = (sea_darcy is not None) and np.isfinite(sea_darcy)

    sea_primary        = sea_darcy  if has_darcy else sea_resid
    sea_primary_label  = 'Sea discharge\n(Darcy face)'   if has_darcy else 'Sea discharge\n(residual)'
    sea_primary_note   = '(direct Darcy estimate)'       if has_darcy else '(mass-balance residual)'

    rch    = wb.get('recharge_mm_yr',          0.0)
    lake   = wb.get('lake_discharge_mm_yr',    0.0)
    river  = wb.get('river_discharge_mm_yr',   0.0)
    seep   = wb.get('upland_seepage_mm_yr',    0.0)
    imbal  = wb.get('budget_imbalance_darcy_mm_yr',
                    wb.get('budget_imbalance_residual_mm_yr', np.nan))

    # ── Printed table ──────────────────────────────────────────────────────────
    w = 62
    print('=' * w)
    print(f'  Water budget — {label}')
    print('-' * w)
    print(f"  {'Component':<35s}  {'mm/yr':>8s}  {'Sign'}")
    print('-' * w)
    _rows = [
        ('Recharge',                         rch,           '+'),
        ('Lake discharge',                   -lake,         '-'),
        ('River discharge',                  -river,        '-'),
        ('Upland seepage (non-SW land)',      -seep,         '-'),
        (f'Sea discharge {sea_primary_note}', -sea_primary,  '-'),
    ]
    for lbl, val, sgn in _rows:
        print(f"  {lbl:<35s}  {val:>8.1f}   {sgn}")
    print('-' * w)
    if not np.isnan(imbal):
        print(f"  {'Budget imbalance (should ≈ 0)':<35s}  {imbal:>8.1f}")
    area_km2 = wb.get('land_area_m2', np.nan) / 1e6
    if not np.isnan(area_km2):
        print(f"  Active land area: {area_km2:.1f} km²")
    if has_darcy and not np.isnan(sea_resid):
        print(f"  Sea discharge residual check: {sea_resid:.1f} mm/yr "
              f"(Darcy: {sea_darcy:.1f} mm/yr)")
    print('=' * w)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    bar_items = [
        ('Recharge\n(IN)',           rch,           'forestgreen'),
        ('Lake\ndischarge',          lake,           'royalblue'),
        ('River\ndischarge',         river,          'steelblue'),
        ('Upland\nseepage',          seep,           'darkorange'),
        (sea_primary_label,          sea_primary,    'mediumpurple'),
    ]
    labels_b = [b[0] for b in bar_items]
    # Recharge shown positive (inflow), discharges shown negative (outflow)
    vals = [rch, -lake, -river, -seep, -sea_primary]
    colors = [b[2] for b in bar_items]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels_b))
    bars = ax.bar(x, vals, color=colors, alpha=0.80, edgecolor='k', linewidth=0.7)

    # If Darcy estimate is available, overlay the residual as a hatched outline bar
    if has_darcy and not np.isnan(sea_resid):
        ax.bar(x[-1], -sea_resid,
               color='none', edgecolor='mediumpurple',
               linewidth=1.5, hatch='///', alpha=1.0, label='Sea (residual check)')
        ax.legend(fontsize=8, loc='lower right')

    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel('Flux (mm/yr)   [+ = inflow, − = outflow]')
    ax.set_title(f'Catchment water budget — {label}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_b, fontsize=9)
    ax.grid(axis='y', alpha=0.4)

    # Label bar values
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ypos = val + (2.5 if val >= 0 else -4.0)
            ax.text(bar.get_x() + bar.get_width() / 2.0, ypos,
                    f'{abs(val):.0f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig
