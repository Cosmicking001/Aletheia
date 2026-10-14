"""
aletheia_surveys: Aletheia survey grid and FITS cutouts (Session 6)

fetch_cutouts(coord, fov_arcmin, surveys=OPTICAL, pixels=400) -> list of dicts
save_fits(item, folder, object_name) -> path      (FITS with provenance keywords)
check_same_grid(items) -> (ok, message)          (identical WCS, North up, East left)
make_grid_png(items, path, title, size_arcmin)    (panels, ring, scale bar, credits)

All cutouts for one object are requested with the same centre, field and pixel
count from CDS hips2fits, so they share one sky grid by construction. The
check proves it from the files themselves.
"""
import math
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.io import fits
from astropy.wcs import WCS

# name -> (HiPS ID, band label, short credit)
OPTICAL = {
    "DSS2": ("CDS/P/DSS2/red", "red", "DSS2 (STScI)"),
    "Pan-STARRS": ("CDS/P/PanSTARRS/DR1/r", "r", "Pan-STARRS1"),
    "DECaPS2": ("CDS/P/DECaPS/DR2/r", "r", "DECaPS2"),
    "Legacy DR10": ("CDS/P/DESI-Legacy-Surveys/DR10/r", "r", "DESI Legacy Surveys DR10"),
}
MIN_VALID_FRACTION = 0.9


def field_of_view(size_arcmin):
    """Field in arcmin: 3x the object, at least 4', at most 60'."""
    if size_arcmin in (None, "") or (isinstance(size_arcmin, float) and math.isnan(size_arcmin)):
        return 6.0
    return float(min(60.0, max(4.0, 3.0 * float(size_arcmin))))


def _query_hips2fits(hips_id, coord, fov_arcmin, pixels):
    from astroquery.hips2fits import hips2fits
    return hips2fits.query(hips=hips_id, width=pixels, height=pixels, projection="TAN",
                           ra=Longitude(coord.ra), dec=Latitude(coord.dec),
                           fov=Angle(fov_arcmin * u.arcmin), format="fits")


def coverage(data):
    d = np.asarray(data, dtype=float)
    finite = np.isfinite(d)
    if not finite.any():
        return 0.0
    valid = finite & (d != 0)
    frac = valid.mean()
    if np.nanstd(d[finite]) == 0:
        return 0.0
    return float(frac)


def fetch_cutouts(coord, fov_arcmin, surveys=OPTICAL, pixels=400, fetcher=None):
    fetcher = fetcher or _query_hips2fits
    items = []
    for name, (hips_id, band, credit) in surveys.items():
        item = dict(survey=name, hips_id=hips_id, band=band, credit=credit, fov_arcmin=fov_arcmin,
                    hdu=None, status="", coverage=0.0)
        try:
            hdul = fetcher(hips_id, coord, fov_arcmin, pixels)
            hdu = hdul[0]
            cov = coverage(hdu.data)
            item.update(hdu=hdu, coverage=cov,
                        status="ok" if cov >= MIN_VALID_FRACTION else "no coverage")
        except Exception as exc:
            item["status"] = "error: " + " ".join(str(exc).split())[:70]
        items.append(item)
    return items


def save_fits(item, folder, object_name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in object_name)
    path = folder / f"{safe}_{item['survey'].replace(' ', '_')}.fits"
    hdu = fits.PrimaryHDU(data=np.asarray(item["hdu"].data, dtype=np.float32), header=item["hdu"].header)
    h = hdu.header
    h["ALETHOBJ"] = (object_name[:68], "Aletheia field name")
    h["SURVEY"] = (item["survey"], "Survey")
    h["HIPS_ID"] = (item["hips_id"], "CDS HiPS identifier")
    h["FOVAMIN"] = (round(item["fov_arcmin"], 3), "Field of view, arcmin")
    h["CREDIT"] = (item["credit"][:68], "Survey credit")
    h["ORIGIN"] = ("Aletheia via CDS hips2fits", "")
    hdu.writeto(path, overwrite=True)
    return path


def verify_fits(path):
    """Reopen a saved FITS file: data present, celestial WCS valid."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        w = WCS(hdul[0].header)
        ok = data is not None and data.ndim == 2 and w.has_celestial and w.naxis == 2
        return ok, f"{data.shape[1]}x{data.shape[0]} px" if data is not None else "no data"


def _grid_signature(header):
    w = WCS(header)
    m = w.pixel_scale_matrix * 3600.0  # arcsec per pixel
    return dict(crval=tuple(np.round(w.wcs.crval, 7)), crpix=tuple(np.round(w.wcs.crpix, 3)),
                matrix=np.round(m, 6), shape=(header.get("NAXIS1"), header.get("NAXIS2")),
                ctype=tuple(w.wcs.ctype))


def check_same_grid(items):
    good = [i for i in items if i["status"] == "ok"]
    if not good:
        return False, "no survey covers this field"
    sigs = [_grid_signature(i["hdu"].header) for i in good]
    ref = sigs[0]
    for s in sigs[1:]:
        if s["crval"] != ref["crval"] or s["crpix"] != ref["crpix"] or s["shape"] != ref["shape"] \
                or s["ctype"] != ref["ctype"] or not np.allclose(s["matrix"], ref["matrix"], atol=1e-6):
            return False, "grids differ between surveys"
    m = ref["matrix"]
    north_up = abs(m[0, 1]) < 1e-6 * abs(m[0, 0]) + 1e-9 and abs(m[1, 0]) < 1e-6 * abs(m[1, 1]) + 1e-9
    east_left = m[0, 0] < 0 and m[1, 1] > 0
    scale = abs(m[1, 1])
    if not (north_up and east_left):
        return False, "orientation is not North up, East left"
    return True, f"{len(good)} surveys on one grid, {scale:.3f}\"/px, North up, East left"


def _stretch(data):
    d = np.asarray(data, dtype=float)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros_like(d)
    lo, hi = np.nanpercentile(d[finite], [0.5, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    x = np.arcsinh(10 * x) / np.arcsinh(10)
    x[~finite] = 0
    return x


def make_grid_png(items, path, title, size_arcmin=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0), facecolor="black")
    axes = np.atleast_1d(axes)
    credits = []
    for ax, it in zip(axes, items):
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333333")
        ax.set_title(f"{it['survey']}  {it['band']}", color="#E8E6E1", fontsize=10, family="monospace")
        if it["status"] != "ok":
            ax.text(0.5, 0.5, it["status"], color="#9A9A9A", ha="center", va="center",
                    fontsize=9, family="monospace", transform=ax.transAxes, wrap=True)
            continue
        credits.append(it["credit"])
        img = _stretch(it["hdu"].data)
        ny, nx = img.shape
        ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest")
        scale_arcmin = it["fov_arcmin"] / nx  # arcmin per pixel
        if size_arcmin:
            r_px = 0.5 * float(size_arcmin) / scale_arcmin
            ax.add_patch(Circle((nx / 2, ny / 2), r_px, fill=False, ec="#5FD4C4", lw=0.8, ls="--"))
        bar_arcmin = 1.0 if it["fov_arcmin"] <= 10 else 5.0
        bar_px = bar_arcmin / scale_arcmin
        ax.plot([0.06 * nx, 0.06 * nx + bar_px], [0.06 * ny, 0.06 * ny], color="white", lw=1.5)
        ax.text(0.06 * nx, 0.09 * ny, f"{bar_arcmin:g}'", color="white", fontsize=8, family="monospace")
        ax.annotate("N", xy=(0.92 * nx, 0.80 * ny), xytext=(0.92 * nx, 0.92 * ny), color="white", fontsize=8,
                    ha="center", arrowprops=dict(arrowstyle="<-", color="white", lw=0.8))
        ax.annotate("E", xy=(0.80 * nx, 0.80 * ny), xytext=(0.68 * nx, 0.80 * ny), color="white", fontsize=8,
                    va="center", arrowprops=dict(arrowstyle="<-", color="white", lw=0.8))
    fig.suptitle(title, color="#E8E6E1", fontsize=12, family="monospace")
    fig.text(0.5, 0.02, "Data: " + " · ".join(credits) + " · cutouts via CDS hips2fits",
             color="#9A9A9A", ha="center", fontsize=7, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    return path
