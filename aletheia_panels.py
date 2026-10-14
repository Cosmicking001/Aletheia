"""
aletheia_panels: Aletheia full survey panel (Session 7)

resolve_surveys()                          -> registry with the HiPS ID chosen for each survey
fetch_panel(coord, fov, registry)          -> dict of cutouts (grey FITS + colour images)
difference_images(panel)                   -> H-alpha minus scaled r for IPHAS / VPHAS+ / SHS
coverage_note(panel)                       -> text: which surveys cover the field
make_panel_png(...)                        -> overview figure with highlighted coverage banner
make_annotated_png(...)                    -> one cutout with RA/Dec axes, grid, ring, scale, credit
All cutouts of one field share one sky grid (same centre, field, pixels; North up, East left).
"""
import math
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.io import fits
from astropy.wcs import WCS

MOCSERVER = "https://alasky.cds.unistra.fr/MocServer/query"

# key: (label, group, kind, [known IDs in order], MocServer pattern, [preferred substrings], credit)
REGISTRY = [
    ("dss2_red", "DSS2 red", "Optical", "grey", ["CDS/P/DSS2/red"], "*DSS2*", ["red"], "DSS2 (STScI)"),
    ("ps_r", "Pan-STARRS r", "Optical", "grey", ["CDS/P/PanSTARRS/DR1/r"], "*PanSTARRS/DR1*", ["/r"], "Pan-STARRS1"),
    ("decaps_r", "DECaPS2 r", "Optical", "grey", ["CDS/P/DECaPS/DR2/r"], "*DECaPS/DR2*", ["/r"], "DECaPS2"),
    ("legacy_r", "Legacy DR10 r", "Optical", "grey", ["CDS/P/DESI-Legacy-Surveys/DR10/r"], "*Legacy-Surveys/DR10*", ["/r"], "DESI Legacy Surveys"),
    ("dss2_col", "DSS2 colour", "Colour", "colour", ["CDS/P/DSS2/color"], "*DSS2*", ["color"], "DSS2 (STScI)"),
    ("ps_col", "Pan-STARRS colour", "Colour", "colour", ["CDS/P/PanSTARRS/DR1/color-z-zg-g", "CDS/P/PanSTARRS/DR1/color-i-r-g"], "*PanSTARRS/DR1*", ["color"], "Pan-STARRS1"),
    ("decaps_col", "DECaPS2 colour", "Colour", "colour", ["CDS/P/DECaPS/DR2/color"], "*DECaPS/DR2*", ["color"], "DECaPS2"),
    ("legacy_col", "Legacy DR10 colour", "Colour", "colour", ["CDS/P/DESI-Legacy-Surveys/DR10/color"], "*Legacy-Surveys/DR10*", ["color"], "DESI Legacy Surveys"),
    ("iphas_col", "IPHAS colour", "Colour", "colour", [], "*IPHAS*", ["color"], "IPHAS"),
    ("vphas_col", "VPHAS+ colour", "Colour", "colour", ["CDS/P/VPHAS/DR4/color"], "*VPHAS*", ["color"], "VPHAS+ (ESO)"),
    ("iphas_ha", "IPHAS H-alpha", "H-alpha", "grey", ["CDS/P/IPHAS/DR2/halpha"], "*IPHAS*", ["halpha"], "IPHAS"),
    ("iphas_r", "IPHAS r", "H-alpha", "grey", ["CDS/P/IPHAS/DR2/r"], "*IPHAS*", ["/r"], "IPHAS"),
    ("vphas_ha", "VPHAS+ H-alpha", "H-alpha", "grey", ["CDS/P/VPHAS/DR4/Halpha"], "*VPHAS*", ["halpha"], "VPHAS+ (ESO)"),
    ("vphas_r", "VPHAS+ r", "H-alpha", "grey", ["CDS/P/VPHAS/DR4/r"], "*VPHAS*", ["/r"], "VPHAS+ (ESO)"),
    ("shs_ha", "SHS H-alpha", "H-alpha", "grey", ["CDS/P/SHS"], "*SHS*", ["ha", "shs"], "SHS (AAO/UKST)"),
    ("shs_sr", "SHS SR", "H-alpha", "grey", [], "*SHS*", ["sr"], "SHS (AAO/UKST)"),
    ("shassa", "SHASSA H-alpha", "H-alpha", "grey", ["CDS/P/SHASSA/CC", "CDS/P/SHASSA/H"], "*SHASSA*", ["/cc", "/h"], "SHASSA (Gaustad+ 2001)"),
    ("vtss", "VTSS H-alpha", "H-alpha", "grey", [], "*VTSS*", ["ha"], "VTSS (Dennison+ 1998)"),
    ("unwise_w1", "unWISE W1", "Infrared", "grey", ["CDS/P/unWISE/W1"], "*unWISE*", ["w1"], "unWISE (WISE/NEOWISE)"),
    ("allwise_w3", "AllWISE W3 12um", "Infrared", "grey", ["CDS/P/allWISE/W3"], "*allWISE*", ["w3"], "AllWISE"),
    ("allwise_w4", "AllWISE W4 22um", "Infrared", "grey", ["CDS/P/allWISE/W4"], "*allWISE*", ["w4"], "AllWISE"),
    ("allwise_col", "AllWISE colour", "Infrared", "colour", ["CDS/P/allWISE/color"], "*allWISE*", ["color"], "AllWISE"),
    ("galex_nuv", "GALEX NUV", "Ultraviolet", "grey", ["CDS/P/GALEXGR6_7/NUV"], "*GALEX*", ["nuv"], "GALEX (NASA)"),
    ("galex_fuv", "GALEX FUV", "Ultraviolet", "grey", ["CDS/P/GALEXGR6_7/FUV"], "*GALEX*", ["fuv"], "GALEX (NASA)"),
    ("galex_col", "GALEX colour", "Ultraviolet", "colour", ["CDS/P/GALEXGR6_7/color"], "*GALEX*", ["color"], "GALEX (NASA)"),
    ("nvss", "NVSS 1.4 GHz", "Radio", "grey", ["CDS/P/NVSS"], "*NVSS*", ["nvss"], "NVSS (NRAO)"),
]
GROUPS = ["Optical", "Colour", "H-alpha", "Infrared", "Ultraviolet", "Radio"]
DIFF_PAIRS = [("IPHAS", "iphas_ha", "iphas_r"), ("VPHAS+", "vphas_ha", "vphas_r"), ("SHS", "shs_ha", "shs_sr")]
MIN_VALID = 0.9


def _moc_ids(pattern, cache):
    if pattern in cache:
        return cache[pattern]
    import requests
    try:
        r = requests.get(MOCSERVER, params={"ID": pattern, "get": "id", "fmt": "json"}, timeout=60)
        r.raise_for_status()
        ids = [i for i in r.json() if isinstance(i, str)]
    except Exception:
        ids = []
    cache[pattern] = ids
    return ids


def resolve_surveys(lister=None):
    """Pick a registered HiPS ID for every survey. Returns list of dicts."""
    cache, out = {}, []
    for key, label, group, kind, known, pattern, prefer, credit in REGISTRY:
        ids = lister(pattern) if lister else _moc_ids(pattern, cache)
        chosen = next((k for k in known if k in ids), None)
        if chosen is None and ids:
            def score(i):
                low = i.lower()
                s = sum(2 for p in prefer if p in low)
                if kind == "colour":
                    s += 3 if ("color" in low or "rgb" in low) else -5
                else:
                    s -= 5 if ("color" in low or "rgb" in low) else 0
                return s
            best = max(ids, key=score)
            chosen = best if score(best) > 0 else None
        if chosen is None and known and not ids:
            chosen = known[0]  # MocServer unreachable: fall back to the known ID
        out.append(dict(key=key, label=label, group=group, kind=kind, hips_id=chosen, credit=credit,
                        alternatives=[i for i in ids if i != chosen][:6]))
    return out


def field_of_view(size_arcmin, mode="standard", custom_arcmin=None):
    s = None if size_arcmin in (None, "") else float(size_arcmin)
    if mode == "custom" and custom_arcmin:
        return float(custom_arcmin)
    if mode == "wide":
        return float(min(120.0, max(30.0, 10.0 * (s or 3.0))))
    return float(min(60.0, max(4.0, 3.0 * s))) if s else 6.0


def grid_wcs(coord, fov_arcmin, pixels):
    """The TAN grid hips2fits uses: centre, square pixels, North up, East left."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.crpix = [pixels / 2 + 0.5, pixels / 2 + 0.5]
    s = fov_arcmin / 60.0 / pixels
    w.wcs.cdelt = [-s, s]
    return w


def _fetch(hips_id, coord, fov, pixels, fmt):
    from astroquery.hips2fits import hips2fits
    return hips2fits.query(hips=hips_id, width=pixels, height=pixels, projection="TAN",
                           ra=Longitude(coord.ra), dec=Latitude(coord.dec),
                           fov=Angle(fov * u.arcmin), format=fmt)


def _grey_coverage(d):
    d = np.asarray(d, dtype=float)
    f = np.isfinite(d)
    if not f.any() or np.nanstd(d[f]) == 0:
        return 0.0
    return float((f & (d != 0)).mean())


def _colour_coverage(rgb):
    """Fraction of pixels that are not pure black. Dark sky is still data."""
    a = np.asarray(rgb, dtype=float)
    if a.ndim != 3 or a.std() == 0:
        return 0.0
    covered = float((a[..., :3].sum(axis=2) > 0).mean())
    return 1.0 if covered >= 0.5 else covered


def fetch_panel(coord, fov, registry, pixels=400, fetcher=None):
    fetcher = fetcher or _fetch
    panel = {}
    for s in registry:
        item = dict(s, fov=fov, pixels=pixels, data=None, header=None, status="not registered", coverage=0.0)
        if s["hips_id"]:
            try:
                if s["kind"] == "grey":
                    hdu = fetcher(s["hips_id"], coord, fov, pixels, "fits")[0]
                    cov = _grey_coverage(hdu.data)
                    item.update(data=np.asarray(hdu.data, dtype=float), header=hdu.header, coverage=cov)
                else:
                    rgb = np.asarray(fetcher(s["hips_id"], coord, fov, pixels, "jpg"))
                    cov = _colour_coverage(rgb)
                    # JPEG rows run top (North) to bottom; flip so row 0 is South, like FITS
                    item.update(data=rgb[::-1].copy(), coverage=cov)
                item["status"] = "ok" if item["coverage"] >= MIN_VALID else "no coverage"
            except Exception as exc:
                item["status"] = "no coverage" if "500" in str(exc) or "404" in str(exc) else \
                    "error: " + " ".join(str(exc).split())[:60]
        panel[s["key"]] = item
    return panel


def _robust_sigma(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def difference_images(panel):
    """H-alpha minus r scaled on stars (continuum subtraction)."""
    out = {}
    for name, ha_key, r_key in DIFF_PAIRS:
        ha, r = panel.get(ha_key), panel.get(r_key)
        if not ha or not r or ha["status"] != "ok" or r["status"] != "ok":
            continue
        a = ha["data"] - np.nanmedian(ha["data"])
        b = r["data"] - np.nanmedian(r["data"])
        sb = _robust_sigma(b)
        stars = np.isfinite(a) & np.isfinite(b) & (b > 8 * sb)
        if stars.sum() < 10:
            continue
        k = float(np.median(a[stars] / b[stars]))
        out[name] = dict(data=a - k * b, k=k, label=f"{name} H-alpha minus r", credit=ha["credit"])
    return out


def emission_contrast(diff, fov, size_arcmin):
    """Mean inside the object ring minus median of a surrounding annulus, in units of the
    annulus noise per pixel. The mean (not the median) keeps hollow shells visible."""
    n = diff.shape[0]
    r_px = 0.5 * float(size_arcmin) / (fov / n)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5)
    inside = diff[rr <= r_px]
    outside = diff[(rr > 1.3 * r_px) & (rr < 2.5 * r_px)]
    s = _robust_sigma(outside)
    if not inside.size or not np.isfinite(s) or s == 0:
        return float("nan")
    return float((np.nanmean(inside) - np.nanmedian(outside)) / s)


def coverage_note(panel):
    have = [p["label"] for p in panel.values() if p["status"] == "ok"]
    missing = [p["label"] for p in panel.values() if p["status"] not in ("ok", "not registered")]
    unavailable = [p["label"] for p in panel.values() if p["status"] == "not registered"]
    total = len(have) + len(missing)
    note = f"This field is covered by {len(have)} of {total} surveys."
    if missing:
        note += " Not covered here: " + ", ".join(missing) + "."
    if unavailable:
        note += " Not available at CDS: " + ", ".join(unavailable) + "."
    return note, have, missing


def _stretch(d, symmetric=False):
    d = np.asarray(d, dtype=float)
    f = np.isfinite(d)
    if not f.any():
        return np.zeros_like(d)
    if symmetric:
        s = np.nanpercentile(np.abs(d[f]), 99.5) or 1.0
        x = np.clip(d / s, -1, 1) * 0.5 + 0.5
    else:
        lo, hi = np.nanpercentile(d[f], [0.5, 99.7])
        hi = hi if hi > lo else lo + 1
        x = np.clip((d - lo) / (hi - lo), 0, 1)
        x = np.arcsinh(10 * x) / np.arcsinh(10)
    x[~f] = 0
    return x


def _show(ax, item_or_diff, is_diff=False, muted=False):
    d = item_or_diff["data"]
    if d.ndim == 3:
        ax.imshow(d[..., :3].astype(np.uint8), origin="lower", interpolation="nearest")
        return
    if muted and not is_diff:
        d = mute_stars(d)
    ax.imshow(nebula_stretch(d, symmetric=is_diff), cmap="gray", origin="lower", interpolation="nearest")


def make_panel_png(panel, diffs, path, title, size_arcmin, note, fov, muted=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    rows = []
    for g in GROUPS:
        items = [p for p in panel.values() if p["group"] == g and p["status"] == "ok"]
        if g == "H-alpha":
            items += [dict(d, group=g, is_diff=True) for d in diffs.values()]
        if items:
            rows.append((g, items))
    ncol = max([len(i) for _, i in rows] + [1])
    nrow = max(len(rows), 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow + 1.2), facecolor="black", squeeze=False)
    credits = []
    for r, (g, items) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#222222")
            if c >= len(items):
                ax.axis("off")
                continue
            it = items[c]
            _show(ax, it, it.get("is_diff", False), muted)
            if it["credit"] not in credits:
                credits.append(it["credit"])
            n = it["data"].shape[0]
            if size_arcmin:
                ax.add_patch(Circle((n / 2, n / 2), 0.5 * float(size_arcmin) / (fov / n), fill=False,
                                    ec="#5FD4C4", lw=0.7, ls="--"))
            ax.set_title(it["label"], color="#E8E6E1", fontsize=8, family="monospace")
        axes[r][0].set_ylabel(g, color="#9A9A9A", fontsize=9, family="monospace")
    if muted:
        title = "VIEWING AID (stars muted, not for measurement) · " + title
    fig.suptitle(title, color="#E8E6E1", fontsize=12, family="monospace", y=0.995)
    fig.text(0.5, 0.955, note, color="#F2A65A", ha="center", fontsize=8, family="monospace", wrap=True)
    fig.text(0.5, 0.008, "Data: " + " · ".join(credits) + " · cutouts via CDS hips2fits",
             color="#9A9A9A", ha="center", fontsize=6.5, family="monospace", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=120, facecolor="black")
    plt.close(fig)
    return path


def make_annotated_png(item, wcs, path, title, size_arcmin, is_diff=False, muted=False):
    """One cutout with RA/Dec axes and grid printed on it: ready to download and share."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(7.2, 7.6), facecolor="black")
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor("black")
    _show(ax, item, is_diff, muted)
    if muted:
        title = "VIEWING AID: stars muted · " + title
    n = item["data"].shape[0]
    fov = item.get("fov") or (abs(wcs.wcs.cdelt[1]) * 60 * n)
    scale = fov / n
    if size_arcmin:
        ax.add_patch(Circle((n / 2 - 0.5, n / 2 - 0.5), 0.5 * float(size_arcmin) / scale, fill=False,
                            ec="#5FD4C4", lw=1.0, ls="--"))
    for coord_ax, label in ((ax.coords[0], "RA (J2000)"), (ax.coords[1], "Dec (J2000)")):
        coord_ax.set_axislabel(label, color="#E8E6E1", fontsize=9)
        coord_ax.set_ticklabel(color="#E8E6E1", size=8)
        coord_ax.set_ticks(color="#E8E6E1")
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    ax.coords.grid(color="#5FD4C4", alpha=0.25, linestyle=":", linewidth=0.6)
    bar = 1.0 if fov <= 10 else (5.0 if fov <= 40 else 10.0)
    ax.plot([0.05 * n, 0.05 * n + bar / scale], [0.05 * n, 0.05 * n], color="white", lw=1.6)
    ax.text(0.05 * n, 0.075 * n, f"{bar:g}'", color="white", fontsize=9)
    ax.set_title(title, color="#E8E6E1", fontsize=10, family="monospace")
    fig.text(0.5, 0.012, f"{item['credit']} · cutout via CDS hips2fits · Aletheia", color="#9A9A9A",
             ha="center", fontsize=7, family="monospace")
    fig.savefig(path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return path


def save_grey_fits(item, wcs, path, field_name):
    h = item["header"] if item.get("header") is not None else wcs.to_header()
    hdu = fits.PrimaryHDU(np.asarray(item["data"], dtype=np.float32), header=h)
    hdu.header["ALETHOBJ"] = (field_name[:68], "Aletheia field name")
    hdu.header["SURVEY"] = (item["label"][:68], "Survey")
    if item.get("hips_id"):
        hdu.header["HIPS_ID"] = (item["hips_id"][:68], "CDS HiPS identifier")
    hdu.header["CREDIT"] = (item["credit"][:68], "Survey credit")
    hdu.writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------- Session 7b: processing
def _bg_sigma(d):
    f = np.isfinite(d)
    med = float(np.nanmedian(d[f]))
    return med, float(_robust_sigma(d[f] - med))


def mute_stars(data, size_px=15, nsigma=4.0, grow=2):
    """Viewing aid only: replace compact bright sources with the local median background.
    Extended nebulosity (wider than the median window) is kept. Never used for measurements."""
    from scipy import ndimage
    d = np.asarray(data, dtype=float).copy()
    f = np.isfinite(d)
    if not f.any():
        return d
    fill = np.where(f, d, np.nanmedian(d[f]))
    bg = ndimage.median_filter(fill, size=size_px)
    resid = fill - bg
    s = _robust_sigma(resid)
    stars = resid > nsigma * s
    if grow:
        stars = ndimage.binary_dilation(stars, iterations=grow)
    out = fill.copy()
    out[stars] = bg[stars]
    out[~f] = np.nan
    return out


def nebula_stretch(data, symmetric=False, low_sigma=2.0, high_sigma=10.0):
    """Stretch anchored on the sky background so faint emission becomes visible."""
    d = np.asarray(data, dtype=float)
    f = np.isfinite(d)
    if not f.any():
        return np.zeros_like(d)
    med, s = _bg_sigma(d)
    s = s if s > 0 else 1.0
    if symmetric:
        x = np.clip((d - med) / (8 * s), -1, 1) * 0.5 + 0.5
    else:
        x = np.clip((d - (med - low_sigma * s)) / ((high_sigma + low_sigma) * s), 0, 1)
        x = np.arcsinh(6 * x) / np.arcsinh(6)
    x[~f] = 0
    return x


def emission_contrast_smoothed(diff, fov, size_arcmin, star_ref=None, smooth_px=3.0, protect_object=False):
    """Like emission_contrast, but stars (found in the r image) are masked and the
    difference image is smoothed first, so diffuse emission is measured as a whole.
    protect_object=True never masks inside the object itself (Session 13: a compact PN
    looks star-like in r and was masking its own emission)."""
    from scipy import ndimage
    d = np.asarray(diff, dtype=float).copy()
    if star_ref is not None:
        r = np.asarray(star_ref, dtype=float)
        rf = np.where(np.isfinite(r), r, np.nanmedian(r))
        res = rf - ndimage.median_filter(rf, size=15)
        mask = ndimage.binary_dilation(res > 4 * _robust_sigma(res), iterations=2)
        if protect_object:
            n = d.shape[0]
            yy, xx = np.mgrid[:n, :n]
            mask &= np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) > 0.5 * float(size_arcmin) / (fov / n) + 3
        d[mask] = np.nan
    fill = np.where(np.isfinite(d), d, 0.0)
    w = ndimage.gaussian_filter(np.isfinite(d).astype(float), smooth_px)
    sm = ndimage.gaussian_filter(fill, smooth_px) / np.where(w > 0.2, w, np.nan)
    return emission_contrast(sm, fov, size_arcmin)


# ---------------------------------------------------------------- Session 7c: Gaia-guided muting
MIN_STAR_RADIUS_PX = 3.0  # a star always spans a few pixels, however coarse the image


def gaia_radius_arcsec(gmag):
    """Mask radius around a Gaia star: brighter stars have bigger halos."""
    g = np.asarray(gmag, dtype=float)
    return np.clip(2.5 + 1.6 * (19.0 - g), 2.5, 40.0)


def mute_stars_gaia(data, wcs, ra_deg, dec_deg, gmag, nsigma=1.5, ring_px=4, protect_px=0.0,
                    return_radii=False, max_halo_arcsec=60.0, seed=0):
    """Viewing aid only. For each catalogued Gaia DR3 star: pixels inside its disk that stand
    above the surroundings are replaced by values interpolated from the unmasked pixels around
    the disk, plus noise matching the sky. Rules (Session 7d):
      - a disk grows (stars brighter than G=16) only while extra light shows on at least 3 of
        4 sides: stellar halos are round, nebular gradients are one-sided;
      - halo limit 60 arcsec, whatever the pixel scale;
      - sources inside the protected object zone (the target itself) are never muted;
      - pixels with no Gaia star are never touched."""
    from scipy import ndimage
    from astropy.coordinates import SkyCoord
    d = np.asarray(data, dtype=float)
    f = np.isfinite(d)
    if not f.any() or len(ra_deg) == 0:
        return (d.copy(), np.array([])) if return_radii else d.copy()
    rng = np.random.default_rng(seed)
    fill = np.where(f, d, np.nanmedian(d[f]))
    s = _robust_sigma(fill - ndimage.median_filter(fill, size=15))
    ny, nx = d.shape
    gmag = np.asarray(gmag, dtype=float)
    x, y = wcs.world_to_pixel(SkyCoord(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg))
    scale = abs(wcs.pixel_scale_matrix[1, 1]) * 3600
    rpx = np.maximum(MIN_STAR_RADIUS_PX, gaia_radius_arcsec(gmag) / scale)
    rmax = max(40.0, max_halo_arcsec / scale)
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    out = fill.copy()
    mask = np.zeros(fill.shape, dtype=bool)
    for i in np.argsort(-rpx):  # big (bright) stars first
        xi, yi, ri = x[i], y[i], rpx[i]
        if protect_px > 0 and np.hypot(xi - cx, yi - cy) <= protect_px:
            rpx[i] = 0.0
            continue  # the target's own catalogued source
        if not (-ri <= xi < nx + ri and -ri <= yi < ny + ri):
            continue
        while gmag[i] < 16 and ri < rmax:
            x0, x1 = int(max(0, xi - ri - 9)), int(min(nx, xi + ri + 10))
            y0, y1 = int(max(0, yi - ri - 9)), int(min(ny, yi + ri + 10))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r2 = (xx - xi) ** 2 + (yy - yi) ** 2
            shell = (r2 > ri ** 2) & (r2 <= (ri + 2) ** 2)
            outer = (r2 > (ri + 4) ** 2) & (r2 <= (ri + 8) ** 2)
            patch = fill[y0:y1, x0:x1]
            sides = 0
            for qx, qy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                q = ((xx - xi) * qx >= 0) & ((yy - yi) * qy >= 0)
                sh, ou = shell & q, outer & q
                if sh.sum() >= 3 and ou.sum() >= 3 and np.median(patch[sh]) > np.median(patch[ou]) + 2 * s:
                    sides += 1
            if sides < 3:
                break
            ri += 1.0
        rpx[i] = ri
        x0, x1 = int(max(0, xi - ri)), int(min(nx, xi + ri + 1))
        y0, y1 = int(max(0, yi - ri)), int(min(ny, yi + ri + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - xi) ** 2 + (yy - yi) ** 2 <= ri ** 2
    # Fill every star disk by interpolating from the unmasked pixels around it (normalised
    # convolution, widening until every hole is filled). Structure running through a disk,
    # such as a nebular arc, is carried across it instead of being flattened.
    good = (~ndimage.binary_dilation(mask, iterations=2)).astype(float)  # skip star wings at disk edges
    interp = np.full_like(fill, np.nan)
    for sig in (2.0, 4.0, 8.0, 16.0, 32.0):
        num = ndimage.gaussian_filter(fill * good, sig)
        den = ndimage.gaussian_filter(good, sig)
        ok = np.isnan(interp) & (den > 0.15)
        interp[ok] = num[ok] / den[ok]
    interp[np.isnan(interp)] = np.median(fill[~mask]) if (~mask).any() else np.median(fill)
    hit = mask & (fill > interp + nsigma * s)
    out[hit] = interp[hit] + rng.normal(0.0, s, hit.sum())
    if protect_px > 0:  # compact candidates: the object zone is shown untouched
        yy, xx = np.mgrid[:ny, :nx]
        keep = np.hypot(xx - cx, yy - cy) <= protect_px
        out[keep] = fill[keep]
    out[~f] = np.nan
    return (out, rpx) if return_radii else out


def structure_preserved(orig, muted, wcs, size_arcmin, window_px=25):
    """Large-scale shape of the nebula after muting, compared with before, measured on ALL
    pixels of the object zone, including those under stars. A flattened 'plug' shows up here."""
    from scipy import ndimage
    o = np.asarray(orig, dtype=float)
    m = np.asarray(muted, dtype=float)
    of = np.where(np.isfinite(o), o, np.nanmedian(o))
    mf = np.where(np.isfinite(m), m, np.nanmedian(m))
    n = o.shape[0]
    rz = max(3.0, 0.5 * float(size_arcmin) / (abs(wcs.pixel_scale_matrix[1, 1]) * 60))
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    zone = rr <= rz
    ann = (rr > 1.3 * rz) & (rr < min(2.5 * rz, n / 2 - 1))
    if ann.sum() < 200:
        ann = rr > 1.3 * rz
    lb = ndimage.median_filter(of, size=window_px)
    la = ndimage.median_filter(mf, size=window_px)
    sky_b, sky_a = np.median(lb[ann]), np.median(la[ann])
    s = _robust_sigma(lb[ann])
    neb = zone & (lb > sky_b + 3 * s)
    if neb.sum() < 20:
        return float("nan")
    before = lb[neb] - sky_b
    after = la[neb] - sky_a
    return float(np.sum(np.clip(after, 0, None)) / np.sum(before))


def protect_radius_px(size_arcmin, fov_arcmin, pixels):
    """Zone left untouched by star muting: the whole object zone for candidates under 2'."""
    if size_arcmin in (None, "") or float(size_arcmin) >= 2.0:
        return 0.0
    return (0.5 * float(size_arcmin) + 0.25) / (fov_arcmin / pixels)


def muting_scores(orig, muted, wcs, size_arcmin, star_xy, star_r_px):
    """Nebula retained and stars removed, measured so that neither number can be faked.
    Nebula retained: median brightness of nebula pixels that lie AWAY from every catalogued
    Gaia star (so removing a star can never look like losing nebula), relative to the sky.
    Stars removed: share of Gaia G<17 stars, visible in the original, whose peak after muting
    is within 5 sigma of the LOCAL background (so stars on top of nebulosity are judged fairly)."""
    from scipy import ndimage
    o = np.asarray(orig, dtype=float)
    m = np.asarray(muted, dtype=float)
    n = o.shape[0]
    scale_arcmin = abs(wcs.pixel_scale_matrix[1, 1]) * 60
    rz = max(3.0, 0.5 * float(size_arcmin) / scale_arcmin)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    zone = rr <= rz
    ann = (rr > 1.3 * rz) & (rr < min(2.5 * rz, n / 2 - 1))
    if ann.sum() < 200:
        ann = rr > 1.3 * rz
    near_star = np.zeros_like(zone)
    for (x, y), r in zip(star_xy, star_r_px):
        x0, x1 = int(max(0, x - r)), int(min(n, x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(n, y + r + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        sy, sx = np.mgrid[y0:y1, x0:x1]
        near_star[y0:y1, x0:x1] |= (sx - x) ** 2 + (sy - y) ** 2 <= r ** 2
    of = np.where(np.isfinite(o), o, np.nanmedian(o))
    sm = ndimage.gaussian_filter(of, 2.0)
    clean_ann = ann & ~near_star
    s_sm = _robust_sigma(sm[clean_ann])
    neb = zone & ~near_star & (sm > np.median(sm[clean_ann]) + 2 * s_sm)
    retained = float("nan")
    if neb.sum() >= 20:
        before = np.nanmedian(o[neb]) - np.nanmedian(o[clean_ann])
        after = np.nanmedian(m[neb]) - np.nanmedian(m[clean_ann])
        retained = after / before if before > 0 else float("nan")
    s = _robust_sigma(of - ndimage.median_filter(of, size=15))
    mf = np.where(np.isfinite(m), m, np.nanmedian(m))
    local_after = ndimage.median_filter(mf, size=31)
    local_before = ndimage.median_filter(of, size=31)
    gone = total = 0
    for (x, y), r in zip(star_xy, star_r_px):
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < n - 2 and 2 <= yi < n - 2):
            continue
        if of[yi - 1:yi + 2, xi - 1:xi + 2].max() - local_before[yi, xi] < 5 * s:
            continue  # star not visible in this survey to begin with
        total += 1
        gone += (mf[yi - 1:yi + 2, xi - 1:xi + 2].max() - local_after[yi, xi]) < 5 * s
    removed = gone / total if total else float("nan")
    return retained, removed, int(neb.sum()), total


def inject_test_nebula(shape, fov_arcmin, sky_sigma, size_arcmin=None):
    """Synthetic nebula for injection-recovery tests: a faint broad shell (2 sigma per pixel)
    and a bright narrow arc (20 sigma per pixel), both centred on the field."""
    n = shape[0]
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    ang = np.arctan2(yy - (n - 1) / 2, xx - (n - 1) / 2)
    r0 = 0.30 * n
    faint = 2.0 * sky_sigma * np.exp(-((rr - r0) ** 2) / (2 * (0.035 * n) ** 2))
    bright = 20.0 * sky_sigma * np.exp(-((rr - 0.18 * n) ** 2) / (2 * (0.012 * n) ** 2)) * (np.cos(ang - 0.8) > 0.3)
    return faint, bright


def recovery(mute_fn, img, injected):
    """Share of an injected nebula that survives muting: muted(img+inj) - muted(img), over inj."""
    a = mute_fn(img + injected)
    b = mute_fn(img)
    sel = injected > 0.25 * injected.max()
    return float(np.nansum((a - b)[sel]) / np.nansum(injected[sel]))


FETCH_BUDGET_S = 90


def fetch_panel_parallel(coord, fov, registry, pixels=400, fetcher=None, workers=8, progress=None):
    """Same result as fetch_panel, but downloads up to `workers` surveys at once.
    progress(done, total, label) is called as each survey arrives."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total = len(registry)
    panel, done = {}, 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(fetch_panel, coord, fov, [s], pixels, fetcher): s for s in registry}
    try:
        for fut in as_completed(futures, timeout=FETCH_BUDGET_S):
            s = futures[fut]
            try:
                panel.update(fut.result())
            except Exception as exc:  # never lose the whole panel for one survey
                panel[s["key"]] = dict(s, fov=fov, pixels=pixels, data=None, header=None, coverage=0.0,
                                       status="error: " + " ".join(str(exc).split())[:60])
            done += 1
            if progress:
                progress(done, total, s["label"])
    except Exception:   # time limit: slow surveys are marked, the screen carries on (Session 16)
        pass
    pool.shutdown(wait=False, cancel_futures=True)
    for s in registry:
        panel.setdefault(s["key"], dict(s, fov=fov, pixels=pixels, data=None, header=None, coverage=0.0,
                                        status=f"error: no answer within {FETCH_BUDGET_S} s"))
    return {s["key"]: panel[s["key"]] for s in registry}  # keep the registry order


def covering_ids(coord, radius_deg, lister=None):
    """One quick request: which HiPS surveys have data at this position (CDS MocServer footprints).
    Returns a set of IDs, or None if the service could not answer (then nothing is greyed out)."""
    if lister:
        return lister(coord, radius_deg)
    import requests
    try:
        r = requests.get(MOCSERVER, params={"RA": coord.ra.deg, "DEC": coord.dec.deg, "SR": radius_deg,
                                            "get": "id", "fmt": "json"}, timeout=30)
        r.raise_for_status()
        ids = r.json()
        return {i for i in ids if isinstance(i, str)} or None
    except Exception:
        return None
