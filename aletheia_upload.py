"""
aletheia_upload: your own images (Session 17)

read_image(path)      FITS (2D or colour cube), JPG, PNG, TIFF -> grey image in FITS orientation (row 0 = bottom),
                      colour version if any, FITS header if any.
plate_solve(path)     uses the WCS already in the file if there is one; otherwise sends a reduced copy (at most
                      2000 px, private, never public) to astrometry.net with your key. Then checks the answer
                      against Gaia DR3: stars found in your image are matched to Gaia and the typical offset is
                      reported in arcsec.
viewer_html(sol)      your image with live RA/Dec under the mouse; a click picks that position.

The display grid is the reduced image. Display pixel (xs, ys) maps to the solution's own pixel grid through
the factor k (k = reduction factor when the WCS came from the file, k = 1 when astrometry.net solved the
reduced copy), so the file's own WCS, including any distortion terms, is used exactly.
"""
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

IMAGE_TYPES = (".fits", ".fit", ".fts", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
MAX_SIDE = 2000          # longest side of the reduced copy
NOVA = "https://nova.astrometry.net"
SOLVE_LIMIT_S = 480      # astrometry.net: give up after 8 minutes
GRID_STEP = 16           # hover grid spacing in display pixels


# ----------------------------------------------------------------------------- reading
def _to_uint8(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a, [1, 99.8])
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def read_image(path):
    """Returns (grey float32 2D, rgb uint8 HxWx3 or None, header or None); rows run bottom to top (FITS)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".fits", ".fit", ".fts"):
        with fits.open(p, memmap=False) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim in (2, 3)), None)
            if hdu is None:
                raise ValueError("this FITS file has no image")
            d, header = np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
        if d.ndim == 2:
            return d, None, header
        cube = np.moveaxis(d, 0, -1) if d.shape[0] == 3 else d
        if cube.shape[-1] != 3:
            raise ValueError(f"unexpected FITS cube shape {d.shape}")
        return cube.mean(axis=2), _to_uint8(cube), header
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = np.asarray(tifffile.imread(p))
    elif ext in (".jpg", ".jpeg", ".png"):
        from PIL import Image
        im = Image.open(p)
        arr = np.asarray(im if im.mode in ("L", "I;16", "I") else im.convert("RGB"))
    else:
        raise ValueError(f"file type {ext} not supported (use {', '.join(IMAGE_TYPES)})")
    arr = np.flipud(arr)   # picture files store the top row first; FITS stores the bottom row first
    if arr.ndim == 3:
        arr = arr[..., :3]
        rgb = arr if arr.dtype == np.uint8 else _to_uint8(arr)
        return arr.astype(np.float32).mean(axis=2), rgb, None
    return arr.astype(np.float32), None, None


def reduce(a, f):
    """Block average by an integer factor (2D or HxWx3)."""
    if f <= 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].astype(np.float32)
    shp = (h // f, f, w // f, f) + b.shape[2:]
    return b.reshape(shp).mean(axis=(1, 3))


def header_wcs(header):
    """A usable celestial WCS from the file, or None."""
    if header is None:
        return None
    try:
        w = WCS(header, naxis=2)
    except Exception:
        return None
    if not w.has_celestial or not str(w.wcs.ctype[0]).startswith("RA"):
        return None
    try:
        s = w.proj_plane_pixel_scales()
        if not all(np.isfinite([float(x.value if hasattr(x, "value") else x) for x in s])):
            return None
    except Exception:
        return None
    return w


def hints_from_header(header, f):
    """Position and scale hints for astrometry.net from common capture-software keywords (speeds solving)."""
    if header is None:
        return {}
    out = {}
    ra, dec = header.get("RA"), header.get("DEC")
    if ra is None and header.get("OBJCTRA"):
        try:
            c = SkyCoord(header["OBJCTRA"], header["OBJCTDEC"], unit=(u.hourangle, u.deg))
            ra, dec = c.ra.deg, c.dec.deg
        except Exception:
            ra = None
    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        out.update(center_ra=float(ra), center_dec=float(dec), radius=3.0)
    fl, px = header.get("FOCALLEN"), header.get("XPIXSZ")
    if isinstance(fl, (int, float)) and isinstance(px, (int, float)) and fl > 0:
        scale = 206.265 * px / fl * f   # arcsec per pixel of the reduced copy (XPIXSZ includes binning)
        out.update(scale_units="arcsecperpix", scale_lower=0.8 * scale, scale_upper=1.2 * scale)
    return out


# ----------------------------------------------------------------------------- astrometry.net
def solve_nova(fits_bytes, key, hint=None, progress=print, limit_s=SOLVE_LIMIT_S, http=None):
    """Private upload to astrometry.net; returns (header with the solution, job id)."""
    import requests
    http = http or requests
    r = http.post(f"{NOVA}/api/login", data={"request-json": json.dumps({"apikey": key})}, timeout=60).json()
    if r.get("status") != "success":
        raise RuntimeError(f"astrometry.net login failed: {r.get('errormessage', r)}")
    req = dict(session=r["session"], publicly_visible="n", allow_modifications="n", allow_commercial_use="n",
               tweak_order=2)
    req.update(hint or {})
    up = http.post(f"{NOVA}/api/upload", data={"request-json": json.dumps(req)},
                   files={"file": ("aletheia_upload.fits", fits_bytes, "application/octet-stream")},
                   timeout=300).json()
    if up.get("status") != "success":
        raise RuntimeError(f"astrometry.net upload failed: {up.get('errormessage', up)}")
    sub, t0, job, status = up["subid"], time.time(), None, None
    while time.time() - t0 < limit_s:
        time.sleep(3)
        jobs = [j for j in http.get(f"{NOVA}/api/submissions/{sub}", timeout=60).json().get("jobs", []) if j]
        if not jobs:
            progress(f"waiting in the astrometry.net queue ({time.time() - t0:.0f} s)")
            continue
        job = jobs[0]
        status = http.get(f"{NOVA}/api/jobs/{job}", timeout=60).json().get("status")
        progress(f"astrometry.net is solving ({status}, {time.time() - t0:.0f} s)")
        if status in ("success", "failure"):
            break
    if status != "success":
        raise RuntimeError("astrometry.net could not solve this image" if status == "failure" else
                           f"astrometry.net did not finish within {limit_s // 60} minutes")
    raw = http.get(f"{NOVA}/wcs_file/{job}", timeout=60).content
    with fits.open(io.BytesIO(raw)) as h:
        return h[0].header.copy(), job


def _fits_bytes(a):
    """16-bit copy (half the size of 32-bit floats, same stars): a quicker upload (Session 20)."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=float(np.nanmedian(a)))
    lo, hi = np.percentile(a, [0.5, 99.99])
    q = (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 65535).astype(np.uint16)
    buf = io.BytesIO()
    fits.PrimaryHDU(q).writeto(buf)   # astropy stores uint16 as BITPIX 16 with BZERO 32768
    return buf.getvalue()


# ----------------------------------------------------------------------------- checking
def find_stars(d, nmax=300):
    """Bright, isolated peaks with simple centroids (display pixels)."""
    from scipy import ndimage
    d = np.nan_to_num(np.asarray(d, dtype=np.float32), nan=float(np.nanmedian(d)))
    b = 32
    hh, ww = max(1, d.shape[0] // b), max(1, d.shape[1] // b)
    coarse = np.median(d[:hh * b, :ww * b].reshape(hh, b, ww, b).transpose(0, 2, 1, 3).reshape(hh, ww, -1), axis=2) \
        if d.shape[0] >= b and d.shape[1] >= b else np.full((1, 1), np.median(d))
    bkg = ndimage.zoom(coarse, (d.shape[0] / coarse.shape[0], d.shape[1] / coarse.shape[1]), order=1)
    r = ndimage.gaussian_filter(d - bkg, 1.0)
    sig = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
    peaks = (r == ndimage.maximum_filter(r, 7)) & (r > 5 * sig)
    peaks[:4, :] = peaks[-4:, :] = peaks[:, :4] = peaks[:, -4:] = False
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-r[ys, xs])[:nmax]
    out = []
    for y, x in zip(ys[order], xs[order]):
        w = np.clip(r[y - 2:y + 3, x - 2:x + 3], 0, None)
        if w.sum() <= 0:
            continue
        gy, gx = np.mgrid[-2:3, -2:3]
        out.append((x + (gx * w).sum() / w.sum(), y + (gy * w).sum() / w.sum()))
    return np.array(out).reshape(-1, 2)


def _gaia(centre, radius_deg, glim):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"], column_filters={"Gmag": f"<{glim}"}, row_limit=8000)
    v.TIMEOUT = 120
    t = v.query_region(centre, radius=radius_deg * u.deg, catalog="I/355/gaiadr3")[0]
    return SkyCoord(np.array(t["RA_ICRS"], float) * u.deg, np.array(t["DE_ICRS"], float) * u.deg)


def gaia_check(disp, to_world, centre, radius_deg, scale_arcsec, gaia=None):
    """Median offset (arcsec) between your stars and Gaia DR3, with the number of matches."""
    stars = find_stars(disp)
    if len(stars) < 5:
        return dict(n=0, note="too few stars found in the image to check")
    ra, dec = to_world(stars[:, 0], stars[:, 1])
    mine = SkyCoord(ra * u.deg, dec * u.deg)
    glim = 13 if radius_deg > 1.5 else (15 if radius_deg > 0.5 else 17)
    ref = gaia(centre, radius_deg, glim) if gaia else _gaia(centre, min(radius_deg, 5.0), glim)
    idx, sep, _ = mine.match_to_catalog_sky(ref)
    tol = max(5.0, 3 * scale_arcsec)
    good = sep.arcsec[sep.arcsec <= tol]
    if len(good) < 5:
        return dict(n=int(len(good)), note="fewer than 5 stars matched Gaia: the solution may be wrong")
    return dict(n=int(len(good)), of=int(len(stars)), median=float(np.median(good)),
                p90=float(np.percentile(good, 90)), limit=float(max(2.0, 0.5 * scale_arcsec)))


# ----------------------------------------------------------------------------- the whole step
# ----------------------------------------------------------------------------- quick solve (Session 20)
QUICK_LIMIT_S = 40       # the quick solve gives up after this and hands over to astrometry.net
# Web edition: a new visitor's pixel scale is unknown, so a ladder of scales (arcsec per pixel of the reduced copy)
# is tried, each with a wide (30 per cent) tolerance; together they cover about 0.5 to 35 arcsec per pixel.
SCALE_LADDER = (2.6, 4.1, 1.6, 6.6, 1.0, 10.5, 17.0, 0.62, 27.0)
LADDER_STEP_S = 15
LADDER_LIMIT_S = 120


def _gaia_bright(centre, radius_deg, n=6000):
    """Brightest Gaia DR3 stars around a position: (ra, dec, G) arrays, brightest first."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "+Gmag"], column_filters={"Gmag": "<19"}, row_limit=n)
    v.TIMEOUT = 60
    t = v.query_region(centre, radius=radius_deg * u.deg, catalog="I/355/gaiadr3")[0]
    return np.array(t["RA_ICRS"], float), np.array(t["DE_ICRS"], float), np.array(t["Gmag"], float)


def _tangent(ra, dec, c):
    """Gnomonic (TAN) projection around c, in arcsec."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(c.ra.deg), np.radians(c.dec.deg)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi) * 3600, np.degrees(eta) * 3600


def _triangles(p, k=5):
    """Triangles of each star with pairs of its k nearest neighbours: vertex indices ordered by the length of the
    opposite side (longest first), the two side ratios (unchanged by rotation, scale and mirroring) and the
    longest side."""
    from itertools import combinations
    from scipy.spatial import cKDTree
    k = min(k, len(p) - 1)
    _, nn = cKDTree(p).query(p, k + 1)
    tris = np.array(sorted({tuple(sorted((i, a, b))) for i in range(len(p)) for a, b in combinations(nn[i, 1:], 2)}))
    P = p[tris]
    opp = np.stack([np.hypot(*(P[:, 2] - P[:, 1]).T), np.hypot(*(P[:, 0] - P[:, 2]).T),
                    np.hypot(*(P[:, 1] - P[:, 0]).T)], axis=1)        # side opposite each vertex
    order = np.argsort(-opp, axis=1)
    tris = np.take_along_axis(tris, order, axis=1)
    L = np.take_along_axis(opp, order, axis=1)
    ok = L[:, 2] > 1e-6
    return tris[ok], np.c_[L[ok, 1] / L[ok, 0], L[ok, 2] / L[ok, 0]], L[ok, 0]


def _match_patterns(img, cat, s0, tol_scale=0.12, tol_inv=0.015, max_tries=4000):
    """Image star positions (px) against catalogue positions (arcsec, tangent plane). Every pair of similar
    triangles proposes a transform (rotation, mirror, scale near s0, shift); the one that lines up the most stars
    wins. Returns (A 2x2, t, inlier image indices, matched catalogue indices) or None."""
    from scipy.spatial import cKDTree
    ti, ii, li = _triangles(img)
    tc, ic, lc = _triangles(cat)
    pairs = cKDTree(ic).query_ball_point(ii, tol_inv)
    cand = [(a, b) for a, bs in enumerate(pairs) for b in bs if abs(lc[b] / (li[a] * s0) - 1) < tol_scale]
    if not cand:
        return None
    rng = np.random.default_rng(0)
    if len(cand) > max_tries:
        cand = [cand[i] for i in rng.choice(len(cand), max_tries, replace=False)]
    tree = cKDTree(cat)
    tol = max(3.0, 2.5 * s0)
    best = (0, None)
    for a, b in cand:
        X, Y = img[ti[a]], cat[tc[b]]
        M = np.c_[X, np.ones(3)]
        try:
            sol = np.linalg.solve(M, Y)            # affine from three points
        except np.linalg.LinAlgError:
            continue
        A, t = sol[:2].T, sol[2]
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[1] <= 0 or sv[0] / sv[1] > 1.08 or abs(np.sqrt(sv[0] * sv[1]) / s0 - 1) > tol_scale:
            continue                                # not a similarity transform at about the right scale
        d, j = tree.query(img @ A.T + t)
        n = int((d < tol).sum())
        if n > best[0]:
            best = (n, (A, t))
    if best[1] is None or best[0] < 8:
        return None
    A, t = best[1]
    d, j = tree.query(img @ A.T + t)
    keep = d < tol
    return A, t, np.nonzero(keep)[0], j[keep]


def quick_solve(disp, centre, scale, wide=False, gaia=None, progress=print, limit_s=QUICK_LIMIT_S, tol_scale=0.12):
    """Seconds-fast solve when a rough position and the pixel scale (within about 10 per cent) are known: the
    brightest stars of your image are matched to Gaia DR3 by their triangle patterns, whatever the rotation and
    even if the image is mirrored; a TAN solution is then fitted to every matched star and checked.
    wide=True when the rough position is a target somewhere in the image rather than its centre.
    Returns (WCS for the display copy, number of matched stars)."""
    import time
    from astropy.wcs.utils import fit_wcs_from_points
    t0 = time.time()
    H, W = disp.shape
    xy = find_stars(disp, nmax=400)
    if len(xy) < 15:
        raise RuntimeError(f"only {len(xy)} stars found in the image")
    half = 0.5 * np.hypot(W, H) * scale / 3600.0            # image half-diagonal, deg
    ra, dec, g = (gaia or _gaia_bright)(centre, min((2.1 if wide else 1.3) * half, 3.0))
    if len(ra) < 15:
        raise RuntimeError("too few Gaia stars around that position")
    allsky = SkyCoord(ra * u.deg, dec * u.deg)
    gx, gy = _tangent(ra, dec, centre)
    plane = np.c_[gx, gy]
    for n_img, n_cat in ((80, 300 if wide else 160), (150, 600 if wide else 320)):
        if time.time() - t0 > limit_s:
            break
        m = _match_patterns(xy[:n_img], plane[:n_cat], scale, tol_scale=tol_scale)
        if m is None:
            continue
        A, t, ii, jj = m
        w = fit_wcs_from_points((xy[ii, 0], xy[ii, 1]), allsky[jj], proj_point="center", projection="TAN")
        for _ in range(2):   # refine with every star in the image that has a Gaia star where the solution puts it
            wra, wdec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
            i, sep, _ = SkyCoord(wra * u.deg, wdec * u.deg).match_to_catalog_sky(allsky)
            ok = sep.arcsec <= max(2.5, 1.5 * scale)
            if ok.sum() < 10:
                break
            w = fit_wcs_from_points((xy[ok, 0], xy[ok, 1]), allsky[i[ok]], proj_point="center", projection="TAN")
        if ok.sum() >= 10 and np.median(sep.arcsec[ok]) <= max(2.0, 0.6 * scale):
            progress(f"quick solve: {int(ok.sum())} stars matched to Gaia in {time.time() - t0:.0f} s")
            return w, int(ok.sum())
    raise RuntimeError(f"no match found in {time.time() - t0:.0f} s")


def remembered_scale(scales_file, shape):
    """Arcsec per pixel of an image of this size (same camera and telescope) from an earlier solve, or None."""
    try:
        return float(json.loads(Path(scales_file).read_text())[f"{shape[1]}x{shape[0]}"])
    except Exception:
        return None


def remember_scale(scales_file, shape, scale):
    try:
        f = Path(scales_file)
        data = json.loads(f.read_text()) if f.exists() else {}
        data[f"{shape[1]}x{shape[0]}"] = round(float(scale), 5)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, indent=1))
    except Exception:
        pass


def plate_solve(path, key=None, progress=print, http=None, gaia=None, cache_dir=None, hint=None, scales_file=None,
                quick_gaia=None):
    """hint: rough position (SkyCoord) of anything in the image, optional; scales_file: remembered pixel scales."""
    t_start = time.time()
    progress(f"reading {Path(path).name}...")
    grey, rgb, header = read_image(path)
    f = max(1, int(np.ceil(max(grey.shape) / MAX_SIDE)))
    disp, disp_rgb = reduce(grey, f), (reduce(rgb, f).astype(np.uint8) if rgb is not None else None)
    w = header_wcs(header)
    saved = Path(cache_dir) / f"{Path(path).stem}_solved.fits" if cache_dir else None
    if w is not None:
        k, method, job = f, "solution already in the file", None
    elif saved is not None and saved.exists() and saved.stat().st_mtime >= Path(path).stat().st_mtime:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(fits.getheader(saved), naxis=2)
        k, method, job = 1, "saved solution", None
    else:
        hh = hints_from_header(header, f)
        scale_guess = 0.5 * (hh["scale_lower"] + hh["scale_upper"]) if "scale_lower" in hh else None
        if scale_guess is None and scales_file:
            mem = remembered_scale(scales_file, grey.shape)
            scale_guess = mem * f if mem else None
        centre_guess, wide = (hint, True) if hint is not None else (
            (SkyCoord(hh["center_ra"] * u.deg, hh["center_dec"] * u.deg), False) if "center_ra" in hh else (None, False))
        w = job = None
        quick_note = ""
        if centre_guess is not None:
            if scale_guess:
                tries = [(scale_guess, 0.12, QUICK_LIMIT_S)]
            else:                       # pixel scale unknown (a new web visitor): try the ladder
                tries = [(s, 0.3, LADDER_STEP_S) for s in SCALE_LADDER]
            t_q = time.time()
            for n, (s_try, tol, lim) in enumerate(tries, 1):
                if len(tries) > 1:
                    if time.time() - t_q > LADDER_LIMIT_S:
                        break
                    progress(f"quick solve: pixel scale unknown, trying about {s_try * 1.0 / f:.1f}\"/px "
                             f"({n} of {len(tries)})...")
                else:
                    progress("quick solve: matching your stars to Gaia DR3...")
                try:
                    w, _ = quick_solve(disp, centre_guess, s_try, wide, quick_gaia, lambda m: None, lim, tol)
                    k, method = 1, "quick solve (Gaia DR3)"
                    break
                except Exception as exc:
                    quick_note = " ".join(str(exc).split())[:60]
                    if "Gaia" in quick_note and "few" not in quick_note:
                        break
            if w is None and key:
                progress(f"quick solve did not work ({quick_note}); asking astrometry.net")
        if w is None:
            if not key:
                if centre_guess is None:
                    raise RuntimeError("type a rough position in the box (an object name, or the RA Dec of anything in "
                                       "the picture) and press Plate solve again")
                raise RuntimeError(f"the stars could not be matched to Gaia near that position ({quick_note}). Check the "
                                   "rough position is inside the picture; a less stretched image or a FITS file helps")
            if scale_guess and "scale_lower" not in hh:
                hh.update(scale_units="arcsecperpix", scale_lower=0.85 * scale_guess, scale_upper=1.15 * scale_guess)
            if hint is not None:
                half = 0.5 * np.hypot(*disp.shape) * (scale_guess or 3.0) / 3600.0
                hh.update(center_ra=hint.ra.deg, center_dec=hint.dec.deg, radius=float(min(10.0, max(1.0, 2.5 * half))))
            progress("sending a reduced, private copy to astrometry.net" + (" with hints" if hh else "") + "...")
            hdr, job = solve_nova(_fits_bytes(disp), key, hh, progress, http=http)
            w, k, method = WCS(hdr, naxis=2), 1, "astrometry.net"

    def to_world(xs, ys):
        return w.all_pix2world(np.asarray(xs, float) * k + (k - 1) / 2.0,
                               np.asarray(ys, float) * k + (k - 1) / 2.0, 0)
    def to_pixel(ra, dec):   # sky -> display pixel (FITS orientation, row 0 = bottom)
        ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
        try:
            sx, sy = w.all_world2pix(ra, dec, 0, quiet=True)
        except Exception:
            sx, sy = w.wcs_world2pix(ra, dec, 0)
        return (sx - (k - 1) / 2.0) / k, (sy - (k - 1) / 2.0) / k
    H, W_ = disp.shape
    cx, cy = (W_ - 1) / 2.0, (H - 1) / 2.0
    c0, cn = to_world([cx, cx], [cy, cy + 10])
    centre = SkyCoord(c0[0] * u.deg, cn[0] * u.deg)
    up = SkyCoord(c0[1] * u.deg, cn[1] * u.deg)
    scale = centre.separation(up).arcsec / 10.0
    pa_up = centre.position_angle(up).deg % 360          # direction of the image's "up", east of north
    cor = to_world([0, W_ - 1], [0, H - 1])
    radius = max(SkyCoord(cor[0] * u.deg, cor[1] * u.deg).separation(centre).deg)
    progress("checking the solution against Gaia DR3...")
    try:
        check = gaia_check(disp, to_world, centre, radius, scale, gaia)
    except Exception as exc:
        check = dict(n=0, note=f"Gaia check could not run: {' '.join(str(exc).split())[:60]}")
    if scales_file and check.get("median") is not None and check["median"] <= check["limit"]:
        remember_scale(scales_file, grey.shape, scale / f)   # next image from this camera solves quickly
    return dict(path=str(path), method=method, job=job, factor=f, k=k, wcs=w, to_world=to_world, to_pixel=to_pixel,
                seconds=round(time.time() - t_start, 1),
                disp=disp,
                rgb=disp_rgb, centre=centre, scale=scale, pa_up=pa_up, radius=radius,
                width_arcmin=W_ * scale / 60, height_arcmin=H * scale / 60, check=check)


def summary(sol):
    c, ch = sol["centre"], sol["check"]
    s = (f"{Path(sol['path']).name}: {sol['method']}" + (f" in {sol['seconds']:.0f} s" if sol.get("seconds") else "")
         + f"; centre RA {c.ra.to_string(u.hour, sep=':', precision=1)} "
         f"Dec {c.dec.to_string(sep=':', precision=0, alwayssign=True)}; {sol['scale']:.2f}\"/px on the "
         f"reduced copy (x{sol['factor']}); field {sol['width_arcmin']:.1f}' x {sol['height_arcmin']:.1f}'; "
         f"image up points {sol['pa_up']:.1f} deg east of north")
    if ch.get("median") is not None:
        ok = ch["median"] <= ch["limit"]
        s += (f"; Gaia check: {ch['n']} stars matched, typical offset {ch['median']:.2f}\" "
              f"({'within' if ok else 'ABOVE'} the {ch['limit']:.1f}\" limit)")
    else:
        s += f"; Gaia check: {ch.get('note', 'not done')}"
    return s


def save_solution(sol, folder):
    """Text summary, plus the astrometry.net solution (valid for the reduced copy) so it is never solved twice.
    Solutions that came with the file are not copied: the original file already holds them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(sol["path"]).stem
    if sol["method"] in ("astrometry.net", "quick solve (Gaia DR3)"):   # solved here: keep it
        hdr = sol["wcs"].to_header(relax=True)
        hdr["ALETHF"] = (sol["factor"], "solution is for the copy reduced by this factor")
        fits.PrimaryHDU(header=hdr).writeto(folder / f"{stem}_solved.fits", overwrite=True)
    (folder / f"{stem}_solution.txt").write_text(summary(sol) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- hover viewer
def hover_grid(sol, step=GRID_STEP):
    """RA/Dec on a coarse grid in screen order (row 0 = top), RA unwrapped around the centre."""
    H, W_ = sol["disp"].shape
    xs = np.unique(np.r_[np.arange(0, W_, step), W_ - 1]).astype(float)
    ys_screen = np.unique(np.r_[np.arange(0, H, step), H - 1]).astype(float)
    X, Y = np.meshgrid(xs, ys_screen)
    ra, dec = sol["to_world"](X, (H - 1) - Y)
    ra0 = sol["centre"].ra.deg
    ra = ((ra - ra0 + 180.0) % 360.0) - 180.0 + ra0
    return xs, ys_screen, np.round(ra, 7), np.round(dec, 7)


def _jpeg(sol):
    """The display copy as a JPEG (screen order, top row first); JPEG keeps the notebook small."""
    from PIL import Image
    d = sol["disp"]
    lo, hi = np.nanpercentile(d, 5), np.nanpercentile(d, 99.7)
    img = sol["rgb"] if sol["rgb"] is not None else \
        (np.arcsinh(5 * np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)) / np.arcsinh(5) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.flipud(img))).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def viewer_html(sol, uid, width=760, callback="aletheia.pick", known=None, circle_callback="aletheia.circle"):
    """Your image with a crosshair: wheel to zoom, drag to pan, live RA/Dec, a magnifier, click to pin a point
    (optionally snapped to the nearest star's centre); the last pin is sent to the position box."""
    xs, ys, ra, dec = hover_grid(sol)
    H, W_ = sol["disp"].shape
    st = find_stars(sol["disp"], nmax=4000)       # precise centres from the real data, for snapping
    stars = [[round(float(x), 2), round(float((H - 1) - y), 2)] for x, y in st]
    cfg = dict(W=W_, H=H, xs=xs.tolist(), ys=ys.tolist(), ra=ra.tolist(), dec=dec.tolist(), cb=callback,
               scale=float(sol["scale"]), stars=stars, known=known or [], cb2=circle_callback,
               rmax=round(max(0.5, min(30.0, min(W_, H) * float(sol["scale"]) / 120.0)), 2))
    cw = int(min(width, round(width * W_ / H)))     # tall images: keep the canvas within width x width
    ch = int(round(cw * H / W_))
    return """
<div id="V{uid}" style="font-family:monospace;color:#F5F5F7;display:flex;gap:12px;align-items:flex-start">
 <div>
  <div style="margin-bottom:4px;font-size:11px;color:#8E8E93">
   <button id="F{uid}">Fit</button>
   <label><input id="S{uid}" type="checkbox" checked> snap to nearest star</label>
   <button id="X{uid}">Clear pins</button>
   <label><input id="O{uid}" type="checkbox" checked> catalogued objects</label>
   &nbsp;wheel = zoom · drag = pan · click = pin
  </div>
  <div style="margin-bottom:6px;font-size:12px;color:#8E8E93;display:flex;align-items:center;gap:8px">
   <button id="Q{uid}" title="Circle tool: click to place, or press and drag to draw it to the nebula's size">Circle</button>
   <input id="Z{uid}" type="range" min="0.05" max="{rmax}" step="0.05" value="1" style="width:150px;accent-color:#FF453A">
   <span id="Y{uid}" style="min-width:90px">radius 1.00'</span>
   <button id="A{uid}" title="Analyse the area inside the circle and use it in the report">Analyse circle</button>
  </div>
  <canvas id="M{uid}" width="{w}" height="{h}" style="background:#000;cursor:none;display:block;border-radius:12px"></canvas>
  <div id="C{uid}" style="color:#64D2FF;margin-top:4px">RA --  Dec --</div>
  <div id="G{uid}" style="font-size:11px;margin-top:2px"></div>
 </div>
 <div style="width:240px">
  <canvas id="L{uid}" width="200" height="200" style="background:#000;border:1px solid #333"></canvas>
  <div style="color:#8E8E93;font-size:11px">magnifier (x8 of the screen)</div>
  <div id="P{uid}" style="font-size:12px;margin-top:8px;white-space:pre-line"></div>
  <div id="K{uid}" style="color:#FF9F0A;font-size:12px;margin-top:6px"></div>
 </div>
</div>
<script>
(function() {{
 const aletheiaCall = (name, args) => {{   // Colab, or the web edition's bridge (Session 22)
   if (window.google && google.colab && google.colab.kernel) return google.colab.kernel.invokeFunction(name, args, {{}});
   if (window.aletheiaBridge) return window.aletheiaBridge(name, args);
   throw new Error("not connected to Aletheia"); }};
 const g = {cfg}, $ = (k) => document.getElementById(k + "{uid}");
 const cv = $("M"), cx = cv.getContext("2d"), lp = $("L"), lx = lp.getContext("2d");
 const img = new Image();
 let s = 1, ox = 0, oy = 0, mouse = null, down = null, pins = [];
 let circleMode = false, circ = null, rArc = 1.0;      // circle tool (Session 21): centre in image pixels, radius in arcmin
 function setR(v) {{ rArc = Math.min(Math.max(v, 0.05), g.rmax); $("Z").value = rArc; $("Y").textContent = "radius " + rArc.toFixed(2) + "'"; }}
 function idx(a, v) {{ let i = 0; while (i < a.length - 2 && a[i + 1] < v) i++; return i; }}
 function sky(px, py) {{
   const i = idx(g.xs, px), j = idx(g.ys, py);
   const tx = Math.min(Math.max((px - g.xs[i]) / (g.xs[i + 1] - g.xs[i]), 0), 1);
   const ty = Math.min(Math.max((py - g.ys[j]) / (g.ys[j + 1] - g.ys[j]), 0), 1);
   const b = (A) => (1 - ty) * ((1 - tx) * A[j][i] + tx * A[j][i + 1]) + ty * ((1 - tx) * A[j + 1][i] + tx * A[j + 1][i + 1]);
   return [((b(g.ra) % 360) + 360) % 360, b(g.dec)];
 }}
 function hms(d) {{ let h = d / 15, H = Math.floor(h), m = (h - H) * 60, M = Math.floor(m), S = (m - M) * 60;
   if (S >= 59.995) {{ S = 0; M += 1; }} if (M >= 60) {{ M = 0; H = (H + 1) % 24; }}
   return String(H).padStart(2, "0") + ":" + String(M).padStart(2, "0") + ":" + S.toFixed(2).padStart(5, "0"); }}
 function dms(d) {{ const sg = d < 0 ? "-" : "+"; d = Math.abs(d); let D = Math.floor(d), m = (d - D) * 60, M = Math.floor(m), S = (m - M) * 60;
   if (S >= 59.95) {{ S = 0; M += 1; }} if (M >= 60) {{ M = 0; D += 1; }}
   return sg + String(D).padStart(2, "0") + ":" + String(M).padStart(2, "0") + ":" + S.toFixed(1).padStart(4, "0"); }}
 // canvas position <-> image pixel (pixel centres at integers, y downwards as on screen)
 const toImg = (x, y) => [ox + x / s - 0.5, oy + y / s - 0.5];
 const toCv = (px, py) => [(px + 0.5 - ox) * s, (py + 0.5 - oy) * s];
 function fit() {{ s = Math.min(cv.width / g.W, cv.height / g.H); ox = (g.W - cv.width / s) / 2; oy = (g.H - cv.height / s) / 2; draw(); }}
 function snap(px, py) {{   // nearest star found by Aletheia (centres measured on the real data), within 6 pixels
   let best = null, bd = 36;
   g.stars.forEach((t) => {{ const d = (t[0] - px) ** 2 + (t[1] - py) ** 2; if (d < bd) {{ bd = d; best = t; }} }});
   return best ? best : [px, py];
 }}
 function draw() {{
   cx.fillStyle = "#000"; cx.fillRect(0, 0, cv.width, cv.height);
   if (img.complete && img.naturalWidth) {{ cx.imageSmoothingEnabled = s < 2;
     cx.drawImage(img, ox, oy, cv.width / s, cv.height / s, 0, 0, cv.width, cv.height); }}
   cx.strokeStyle = "#FF9F0A"; cx.fillStyle = "#FF9F0A"; cx.font = "12px Inter, -apple-system, 'Segoe UI', sans-serif";
   if ($("O").checked) {{ cx.font = "11px Inter, -apple-system, 'Segoe UI', sans-serif"; const boxes = [];
     g.known.forEach((o) => {{ const c = toCv(o.x, o.y); if (c[0] < -20 || c[1] < -20 || c[0] > cv.width + 20 || c[1] > cv.height + 20) return;
       cx.strokeStyle = cx.fillStyle = colour(o.kind); cx.beginPath();
       if (o.kind === "your catalogue") cx.rect(c[0] - 8, c[1] - 8, 16, 16); else cx.arc(c[0], c[1], 9, 0, 2 * Math.PI);
       cx.stroke();
       const w = cx.measureText(o.name).width; let y = c[1] + 4;   // move a label down until it overlaps no other
       while (boxes.some((b) => c[0] + 11 < b[0] + b[2] && b[0] < c[0] + 11 + w && Math.abs(b[1] - y) < 13)) y += 13;
       boxes.push([c[0] + 11, y, w]);
       if (y !== c[1] + 4) {{ cx.beginPath(); cx.moveTo(c[0] + 9, c[1] + 2); cx.lineTo(c[0] + 11, y - 4); cx.stroke(); }}
       cx.fillText(o.name, c[0] + 11, y); }}); }}
   if (circ) {{ const c = toCv(circ[0], circ[1]), R = rArc * 60 / g.scale * s;
     cx.save(); cx.strokeStyle = "#FF453A"; cx.lineWidth = 1.6; cx.setLineDash([6, 4]); cx.beginPath(); cx.arc(c[0], c[1], R, 0, 2 * Math.PI); cx.stroke();
     cx.setLineDash([]); cx.beginPath(); cx.moveTo(c[0] - 5, c[1]); cx.lineTo(c[0] + 5, c[1]); cx.moveTo(c[0], c[1] - 5); cx.lineTo(c[0], c[1] + 5); cx.stroke();
     cx.fillStyle = "#FF453A"; cx.font = "12px Inter, -apple-system, 'Segoe UI', sans-serif";
     cx.fillText("r " + rArc.toFixed(2) + "'", c[0] + R * 0.72 + 4, c[1] - R * 0.72 - 4); cx.restore(); }}
   cx.strokeStyle = "#FF9F0A"; cx.fillStyle = "#FF9F0A"; cx.font = "12px Inter, -apple-system, 'Segoe UI', sans-serif";
   pins.forEach((p, n) => {{ const c = toCv(p[0], p[1]); cx.beginPath(); cx.arc(c[0], c[1], 7, 0, 2 * Math.PI); cx.stroke();
     cx.fillText(String(n + 1), c[0] + 9, c[1] - 9); }});
   if (mouse) {{
     cx.strokeStyle = "rgba(95,212,196,0.85)"; cx.lineWidth = 1; cx.beginPath();
     cx.moveTo(0, mouse[1] + 0.5); cx.lineTo(mouse[0] - 6, mouse[1] + 0.5); cx.moveTo(mouse[0] + 6, mouse[1] + 0.5); cx.lineTo(cv.width, mouse[1] + 0.5);
     cx.moveTo(mouse[0] + 0.5, 0); cx.lineTo(mouse[0] + 0.5, mouse[1] - 6); cx.moveTo(mouse[0] + 0.5, mouse[1] + 6); cx.lineTo(mouse[0] + 0.5, cv.height);
     cx.stroke();
     const q = toImg(mouse[0], mouse[1]), r = 12.5 / s;   // magnifier: 25 screen pixels -> 200
     lx.fillStyle = "#000"; lx.fillRect(0, 0, 200, 200); lx.imageSmoothingEnabled = false;
     if (img.complete) lx.drawImage(img, q[0] + 0.5 - r, q[1] + 0.5 - r, 2 * r, 2 * r, 0, 0, 200, 200);
     lx.strokeStyle = "#64D2FF"; lx.beginPath(); lx.moveTo(0, 100); lx.lineTo(90, 100); lx.moveTo(110, 100); lx.lineTo(200, 100);
     lx.moveTo(100, 0); lx.lineTo(100, 90); lx.moveTo(100, 110); lx.lineTo(100, 200); lx.stroke();
   }}
 }}
 const COLOURS = {{"PN": "#FFD60A", "possible PN": "#FFD60A", "H II region": "#FF453A", "SNR": "#BF5AF2",
                   "possible SNR": "#BF5AF2", "reflection nebula": "#0A84FF", "your catalogue": "#64D2FF",
                   "planetarynebulae.net": "#FF9F0A"}};
 const colour = (k) => COLOURS[k] || "#B0B0B0";
 function nearKnown(q) {{ if (!$("O").checked) return "";
   let best = null, bd = (10 / s) ** 2;
   g.known.forEach((o) => {{ const d = (o.x - q[0]) ** 2 + (o.y - q[1]) ** 2; if (d < bd) {{ bd = d; best = o; }} }});
   return best ? "   ·   " + best.name + " (" + best.kind + ")" : ""; }}
 function legend() {{ const n = {{}}; g.known.forEach((o) => n[o.kind] = (n[o.kind] || 0) + 1);
   const parts = Object.keys(n).map((k) => '<span style="color:' + colour(k) + '">' + k + " " + n[k] + "</span>");
   $("G").innerHTML = parts.length ? "catalogued here: " + parts.join(" · ") + ' <span style="color:#6E6E73">(anything glowing without a marker is not in these lists)</span>'
                                   : '<span style="color:#6E6E73">no catalogued nebulae in this image</span>'; }}
 function readout(q) {{ const p = sky(q[0], q[1]);
   $("C").textContent = "RA " + hms(p[0]) + "   Dec " + dms(p[1]) + "   (" + p[0].toFixed(5) + ", " + p[1].toFixed(5) + ")   x " +
     q[0].toFixed(1) + " y " + q[1].toFixed(1) + "   zoom x" + (s * g.W / cv.width).toFixed(1) + nearKnown(q); }}
 function pos(ev) {{ const r = cv.getBoundingClientRect(); return [(ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height]; }}
 cv.addEventListener("mousemove", (ev) => {{ const m = pos(ev);
   if (down && circleMode) {{ const q = toImg(m[0], m[1]); circ = down[3];
     setR(Math.hypot(q[0] - circ[0], q[1] - circ[1]) * g.scale / 60); down[2] += 10; }}
   else if (down) {{ ox -= (m[0] - down[0]) / s; oy -= (m[1] - down[1]) / s; down = [m[0], m[1], down[2] + Math.abs(m[0] - down[0]) + Math.abs(m[1] - down[1])]; }}
   mouse = m; readout(toImg(m[0], m[1])); draw(); }});
 cv.addEventListener("mouseleave", () => {{ mouse = null; down = null; draw(); }});
 cv.addEventListener("mousedown", (ev) => {{ const m = pos(ev); down = [m[0], m[1], 0, toImg(m[0], m[1])]; }});
 cv.addEventListener("mouseup", (ev) => {{ const moved = down ? down[2] : 0; down = null;
   if (circleMode) {{ if (moved <= 3) {{ const m = pos(ev); circ = toImg(m[0], m[1]); }}
     const p = sky(circ[0], circ[1]); $("K").textContent = "circle at " + hms(p[0]) + " " + dms(p[1]) + ", radius " + rArc.toFixed(2) + "': press Analyse circle"; draw(); return; }}
   if (moved > 3) return;
   const m = pos(ev); let q = toImg(m[0], m[1]); if ($("S").checked) q = snap(q[0], q[1]);
   pins.push(q); const p = sky(q[0], q[1]), txt = hms(p[0]) + " " + dms(p[1]);
   $("P").textContent = pins.map((t, n) => {{ const k = sky(t[0], t[1]); return "#" + (n + 1) + "  " + hms(k[0]) + " " + dms(k[1]); }}).join("\\n");
   try {{ aletheiaCall(g.cb, [p[0], p[1]]); $("K").textContent = "#" + pins.length + " " + txt + " is in the position box: press Analyse field"; }}
   catch (e) {{ $("K").textContent = "#" + pins.length + " " + txt + " (copy it into the position box)"; }}
   draw(); }});
 cv.addEventListener("wheel", (ev) => {{ ev.preventDefault();
   if (circleMode && circ && ev.shiftKey) {{ setR(rArc * (ev.deltaY < 0 ? 1.08 : 1 / 1.08)); draw(); return; }}
   const m = pos(ev), q = toImg(m[0], m[1]);
   const f = ev.deltaY < 0 ? 1.25 : 0.8, smin = Math.min(cv.width / g.W, cv.height / g.H);
   s = Math.min(Math.max(s * f, smin), 16); ox = q[0] + 0.5 - m[0] / s; oy = q[1] + 0.5 - m[1] / s;
   readout(toImg(m[0], m[1])); draw(); }}, {{passive: false}});
 $("F").onclick = fit; $("O").onchange = draw; legend(); $("X").onclick = () => {{ pins = []; circ = null; $("P").textContent = ""; $("K").textContent = ""; draw(); }};
 $("Q").onclick = () => {{ circleMode = !circleMode; $("Q").classList.toggle("al-circle-on", circleMode);
   $("Q").style.background = circleMode ? "#FF453A" : ""; $("Q").style.color = circleMode ? "#fff" : ""; $("Q").textContent = circleMode ? "Circle: on" : "Circle";
   $("K").textContent = circleMode ? "Circle tool: click the nebula, or press and drag outwards to its edge; shift + wheel or the slider changes the size" : ""; }};
 $("Z").oninput = () => {{ setR(parseFloat($("Z").value)); draw(); }};
 $("A").onclick = () => {{ if (!circ) {{ $("K").textContent = "Place a circle first: press Circle, then click the nebula"; return; }}
   const p = sky(circ[0], circ[1]);
   try {{ aletheiaCall(g.cb2, [p[0], p[1], rArc]);
     $("K").textContent = "Analysing the circle at " + hms(p[0]) + " " + dms(p[1]) + " (radius " + rArc.toFixed(2) + "'): results appear below"; }}
   catch (e) {{ $("K").textContent = "Circle " + hms(p[0]) + " " + dms(p[1]) + ", diameter " + (2 * rArc).toFixed(2) + "' (type these into Search a position)"; }} }};
 setR(1.0);
 img.onload = fit;
 img.src = "data:image/jpeg;base64,{jpg}";
}})();
</script>""".format(uid=uid, w=cw, h=ch, jpg=_jpeg(sol), cfg=json.dumps(cfg), rmax=cfg["rmax"])


# ----------------------------------------------------------------------------- what is already known in the image
FIELD_TYPES = {"PN": "PN", "PN?": "possible PN", "HII": "H II region", "SNR": "SNR", "SNR?": "possible SNR",
               "RNe": "reflection nebula", "EmO": "emission object", "bub": "bubble"}


def _simbad_field(centre, radius_deg):
    from astroquery.simbad import Simbad
    types = ", ".join(f"'{t}'" for t in FIELD_TYPES)
    q = ("SELECT TOP 800 main_id, otype, ra, dec FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', "
         f"{centre.ra.deg}, {centre.dec.deg}, {radius_deg})) = 1 AND otype IN ({types})")
    t = Simbad.query_tap(q)
    cols = {c.lower(): c for c in t.colnames}
    return [(" ".join(str(r[cols["main_id"]]).split()), str(r[cols["otype"]]).strip(), float(r[cols["ra"]]),
             float(r[cols["dec"]])) for r in t]


def known_in_field(sol, own=None, extra=None, querier=None):
    """Catalogued nebulae inside your image (SIMBAD), plus your own catalogue and the planetarynebulae.net copy.
    Returns dicts with name, kind and screen position (x right, y down) for the viewer."""
    c, r = sol["centre"], min(sol["radius"], 5.0)
    rows = querier(c, r) if querier else _simbad_field(c, r)
    items = [dict(name=n, kind=FIELD_TYPES.get(t, t), ra=ra, dec=dec) for n, t, ra, dec in rows]
    for o in own or []:
        try:
            items.append(dict(name=o.get("designation") or o["own_id"], kind="your catalogue",
                              ra=float(o["ra_deg"]), dec=float(o["dec_deg"])))
        except (KeyError, TypeError, ValueError):
            continue
    for e in extra or []:
        items.append(dict(name=e.get("label", e["name"]), kind="planetarynebulae.net", ra=e["coord"].ra.deg,
                          dec=e["coord"].dec.deg))
    if not items:
        return []
    ra, dec = np.array([i["ra"] for i in items]), np.array([i["dec"] for i in items])
    near = SkyCoord(ra * u.deg, dec * u.deg).separation(c).deg <= r   # far-away points can fold into the image
    H, W_ = sol["disp"].shape
    out = []
    x, y = sol["to_pixel"](ra[near], dec[near])
    for it, xi, yi in zip([i for i, n in zip(items, near) if n], x, y):
        if np.isfinite(xi) and np.isfinite(yi) and -0.5 <= xi < W_ - 0.5 and -0.5 <= yi < H - 0.5:
            out.append(dict(name=it["name"], kind=it["kind"], x=round(float(xi), 2), y=round(float(H - 1 - yi), 2)))
    return out


CHUNK_BYTES = 768 * 1024   # each piece sent to Python (about 1 MB once encoded)


def uploader_html(callback="aletheia.chunk", where="My Drive › Aletheia › uploads"):
    """Drop zone (Session 21b): drag images onto it or click to choose. Files go to Python in pieces through Colab's
    kernel channel (the ipywidgets Upload button proved unreliable in Colab, Session 17), with a smooth bar."""
    uid = abs(hash(("uploader", time.time()))) % 10 ** 8
    return """
<style>
 #D{uid} {{ display:flex; align-items:center; gap:16px; padding:18px 20px; margin:6px 0 10px; border-radius:16px;
   border:1.5px dashed rgba(100,210,255,0.35); background:rgba(28,28,30,0.55); cursor:pointer;
   transition:border-color .2s ease, background .2s ease, transform .2s ease;
   font-family:Inter,-apple-system,'Segoe UI',sans-serif; color:#F5F5F7; }}
 #D{uid}:hover, #D{uid}.over {{ border-color:#64D2FF; background:rgba(10,132,255,0.12); transform:translateY(-1px); }}
 #D{uid} .ic {{ width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
   background:linear-gradient(135deg, rgba(10,132,255,0.35), rgba(100,210,255,0.15)); flex:none; }}
 #D{uid} .t1 {{ font-size:14px; font-weight:600; }}
 #D{uid} .t1 span {{ color:#64D2FF; }}
 #D{uid} .t2 {{ font-size:12px; color:#8E8E93; margin-top:3px; }}
 #D{uid} .bar {{ height:4px; border-radius:999px; background:rgba(255,255,255,0.08); margin-top:10px; overflow:hidden;
   width:320px; display:none; }}
 #D{uid} .fill {{ height:100%; width:0; border-radius:999px; background:linear-gradient(90deg,#0A84FF,#64D2FF);
   box-shadow:0 0 10px rgba(100,210,255,0.6); transition:width .45s cubic-bezier(.22,.9,.24,1); }}
 #D{uid} .msg {{ font-size:12px; color:#C7C7CC; margin-top:6px; min-height:16px; }}
</style>
<div id="D{uid}" title="Drop images here, or click to choose">
 <div class="ic"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F5F5F7" stroke-width="1.8"
   stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="M7 9l5-5 5 5"/>
   <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg></div>
 <div>
  <div class="t1">Drop your images here <span>or click to choose</span></div>
  <div class="t2">FITS, JPG, PNG, TIFF · any size · saved to {where}</div>
  <div class="bar"><div class="fill"></div></div>
  <div class="msg" id="T{uid}"></div>
 </div>
 <input id="I{uid}" type="file" multiple accept="{accept}" style="display:none">
</div>
<script>
(function() {{
 const zone = document.getElementById("D{uid}"), inp = document.getElementById("I{uid}"),
       out = document.getElementById("T{uid}"), bar = zone.querySelector(".bar"), fill = zone.querySelector(".fill");
 const aletheiaCall = (name, args) => {{   // Colab, or the web edition's bridge (Session 22)
   if (window.google && google.colab && google.colab.kernel) return google.colab.kernel.invokeFunction(name, args, {{}});
   if (window.aletheiaBridge) return window.aletheiaBridge(name, args);
   throw new Error("not connected to Aletheia"); }};
 const CH = {chunk}, OK = {accept_js};
 let busy = false;
 function b64(buf) {{ const u = new Uint8Array(buf); let s = "";
   for (let k = 0; k < u.length; k += 32768) s += String.fromCharCode.apply(null, u.subarray(k, k + 32768));
   return btoa(s); }}
 async function send(files) {{
   if (busy || !files.length) return; busy = true; bar.style.display = "block"; out.style.color = "#C7C7CC";
   for (const f of files) {{
     const ext = "." + f.name.split(".").pop().toLowerCase();
     if (!OK.includes(ext)) {{ out.textContent = f.name + ": not an image type Aletheia reads"; out.style.color = "#FF9F0A"; continue; }}
     const n = Math.max(1, Math.ceil(f.size / CH));
     try {{
       for (let i = 0; i < n; i++) {{
         const piece = await f.slice(i * CH, (i + 1) * CH).arrayBuffer();
         await aletheiaCall("{cb}", [f.name, i, n, b64(piece)]);
         const pc = Math.round(100 * (i + 1) / n);
         fill.style.width = pc + "%";
         out.textContent = "Sending " + f.name + " · " + pc + "% of " + (f.size / 1e6).toFixed(1) + " MB";
       }}
       out.textContent = "Saved " + f.name + " · type a rough position below, then press Plate solve";
     }} catch (e) {{ out.textContent = "Could not send " + f.name + ": " + e; out.style.color = "#FF9F0A"; break; }}
   }}
   busy = false; inp.value = ""; setTimeout(() => {{ bar.style.display = "none"; fill.style.width = "0"; }}, 1200);
 }}
 zone.addEventListener("click", () => {{ if (!busy) inp.click(); }});
 inp.addEventListener("change", () => send(Array.from(inp.files)));
 ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => {{ e.preventDefault(); zone.classList.add("over"); }}));
 ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => {{ e.preventDefault(); zone.classList.remove("over"); }}));
 zone.addEventListener("drop", (e) => send(Array.from(e.dataTransfer.files)));
}})();
</script>""".format(uid=uid, accept=",".join(IMAGE_TYPES), accept_js=json.dumps(list(IMAGE_TYPES)), chunk=CHUNK_BYTES,
                    cb=callback, where=where)
