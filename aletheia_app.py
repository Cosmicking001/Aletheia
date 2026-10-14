"""
aletheia_app: one command + first working interface (Session 8)

analyse(position_text, size_arcmin=None, own=None, zone_override=None) -> dict
    Everything Aletheia knows so far for one position: coordinates, PN G name,
    known-object verdict (Session 4), imposter flags (Session 5).
AletheiaApp(base).show()
    Interactive screen: position, object size, cutout size, automatic search radius
    (editable), survey checkboxes with info buttons, Show all, stretch, smoothing,
    star muting (experimental), faint-object hints, download of annotated cutouts.
"""
import math
import shutil
from pathlib import Path

import numpy as np

import aletheia_coords as ac
import aletheia_cspn as cs
import aletheia_crossmatch as xm
import aletheia_imposters as im
import aletheia_panels as pn
import aletheia_scorecard as sc
import aletheia_compact as cp
import aletheia_report as rp
import aletheia_upload as up
import aletheia_health as hl

# --------------------------------------------------------------------------- survey info
SURVEY_INFO = {
    "dss2_red": ("Photographic red plates of the whole sky (1980s and 90s).",
                 "Its band includes H-alpha and [N II], so many nebulae show. Being a second epoch, it helps spot "
                 "plate flaws, moving objects and changes. Check any feature against a CCD survey."),
    "ps_r": ("Deep modern CCD survey of the sky north of Dec -30.",
             "The r band contains H-alpha, so emission nebulae show, mixed with starlight. Sharp stars make it good "
             "for judging whether a smudge is a galaxy, a star group or diffuse gas."),
    "decaps_r": ("Deep CCD survey of the southern Galactic plane.",
                 "Built for crowded Milky Way fields; very sharp stars help separate faint nebulosity from blended "
                 "starlight in the south."),
    "legacy_r": ("Very deep CCD imaging, mostly away from the Galactic plane.",
                 "Excellent for faint high-latitude PNe and for recognising background galaxies, which are a common "
                 "imposter."),
    "dss2_col": ("Colour composite of the DSS plates.",
                 "Colour separates red emission (H-alpha), blue reflection nebulae and yellowish galaxies at a glance."),
    "ps_col": ("Pan-STARRS colour composite (z, z+g, g).",
               "[O III] 5007 falls in g, so high-excitation PNe tend to look blue-green; H II regions and dust look "
               "redder."),
    "decaps_col": ("DECaPS colour composite.", "Colour helps separate emission, reflection and starlight in the south."),
    "legacy_col": ("Legacy Surveys colour composite.",
                   "Background galaxies stand out by shape and colour; useful to rule out a galaxy imposter."),
    "iphas_col": ("IPHAS colour composite.", "Not available at CDS at present."),
    "vphas_col": ("VPHAS+ colour composite (southern plane).",
                  "Combines broad bands with H-alpha, so emission regions show a distinct tint."),
    "iphas_ha": ("Narrow-band H-alpha survey of the northern Galactic plane (|b| < 5 deg).",
                 "The key survey for faint PNe in the north. Always compare with IPHAS r: what is brighter in H-alpha "
                 "than in r is emission, not starlight."),
    "iphas_r": ("Broad r band from IPHAS: the starlight reference.",
                "Used to subtract starlight from H-alpha (H-alpha minus r), which leaves only the emission."),
    "vphas_ha": ("Narrow-band H-alpha survey of the southern Galactic plane.",
                 "The southern counterpart of IPHAS; compare with VPHAS+ r in the same way."),
    "vphas_r": ("Broad r band from VPHAS+: the starlight reference.", "Used for H-alpha minus r in the south."),
    "shs_ha": ("AAO/UKST SuperCOSMOS H-alpha Survey of the southern plane.",
               "About 1 arcsec resolution; the survey behind the MASH PN catalogues. Faint emission shows well; "
               "check plate flaws against VPHAS+."),
    "shs_sr": ("SHS short-red continuum band.", "Not available at CDS at present."),
    "shassa": ("Wide-angle southern H-alpha survey, coarse (about 48 arcsec pixels).",
               "Too coarse for small PNe, but shows very large faint nebulae and the surrounding H II environment."),
    "vtss": ("Wide-angle northern H-alpha survey, coarse (about 1.6 arcmin pixels).",
             "Best for very large nebulae and for seeing whether a field sits inside a big H II complex."),
    "unwise_w1": ("Infrared 3.4 microns: mostly starlight.",
                  "Marks stellar crowding and red stars. A PN shell is rarely visible here; a bright compact source "
                  "can be a dusty star or YSO."),
    "allwise_w3": ("Infrared 12 microns: warm dust and PAH emission.",
                   "H II regions are very bright at 12 microns; many PNe glow too. Strong 12 micron emission with "
                   "little H-alpha points to dust or a young stellar object."),
    "allwise_w4": ("Infrared 22 microns: warm dust.",
                   "Compact PNe are often bright here. Together with W3 it helps separate PNe from H II regions."),
    "allwise_col": ("Mid-infrared colour composite.",
                    "Red means warm dust, blue-green means starlight: dusty nebulae and H II regions show in red."),
    "galex_nuv": ("Near-ultraviolet.", "Hot central stars and some PN shells show; most cool stars are faint here."),
    "galex_fuv": ("Far-ultraviolet.", "The strongest hint of a very hot star: a PN central star is often bright in FUV."),
    "galex_col": ("GALEX colour composite.", "Blue sources are the hottest: candidates for a central star."),
    "nvss": ("Radio continuum at 1.4 GHz (45 arcsec beam).",
             "Ionised gas emits free-free radio waves: compact PNe and H II regions can show, as do background radio "
             "galaxies. Supernova remnants show non-thermal emission."),
    "diff": ("H-alpha minus r: starlight cancels, emission remains.",
             "The single best view for faint emission nebulae. Residual star shapes and tile steps are artefacts."),
}
MUTING_INFO = ("Star muting is an experimental viewing aid, never used for measurements. Measured by "
               "injection-recovery tests: Gentle keeps 88 to 99 percent of nebulosity and removes about 65 percent "
               "of stars. Strong removes 99 to 100 percent of stars and keeps 93 to 97 percent of faint nebulosity, "
               "but can shave narrow bright arcs under stars, so it switches off in crowded fields. Proper star "
               "subtraction comes in a later session.")


# --------------------------------------------------------------------------- engine
def warm_imports():
    """Session 18c: astroquery reads its settings when first imported. When several threads do that first import at
    the same moment it can fail ("Cannot automatically determine get_config module"), so it is done once here, on
    the main thread, before any search starts."""
    import importlib
    for m in ("astroquery.simbad", "astroquery.vizier", "astroquery.hips2fits", "pyvo"):
        try:
            importlib.import_module(m)
        except Exception as exc:
            hl.log_problem(f"start-up import of {m}", exc)


warm_imports()
CHECK_BUDGET_S = 120   # Session 16: the screen never waits longer than this for the catalogue checks


def analyse(position_text, size_arcmin=None, own=None, zone_override=None):
    """The one command: everything known so far for one position (catalogue checks run in parallel)."""
    from concurrent.futures import ThreadPoolExecutor
    coord = ac.parse_position(position_text)
    desc = ac.describe(coord)
    zone, search = xm.zones(size_arcmin)
    if zone_override:
        zone = float(zone_override)
        search = min(xm.MAX_SEARCH_ARCMIN, max(xm.MIN_SEARCH_ARCMIN, 1.5 * zone))
    from concurrent.futures import wait
    pool = ThreadPoolExecutor(max_workers=4)
    jobs = dict(compact=pool.submit(cp.compact_evidence, coord, size_arcmin),
                known=pool.submit(xm.check_known, coord, size_arcmin, own, None, zone_override),
                imposters=pool.submit(im.check_imposters, coord, size_arcmin, zone_override),
                cspn=pool.submit(cs.find_central_stars, coord, size_arcmin))
    done, _ = wait(jobs.values(), timeout=CHECK_BUDGET_S)
    pool.shutdown(wait=False, cancel_futures=True)   # never wait for a service that does not answer

    def get(key, fallback):
        f = jobs[key]
        if f not in done:
            hl.log_problem(f"analyse: {key}", note=f"no answer within {CHECK_BUDGET_S} s at {position_text}")
            return fallback(f"no answer within {CHECK_BUDGET_S} s; press Analyse again to retry")
        try:
            return f.result()
        except Exception as exc:
            hl.log_problem(f"analyse: {key}", exc, note=f"position {position_text}")
            return fallback(" ".join(str(exc).split())[:80])
    known = get("known", lambda why: dict(verdict=f"NOT CHECKED ({why})", inside=[], nearby=[], own=[], galaxies=[]))
    if jobs["known"] in done and jobs["known"].exception() is not None:   # failed quickly: one more try (Session 18c)
        try:
            known = xm.check_known(coord, size_arcmin, own, None, zone_override)
        except Exception as exc:
            hl.log_problem("analyse: known (second try)", exc, note=f"position {position_text}")
    imp = get("imposters", lambda why: dict(primary=f"not checked ({why})", flags=[], variables=[],
                                            zone_arcmin=zone, not_checked=True))
    cspn = get("cspn", lambda why: dict(stars=[], radius=cs.search_radius_arcsec(size_arcmin), error=why))
    compact = get("compact", lambda why: dict(applies=False, note=f"compact-object tests not done: {why}"))
    return dict(compact=compact, coord=coord, desc=desc, size=size_arcmin, zone=zone, search=search, known=known, imposters=imp,
                official=official_name(known), cspn=cspn)


RANK_ACCURACY = None   # set at start-up from config/ranking_accuracy.json (Session 12)


def cspn_html(result, top=3):
    """Readable summary: the top candidates in plain words, the best one highlighted with its reason."""
    c = result.get("cspn") or {}
    if c.get("error"):
        return f"<span style='color:#FF9F0A'>Central star search failed: {c['error'][:100]}</span>"
    stars = c.get("stars", [])
    head = (f"<div style='margin-top:12px'><b>White dwarf / central star</b> "
            f"<span style='color:#8E8E93'>· {len(stars)} Gaia DR3 stars within {c.get('radius', 0):.0f}\" · "
            f"numbers match the orange markers</span></div>")
    if RANK_ACCURACY:
        a = RANK_ACCURACY
        head += (f"<div style='color:#8E8E93;font-size:12px'>Best guess plus a top-3 shortlist. Tested on {a['n']} "
                 f"confirmed central stars: the right star was #1 in {a['top1']:.0%} (likely range "
                 f"{a['lo']:.0%}-{a['hi']:.0%}) and in the top 3 in {a['top3']:.0%}.</div>")
    if not stars:
        return head + "<div style='color:#8E8E93'>No Gaia DR3 star in the search area.</div>"
    lines = []
    for i, st in enumerate(stars[:top]):
        line, why = cs.plain_summary(st, len(stars))
        extra = "".join(f"<div style='color:#8E8E93;margin-left:18px'>· {p}</div>" for p in st.get("plausibility", []))
        if i == 0:
            lines.append(f"<div style='border-left:2px solid #64D2FF;padding:2px 8px;margin:4px 0'>"
                         f"<span style='color:#64D2FF'>{line}</span>"
                         f"{'<br><span style=color:#8E8E93>why: ' + why + '</span>' if why else ''}{extra}</div>")
        else:
            lines.append(f"<div style='padding:2px 10px'>{line}{extra}</div>")
    note = ("<div style='color:#6E6E73;font-size:11px;margin-top:4px'>Catalogue values: Gaia DR3; distances "
            "Bailer-Jones et al. 2021; white dwarf probability, temperature, gravity and mass Gentile Fusillo et al. "
            "2021; central-star catalogues Chornay &amp; Walton 2021, Gonzalez-Santamaria et al. 2021. Nebula radius "
            "and absolute magnitude are computed estimates without extinction correction.</div>")
    return head + "".join(lines) + note


def cspn_table_html(result):
    """Full table, shown on request. Columns with no values at all are hidden."""
    stars = (result.get("cspn") or {}).get("stars", [])
    if not stars:
        return ""
    cols = [("#", lambda s: str(s["rank"])), ("Gaia DR3", lambda s: s["source"]),
            ("offset", lambda s: f"{s['sep']:.1f}\""), ("G", lambda s: _fmt(s.get("G"), ".2f")),
            ("BP-RP", lambda s: _fmt(s.get("bp_rp"), "+.2f")),
            ("distance pc", lambda s: (f"{s['dist']:.0f} ({s['dist_lo']:.0f}-{s['dist_hi']:.0f})"
                                       if s.get("dist") and s.get("dist_lo") and s.get("dist_hi") else
                                       _fmt(s.get("dist"), ".0f"))),
            ("abs G", lambda s: _fmt(s.get("abs_G"), ".1f")), ("RUWE", lambda s: _fmt(s.get("ruwe"), ".2f")),
            ("P_WD", lambda s: _fmt(s.get("p_wd"), ".2f")), ("Teff K", lambda s: _fmt(s.get("teff"), ",.0f")),
            ("log g", lambda s: _fmt(s.get("logg"), ".2f")), ("mass", lambda s: _fmt(s.get("mass"), ".2f")),
            ("nebula radius pc", lambda s: _fmt(s.get("radius_pc"), ".2f")),
            ("catalogued", lambda s: ", ".join(x.split(" ")[0] for x in s["cspn"])),
            ("score", lambda s: f"{s['score']:.2f}")]
    keep = [(h, f) for h, f in cols if any(f(s) for s in stars)]
    head = "".join(f"<td style='padding:0 8px'>{h}</td>" for h, _ in keep)
    body = "".join("<tr>" + "".join(f"<td style='padding:0 8px'>{f(s)}</td>" for _, f in keep) + "</tr>"
                   for s in stars[:25])
    return (f"<table style='font-family:monospace;font-size:11px'><tr style='color:#8E8E93'>{head}</tr>{body}</table>")


def _fmt(v, spec):
    return "" if v is None else format(v, spec)


def cspn_marks(result, fov, pixels=400, top=5):
    """Marker positions (fraction from left, fraction from top) for the top candidates."""
    stars = (result.get("cspn") or {}).get("stars", [])[:top]
    if not stars:
        return []
    w = pn.grid_wcs(result["coord"], fov, pixels)
    out = []
    for st in stars:
        x, y = w.world_to_pixel(st["coord"])
        out.append((float(x + 0.5) / pixels, 1.0 - float(y + 0.5) / pixels, str(st["rank"])))
    return out


def official_name(known):
    """The catalogued PN G name, when a known PN sits in the zone (names are frozen identifiers)."""
    for f in known.get("inside", []):
        if f["category"] == "KNOWN PN" and "Acker" in f["source"]:
            return "PN G" + str(f["name"]).replace("PN G", "").strip()
    for f in known.get("inside", []):
        if f["category"] == "KNOWN PN" and "Kohoutek" in f["source"]:
            return "PN G" + str(f["name"]).replace("PN G", "").strip()
    return None


def summary_lines(result):
    k, i = result["known"], result["imposters"]
    inside, nearby = xm.summarise(k)
    lines = xm.identify(k, result.get("official"))[1]
    lines += [f"Known object check: {k['verdict']}" + (f" ({inside})" if inside else "")]
    if nearby:
        lines.append(f"Nearby, outside the zone: {nearby}")
    if k.get("own") and k["verdict"] != "NO MATCH" and min(h["sep_arcmin"] for h in k["own"]) < 0.1:
        lines.append("Note: your own catalogue has an object at this position, so the known entry above may be "
                     "your own object, already listed elsewhere")
    if k.get("own"):
        lines.append("Your catalogue: " + "; ".join(f"{h['name']} ({h['sep_arcmin']:.2f}')" for h in k["own"]))
    lines.append(f"Imposter flags: {i['primary']}" + (f" ({im.summarise(i)})" if i["flags"] else ""))
    if i.get("variables"):
        lines.append("Variable stars within 30 arcsec: " + "; ".join(i["variables"][:3]))
    return lines


# --------------------------------------------------------------------------- image processing for display
MUTABLE_GROUPS = ("Optical", "H-alpha")  # star muting is meaningless for infrared, UV and radio


def bin_image(d, factor):
    if factor <= 1:
        return d
    import warnings
    n = (d.shape[0] // factor) * factor
    m = (d.shape[1] // factor) * factor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # blank survey edges are expected
        return np.nanmean(d[:n, :m].reshape(n // factor, factor, m // factor, factor), axis=(1, 3))


def prepare(item, smoothing="none", muting="off", gaia=None, wcs=None, protect_px=0.0):
    """Muting and smoothing (slow steps, cached by the app). Returns (float image, note)."""
    from scipy import ndimage
    d = item["data"]
    if d.ndim == 3:
        return d, ""
    work, note = np.array(d, dtype=float), ""
    can_mute = item.get("group") in MUTABLE_GROUPS and not item.get("is_diff")
    if muting != "off" and not can_mute:
        note = "star muting not used for this wavelength"
    elif muting == "gentle":
        work = pn.mute_stars(work)
    elif muting == "strong":
        if gaia is None or wcs is None:
            work, note = pn.mute_stars(work), "Gaia stars unavailable: Gentle used"
        else:
            muted, radii = pn.mute_stars_gaia(work, wcs, *gaia, protect_px=protect_px, return_radii=True)
            cover = float(np.nansum(np.pi * np.asarray(radii) ** 2) / work.size)
            if cover > 0.30:
                work, note = pn.mute_stars(work), "crowded field: Strong switched to Gentle"
            else:
                work = muted
    if smoothing == "bin2":
        work = bin_image(work, 2)
    elif smoothing == "bin4":
        work = bin_image(work, 4)
    elif smoothing == "gauss":
        f = np.isfinite(work)
        work = ndimage.gaussian_filter(np.where(f, work, np.nanmedian(work[f]) if f.any() else 0), 1.5)
    return work, note


def levels(work, black_sigma=2.0, white_sigma=10.0, symmetric=False):
    """Fast levels stretch (black and white point in units of sky noise). Returns 0..1 image."""
    if work.ndim == 3:
        return work
    return pn.nebula_stretch(work, symmetric=symmetric, low_sigma=black_sigma, high_sigma=white_sigma)


def to_png(img01, ring_frac=None, size=300, highlight=False, marks=None):
    """Image (0..1 grey, or RGB uint8, FITS orientation) -> PNG bytes, North up, with object ring."""
    import io
    from PIL import Image, ImageDraw
    if img01.ndim == 3:
        arr = np.asarray(img01[..., :3], dtype=np.uint8)
    else:
        arr = (np.clip(np.nan_to_num(img01), 0, 1) * 255).astype(np.uint8)
    pic = Image.fromarray(np.flipud(arr)).convert("RGB").resize((size, size), Image.NEAREST)
    draw = ImageDraw.Draw(pic)
    if ring_frac:
        r = ring_frac * size / 2
        c = size / 2
        for a in range(0, 360, 8):  # dashed teal ring
            draw.arc([c - r, c - r, c + r, c + r], a, a + 4, fill=(95, 212, 196), width=1)
    for fx, fy, label in marks or []:
        x, y, r = fx * size, fy * size, max(4, size / 60)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(242, 166, 90), width=1)
        draw.text((x + r + 1, y - r - 1), label, fill=(242, 166, 90))
    if highlight:
        draw.rectangle([0, 0, size - 1, size - 1], outline=(95, 212, 196), width=3)
    buf = io.BytesIO()
    pic.save(buf, format="PNG")
    return buf.getvalue()


def histogram_png(work, black_sigma, white_sigma, width=300, height=70):
    """Small histogram of sky-noise units with the black and white points marked (levels view)."""
    import io
    from PIL import Image, ImageDraw
    pic = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(pic)
    if work.ndim == 2:
        f = np.isfinite(work)
        if f.any():
            med = np.median(work[f])
            s = pn._robust_sigma(work[f] - med) or 1.0
            z = (work[f] - med) / s
            lo, hi = -5.0, max(40.0, white_sigma + 5)
            h, _ = np.histogram(np.clip(z, lo, hi), bins=width, range=(lo, hi))
            h = np.log1p(h)
            h = h / h.max() * (height - 6) if h.max() > 0 else h
            for x, v in enumerate(h):
                draw.line([x, height - 1, x, height - 1 - v], fill=(150, 150, 150))
            for val, col in ((-black_sigma, (95, 212, 196)), (white_sigma, (242, 166, 90))):
                x = int((val - lo) / (hi - lo) * (width - 1))
                draw.line([x, 0, x, height], fill=col, width=2)
    buf = io.BytesIO()
    pic.save(buf, format="PNG")
    return buf.getvalue()


def faint_hint(item, size_arcmin, fov):
    """True when the object zone shows little excess over its surroundings (hint only)."""
    from scipy import ndimage
    d = item["data"]
    if d is None or d.ndim == 3:
        return False
    f = np.isfinite(d)
    if not f.any():
        return False
    sm = ndimage.gaussian_filter(np.where(f, d, np.nanmedian(d[f])), 3.0)
    n = d.shape[0]
    rz = max(4.0, 0.5 * (size_arcmin or 1.0) / (fov / n))
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    zone, ann = rr <= rz, (rr > 1.3 * rz) & (rr < min(2.5 * rz, n / 2 - 1))
    if ann.sum() < 100:
        return False
    s = pn._robust_sigma(sm[ann])
    return bool(s > 0 and (np.mean(sm[zone]) - np.median(sm[ann])) / s < 3.0)


def gaia_for_field(coord, fov_arcmin):
    from astroquery.vizier import Vizier
    import astropy.units as u
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"], column_filters={"Gmag": "<19"}, row_limit=80000)
    v.TIMEOUT = 180
    t = v.query_region(coord, width=fov_arcmin * u.arcmin, height=fov_arcmin * u.arcmin, catalog="I/355/gaiadr3")
    if not t:
        return np.array([]), np.array([]), np.array([])
    t = t[0]
    return np.array(t["RA_ICRS"], float), np.array(t["DE_ICRS"], float), np.array(t["Gmag"], float)


# --------------------------------------------------------------------------- fast levels (Session 8c)
Z_LO, Z_HI, HIST_W, HIST_H = -6.0, 60.0, 260, 64


def normalise(work):
    """Once per image: convert to units of sky noise (z = (value - sky) / sigma)."""
    if work.ndim == 3:
        return None
    f = np.isfinite(work)
    if not f.any():
        return np.zeros_like(work, dtype=np.float32)
    med = float(np.median(work[f]))
    s = float(pn._robust_sigma(work[f] - med)) or 1.0
    z = ((work - med) / s).astype(np.float32)
    z[~f] = Z_LO
    return z


def levels_z(z, black, white, symmetric=False):
    """Per slider move: pure arithmetic on the cached z image (fast)."""
    if symmetric:
        return np.clip(z / max(white, 1e-3), -1, 1) * 0.5 + 0.5
    x = np.clip((z + black) / max(white + black, 1e-3), 0, 1)
    return np.arcsinh(6 * x) / np.arcsinh(6)


def histogram_base(z):
    """Once per image: the histogram picture without markers."""
    from PIL import Image, ImageDraw
    pic = Image.new("RGB", (HIST_W, HIST_H), (10, 10, 10))
    if z is None:
        return pic
    h, _ = np.histogram(np.clip(z, Z_LO, Z_HI), bins=HIST_W, range=(Z_LO, Z_HI))
    h = np.log1p(h)
    h = h / h.max() * (HIST_H - 4) if h.max() > 0 else h
    d = ImageDraw.Draw(pic)
    for x, v in enumerate(h):
        d.line([x, HIST_H - 1, x, HIST_H - 1 - v], fill=(140, 140, 140))
    return pic


def histogram_png_fast(base, black, white):
    import io
    from PIL import ImageDraw
    pic = base.copy()
    d = ImageDraw.Draw(pic)
    for val, col in ((-black, (95, 212, 196)), (white, (242, 166, 90))):
        x = int((min(max(val, Z_LO), Z_HI) - Z_LO) / (Z_HI - Z_LO) * (HIST_W - 1))
        d.line([x, 0, x, HIST_H], fill=col, width=2)
    buf = io.BytesIO()
    pic.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- high resolution (Session 9b)
# approximate native pixel scales, arcsec/pixel
NATIVE_ARCSEC = {"dss2_red": 1.0, "dss2_col": 1.0, "ps_r": 0.25, "ps_col": 0.25, "decaps_r": 0.26,
                 "decaps_col": 0.26, "legacy_r": 0.262, "legacy_col": 0.262, "iphas_ha": 0.333, "iphas_r": 0.333,
                 "vphas_ha": 0.21, "vphas_r": 0.21, "vphas_col": 0.21, "shs_ha": 0.67, "shs_sr": 0.67,
                 "shassa": 47.6, "vtss": 96.0, "unwise_w1": 2.75, "allwise_w3": 1.375, "allwise_w4": 1.375,
                 "allwise_col": 1.375, "galex_nuv": 1.5, "galex_fuv": 1.5, "galex_col": 1.5, "nvss": 15.0}
MAX_HIRES_PX = 2400


def hires_pixels(key, fov_arcmin):
    """Pixels needed to reach the survey's native scale, between 400 and MAX_HIRES_PX."""
    scale = NATIVE_ARCSEC.get(key, 1.0)
    return int(min(MAX_HIRES_PX, max(400, math.ceil(fov_arcmin * 60.0 / scale))))


def zoom_viewer_html(jpeg_bytes, wcs, uid, title, width=760):
    """Self-contained pan/zoom viewer: mouse wheel zooms at the cursor, drag pans, double-click resets,
    and the RA/Dec under the cursor is shown (gnomonic projection, checked against astropy)."""
    import base64
    b64 = base64.b64encode(jpeg_bytes).decode()
    (cr1, cr2), (cp1, cp2), (cd1, cd2) = wcs.wcs.crval, wcs.wcs.crpix, wcs.wcs.cdelt
    n = int(round(2 * cp2 - 1))
    return f"""
<div style="font-family:monospace;color:#F5F5F7;margin:6px 0">{title}
 <span style="color:#8E8E93">· wheel = zoom · drag = pan · double-click = reset</span></div>
<div id="v{uid}" style="position:relative;width:{width}px;height:{width}px;overflow:hidden;background:#000;
 cursor:crosshair;border:1px solid #333">
 <img id="i{uid}" src="data:image/jpeg;base64,{b64}" draggable="false"
  style="position:absolute;left:0;top:0;width:{width}px;height:{width}px;transform-origin:0 0;image-rendering:auto">
</div>
<div id="c{uid}" style="font-family:monospace;color:#64D2FF;margin-top:4px">RA --  Dec --</div>
<script>
(function(){{
 const box=document.getElementById("v{uid}"), img=document.getElementById("i{uid}"),
       out=document.getElementById("c{uid}"), W={width}, N={n};
 let s=1, tx=0, ty=0, drag=null;
 const R=Math.PI/180, ra0={cr1}*R, de0={cr2}*R;
 function apply(){{ img.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{s}})`; }}
 function sky(px,py){{ // px, py: FITS 1-based pixel
  const xi={cd1}*(px-{cp1})*R, eta={cd2}*(py-{cp2})*R, rho=Math.hypot(xi,eta), c=Math.atan(rho);
  if(rho===0) return [{cr1},{cr2}];
  const de=Math.asin(Math.cos(c)*Math.sin(de0)+eta*Math.sin(c)*Math.cos(de0)/rho);
  const ra=ra0+Math.atan2(xi*Math.sin(c), rho*Math.cos(de0)*Math.cos(c)-eta*Math.sin(de0)*Math.sin(c));
  return [((ra/R)%360+360)%360, de/R];
 }}
 function hms(d){{ let h=d/15, H=Math.floor(h), m=(h-H)*60, M=Math.floor(m), S=(m-M)*60;
   if(S>=59.995){{S=0;M+=1;}} if(M>=60){{M=0;H=(H+1)%24;}}
   return String(H).padStart(2,"0")+":"+String(M).padStart(2,"0")+":"+S.toFixed(2).padStart(5,"0"); }}
 function dms(d){{ const g=d<0?"-":"+"; d=Math.abs(d); let D=Math.floor(d), m=(d-D)*60, M=Math.floor(m), S=(m-M)*60;
   if(S>=59.95){{S=0;M+=1;}} if(M>=60){{M=0;D+=1;}}
   return g+String(D).padStart(2,"0")+":"+String(M).padStart(2,"0")+":"+S.toFixed(1).padStart(4,"0"); }}
 box.addEventListener("wheel", e=>{{ e.preventDefault(); const r=box.getBoundingClientRect(),
   mx=e.clientX-r.left, my=e.clientY-r.top, f=e.deltaY<0?1.2:1/1.2, ns=Math.min(40,Math.max(1,s*f));
   tx=mx-(mx-tx)*ns/s; ty=my-(my-ty)*ns/s; s=ns; if(s===1){{tx=0;ty=0;}} apply(); }}, {{passive:false}});
 box.addEventListener("mousedown", e=>{{ drag=[e.clientX-tx, e.clientY-ty]; }});
 window.addEventListener("mouseup", ()=>{{ drag=null; }});
 box.addEventListener("dblclick", ()=>{{ s=1; tx=0; ty=0; apply(); }});
 box.addEventListener("mousemove", e=>{{
   if(drag){{ tx=e.clientX-drag[0]; ty=e.clientY-drag[1]; apply(); }}
   const r=box.getBoundingClientRect(), ix=((e.clientX-r.left)-tx)/s*N/W, iy=((e.clientY-r.top)-ty)/s*N/W;
   if(ix<0||iy<0||ix>N||iy>N){{ out.textContent="RA --  Dec --"; return; }}
   const p=sky(ix+0.5, N-iy+0.5);  // image rows run North (top) to South: FITS y = N - row
   out.textContent="RA "+hms(p[0])+"   Dec "+dms(p[1])+"   ("+p[0].toFixed(5)+", "+p[1].toFixed(5)+")   zoom x"+s.toFixed(1);
 }});
}})();
</script>"""


def to_jpeg(img01, ring_frac=None, marks=None, quality=93):
    """Full-resolution rendering for the zoom viewer (no resizing)."""
    import io
    from PIL import Image, ImageDraw
    if img01.ndim == 3:
        arr = np.asarray(img01[..., :3], dtype=np.uint8)
    else:
        arr = (np.clip(np.nan_to_num(img01), 0, 1) * 255).astype(np.uint8)
    pic = Image.fromarray(np.flipud(arr)).convert("RGB")
    size = pic.size[0]
    draw = ImageDraw.Draw(pic)
    lw = max(1, size // 600)
    if ring_frac:
        r, c = ring_frac * size / 2, size / 2
        for a in range(0, 360, 6):
            draw.arc([c - r, c - r, c + r, c + r], a, a + 3, fill=(95, 212, 196), width=lw)
    for fx, fy, label in marks or []:
        x, y, r = fx * size, fy * size, max(6, size / 80)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(242, 166, 90), width=lw)
        draw.text((x + r + 2, y - r - 2), label, fill=(242, 166, 90))
    buf = io.BytesIO()
    pic.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def placeholder_png(width, text="", height=None):
    import io
    from PIL import Image, ImageDraw
    pic = Image.new("RGB", (width, height or width), (12, 12, 12))
    if text:
        ImageDraw.Draw(pic).text((20, (height or width) // 2 - 6), text, fill=(120, 120, 120))
    buf = io.BytesIO()
    pic.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- interface
# --------------------------------------------------------------------------- look (Session 19)
APP_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.aletheia { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
  --text:#F5F5F7; --text2:#8E8E93; --line:rgba(255,255,255,0.08); --glass:rgba(28,28,30,0.62); --fill:rgba(118,118,128,0.18);
  --blue:#0A84FF; --cyan:#64D2FF;
  background: radial-gradient(1100px 520px at 8% -8%, rgba(10,132,255,0.16), transparent 60%),
              radial-gradient(900px 480px at 100% 0%, rgba(191,90,242,0.10), transparent 60%), #000;
  color:var(--text); padding:22px 24px 28px; border-radius:22px; font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased; }
.aletheia, .aletheia *:not(canvas) { font-family:var(--font) !important; letter-spacing:-0.005em; }
.aletheia .widget-label, .aletheia label, .aletheia .widget-label-basic, .aletheia .widget-label-basic span {
  color:var(--text2) !important; font-size:13px !important; }
.aletheia input[type=text], .aletheia select, .aletheia textarea {
  background:var(--fill) !important; color:var(--text) !important; border:1px solid var(--line) !important;
  border-radius:10px !important; height:36px !important; padding:0 12px !important; font-size:14px !important;
  box-shadow:none !important; outline:none !important; transition:box-shadow .15s ease; }
.aletheia input[type=text]::placeholder { color:#6E6E73 !important; }
.aletheia input[type=text]:focus, .aletheia select:focus { box-shadow:0 0 0 3px rgba(10,132,255,0.45) !important; }
.aletheia input[type=checkbox] { accent-color:var(--blue); width:16px; height:16px; }
.aletheia button { background:rgba(255,255,255,0.08) !important; color:var(--text) !important;
  border:1px solid rgba(255,255,255,0.10) !important; border-radius:999px !important; height:34px !important;
  padding:0 16px !important; font-size:13px !important; font-weight:500 !important; box-shadow:none !important;
  -webkit-backdrop-filter:blur(20px) saturate(160%); backdrop-filter:blur(20px) saturate(160%);
  transition:background .15s ease, transform .1s ease; }
.aletheia button:hover { background:rgba(255,255,255,0.14) !important; }
.aletheia button:active { transform:scale(0.98); }
.aletheia button.mod-info { background:var(--blue) !important; border-color:transparent !important; color:#fff !important; }
.aletheia button.mod-info:hover { background:#409CFF !important; }
.aletheia button:disabled { opacity:.45; }
.aletheia button.al-circle-on { background:#FF453A !important; border-color:transparent !important; color:#fff !important; }
.aletheia .al-seg { background:rgba(118,118,128,0.24); border-radius:11px; padding:2px; gap:2px; width:fit-content; }
.aletheia .al-seg button { background:transparent !important; border:none !important; border-radius:9px !important;
  height:30px !important; color:var(--text) !important; }
.aletheia .al-seg button.al-active { background:#636366 !important; box-shadow:0 3px 8px rgba(0,0,0,.35) !important; }
.aletheia .al-panel { background:var(--glass); border:1px solid var(--line); border-radius:18px; padding:14px 16px;
  margin:10px 0; -webkit-backdrop-filter:blur(30px) saturate(160%); backdrop-filter:blur(30px) saturate(160%); }
.aletheia .progress-bar { background:var(--blue) !important; }
.aletheia .widget-html-content { color:var(--text); font-size:13px; }
.al-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap:12px; margin:12px 0; }
.al-card { background:var(--glass, rgba(28,28,30,0.62)); border:1px solid rgba(255,255,255,0.08); border-radius:18px;
  padding:16px 18px; font-size:13.5px; line-height:1.55; color:#F5F5F7;
  -webkit-backdrop-filter:blur(30px) saturate(160%); backdrop-filter:blur(30px) saturate(160%);
  box-shadow:0 1px 0 rgba(255,255,255,0.04) inset, 0 18px 40px rgba(0,0,0,0.35); }
.al-card.wide { grid-column:1 / -1; }
.al-card h4 { margin:0 0 8px 0; font-size:11px; letter-spacing:.08em; color:#8E8E93; font-weight:600;
  text-transform:uppercase; }
.al-title { font-size:22px; font-weight:600; letter-spacing:-0.02em; margin-bottom:8px; }
.al-badge { display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:500;
  margin:0 8px 8px 0; }
.al-muted { color:#8E8E93; }
.al-head { display:flex; align-items:center; gap:14px; }
.aletheia { color-scheme:dark; }
.aletheia select option, .aletheia select optgroup { background:#1C1C1E !important; color:#F5F5F7 !important; }
.aletheia input[type=range] { accent-color:var(--blue); }
.aletheia .aladin-input-text, .aletheia .aladin-location input { background:rgba(28,28,30,0.78) !important;
  color:#F5F5F7 !important; border:1px solid rgba(255,255,255,0.12) !important; height:auto !important;
  padding:4px 8px !important; font-size:12px !important; border-radius:8px !important; }
.aletheia .aladin-location-copy, .aletheia .aladin-btn { background:rgba(28,28,30,0.78) !important;
  color:#F5F5F7 !important; border-color:rgba(255,255,255,0.12) !important; height:auto !important; }
.aletheia ::-webkit-scrollbar { width:8px; height:8px; }
.aletheia ::-webkit-scrollbar-track { background:transparent; }
.aletheia ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.16); border-radius:8px; }
.aletheia canvas { border-radius:12px; }
.aletheia .al-status .widget-html-content { display:inline-block; background:rgba(28,28,30,0.62);
  border:1px solid rgba(255,255,255,0.08); border-radius:999px; padding:6px 14px; color:#C7C7CC; font-size:12.5px;
  -webkit-backdrop-filter:blur(20px); backdrop-filter:blur(20px); }
.aletheia .al-status .widget-html-content:empty { display:none; }
.al-card:hover { border-color:rgba(255,255,255,0.14); }
.aletheia .al-panel:hover { border-color:rgba(255,255,255,0.12); }
.al-prog { display:flex; align-items:center; gap:12px; margin-left:12px; }
.al-track { position:relative; width:340px; height:6px; border-radius:999px; background:rgba(255,255,255,0.08);
  overflow:hidden; }
.al-fill { position:absolute; left:0; top:0; bottom:0; width:var(--b); border-radius:999px;
  background:linear-gradient(90deg, #0A84FF, #64D2FF); box-shadow:0 0 14px rgba(100,210,255,0.55); overflow:hidden;
  animation:alFill .9s cubic-bezier(.22,.9,.24,1) both; }
@keyframes alFill { from { width:var(--a); } to { width:var(--b); } }
.al-fill::after { content:""; position:absolute; inset:0; border-radius:999px;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent); width:40%;
  animation:alShine 1.6s ease-in-out infinite; }
@keyframes alShine { from { transform:translateX(-120%); } to { transform:translateX(320%); } }
.al-pct { font-size:12.5px; font-weight:600; color:#64D2FF; min-width:40px; }
.al-step { font-size:12px; color:#8E8E93; }
.al-loader { position:relative; width:46px; height:46px; flex:none; }
.al-loader img, .al-loader svg.al-mark { position:absolute; left:7px; top:7px; width:32px; height:32px;
  animation:alBreathe 2.4s ease-in-out infinite; }
@keyframes alBreathe { 0%, 100% { opacity:.2; transform:scale(.9); filter:drop-shadow(0 0 0 rgba(100,210,255,0)); }
  50% { opacity:1; transform:scale(1); filter:drop-shadow(0 0 10px rgba(100,210,255,0.75)); } }
.al-orbit { position:absolute; inset:0; border-radius:50%; border:1.5px solid transparent;
  border-top-color:#64D2FF; border-right-color:rgba(100,210,255,0.25); animation:alSpin 1.3s linear infinite; }
.al-orbit.two { inset:5px; border-top-color:transparent; border-bottom-color:rgba(191,90,242,0.7);
  animation-duration:2.1s; animation-direction:reverse; }
@keyframes alSpin { to { transform:rotate(360deg); } }
@keyframes alIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
.al-card { animation:alIn .35s ease both; }
.al-card:nth-child(2) { animation-delay:.05s; } .al-card:nth-child(3) { animation-delay:.1s; }
.al-card:nth-child(4) { animation-delay:.15s; } .al-card:nth-child(5) { animation-delay:.2s; }
.al-tag { font-size:10.5px; letter-spacing:.32em; color:#8E8E93; font-weight:500; margin-top:6px; }
</style>"""

VERDICT_BADGE = {   # known-object verdict -> (plain words, colour)
    "KNOWN PN": ("Already known: planetary nebula", "#FF9F0A"),
    "KNOWN CANDIDATE": ("Already listed as a candidate", "#FFD60A"),
    "PREVIOUSLY REJECTED": ("Listed, but not as a PN", "#8E8E93"),
    "EXTRAGALACTIC PN": ("PN in another galaxy", "#8E8E93"),
    "INSIDE EXTERNAL GALAXY": ("Inside another galaxy", "#8E8E93"),
    "NO MATCH": ("Not in any catalogue checked", "#30D158"),
}


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def logo_html(base):
    """Header: your emblem and wordmark (Aletheia/config/brand, written by the session cell), or your own
    config/logo.png, or a small ring."""
    import base64
    def img(f, h):
        return (f"<img src='data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}' "
                f"style='height:{h}px;width:auto;display:block'>")
    brand = Path(base) / "config" / "brand"
    if (brand / "emblem.png").exists() and (brand / "wordmark.png").exists():
        return (f"<div class='al-head'>{img(brand / 'emblem.png', 52)}<div>{img(brand / 'wordmark.png', 20)}"
                "<div class='al-tag'>NEBULA DISCOVERY SOFTWARE</div></div></div>")
    for name in ("logo.png", "logo.jpg", "logo.jpeg"):
        f = Path(base) / "config" / name
        if f.exists() and f.stat().st_size < 3_000_000:
            mime = "png" if name.endswith("png") else "jpeg"
            return (f"<div class='al-head'><img src='data:image/{mime};base64,{base64.b64encode(f.read_bytes()).decode()}'"
                    " style='height:52px;width:auto;border-radius:10px'><div><div style='font-size:20px;"
                    "font-weight:600;letter-spacing:.24em'>ALETHEIA</div><div class='al-tag'>NEBULA DISCOVERY "
                    "SOFTWARE</div></div></div>")
    return ("<div class='al-head'><svg width='44' height='44' viewBox='0 0 38 38'><circle cx='19' cy='19' r='13' "
            "fill='none' stroke='#64D2FF' stroke-width='3' opacity='.85'/><circle cx='19' cy='19' r='2.2' "
            "fill='#F5F5F7'/></svg><div><div style='font-size:20px;font-weight:600;letter-spacing:.24em'>ALETHEIA"
            "</div><div class='al-tag'>NEBULA DISCOVERY SOFTWARE</div></div></div>")


def progress_html(a, b, step=""):
    """Smooth progress bar: fills from a to b per cent with an eased animation and a moving shine."""
    a, b = max(0.0, min(100.0, a)), max(0.0, min(100.0, b))
    return (f"<div class='al-prog'><div class='al-track'><div class='al-fill' style='--a:{a:.1f}%;--b:{b:.1f}%'>"
            f"</div></div><span class='al-pct'>{b:.0f}%</span></div>")   # the step is already in the status pill


def loader_html(base):
    """Your emblem breathing inside two slow orbits, shown while Aletheia works."""
    import base64
    f = Path(base) / "config" / "brand" / "emblem.png"
    if f.exists():
        mark = f"<img src='data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}'>"
    else:
        mark = ("<svg class='al-mark' viewBox='0 0 38 38'><circle cx='19' cy='19' r='12' fill='none' stroke='#64D2FF' "
                "stroke-width='3'/><circle cx='19' cy='19' r='2.4' fill='#F5F5F7'/></svg>")
    return f"<div class='al-loader'><div class='al-orbit'></div><div class='al-orbit two'></div>{mark}</div>"


def result_cards_html(r, fov, auto, note, card_html, cspn_part):
    """Session 19: the results as cards. Identity first (known or not, exact name and status), then position,
    look-alikes, the evidence scorecard and the central star. The content is the same as before, only grouped."""
    d, k = r["desc"], r["known"]
    title = xm.identify(k, r["official"])[0]
    verdict = str(k.get("verdict", ""))
    words, colour = VERDICT_BADGE.get(verdict, (verdict.title(), "#8E8E93"))
    if verdict.startswith("NOT CHECKED"):
        words, colour = "Not checked: catalogues did not answer (press Analyse again)", "#FF453A"
    lines = summary_lines(r)
    alike = [x for x in lines if x.startswith(("Imposter flags", "Variable stars"))]
    ident = [x for x in lines if x not in alike]
    head = _esc(title) if title else f"Uncatalogued position {d['pn_g']}"
    near = None if title else xm.nearest_known(k)
    if near and not verdict.startswith("NOT CHECKED"):   # Session 19b: outside, but close to a known nebula
        head = f"Uncatalogued point near {_esc(near['name'])} ({_esc(near['cls'])})"
        words, colour = f"Near a known object: {near['name']}", "#FFD60A"
        ident = [f"Nearest known object: {near['text']}"] + ([xm.quote_entry(near["entry"])] if near["entry"] else []) \
            + ident
    cards = [
        f"<div class='al-card wide'><h4>Identity</h4><div class='al-title'>{head}</div>"
        f"<span class='al-badge' style='background:{colour}22;color:{colour};border:1px solid {colour}'>{words}</span>"
        + "".join(f"<div>{_esc(x)}</div>" for x in ident) + "</div>",
        f"<div class='al-card'><h4>Position</h4>{d['pn_g']}<br>RA {d['ra_hms']} &nbsp; Dec {d['dec_dms']}<br>"
        f"l {d['l_deg']:.3f} &nbsp; b {d['b_deg']:+.3f} &nbsp; {d['constellation']}<br>"
        f"<span class='al-muted'>cutout {fov:.1f}' · match zone {r['zone']:.1f}' · search radius "
        f"{r['search']:.1f}'{auto}</span></div>",
        "<div class='al-card'><h4>Look-alikes</h4>" + ("".join(f"<div>{_esc(x)}</div>" for x in alike)
                                                      or "<span class='al-muted'>not checked</span>")
        + (f"<div class='al-muted' style='margin-top:6px;font-size:12px'>{_esc(note)}</div>" if note else "") + "</div>",
        "<div class='al-card wide'><h4>Evidence scorecard</h4>"
        + card_html.replace("<div style='margin-top:12px'><b>Evidence scorecard</b><br>", "<div>") + "</div>",
        "<div class='al-card wide'><h4>Central star</h4>"
        + cspn_part.replace("<div style='margin-top:12px'><b>White dwarf / central star</b> ", "<div class='al-muted'>")
        + "</div>",
    ]
    return "<div class='al-cards'>" + "".join(cards) + "</div>"


def load_ranking(base):
    """Use the frozen ranking model and its tested accuracy if earlier sessions saved them."""
    import json
    global RANK_ACCURACY
    cfg = Path(base) / "config"
    try:
        f = cfg / "rank_model.json"
        cs.RANK_MODEL = json.loads(f.read_text()) if f.exists() else None
        f = cfg / "ranking_accuracy.json"
        RANK_ACCURACY = json.loads(f.read_text()) if f.exists() else None
    except Exception:
        cs.RANK_MODEL, RANK_ACCURACY = None, None


DEFAULT_ON = ("dss2_red", "dss2_col", "ps_col", "iphas_ha", "vphas_ha", "shs_ha", "diff", "allwise_w3", "galex_nuv")


class AletheiaApp:
    """First working interface (Colab), Session 8c. Two views:
    Grid    - every ticked survey side by side (Adjust jumps to Inspect); switched by two buttons
    Inspect - one large image with its own levels, smoothing and muting right beside it,
              a thumbnail strip underneath, and previous/next buttons. No scrolling back and forth."""

    BIG, THUMB = 520, 120

    def __init__(self, base, own=None, registry=None, fetcher=None, gaia_getter=None, workers=8,
                 coverage_lister=None):
        import ipywidgets as W
        self.W, self.base, self.own, self.workers = W, Path(base), own or [], workers
        self.registry = registry or pn.resolve_surveys()
        self.fetcher, self.gaia_getter = fetcher, gaia_getter or gaia_for_field
        self.result = self.panel = self.diffs = self.gaia = self.covered = self.card = None
        self.coverage_lister = coverage_lister
        self.fov, self.items, self.settings, self.cache, self.active = 6.0, {}, {}, {}, None
        self.grid_imgs, self.thumbs, self._last = {}, {}, None
        self._busy = self._batch = False
        import threading
        self._job_lock, self._job_running = threading.Lock(), False
        load_ranking(self.base)
        hl.set_base(self.base)   # Session 18: every problem is written to results/problems.log
        xm.load_extra(self.base / "data" / "planetarynebulae_net.csv")
        lay = W.Layout
        # --- search row
        self.pos = W.Text(placeholder="Position, any format: 01 16 44.66 +60 00 52.7", layout=lay(width="420px"))
        self.size = W.Text(placeholder="object size, arcmin (optional)", layout=lay(width="230px"))
        self.cut = W.Text(placeholder="cutout size, arcmin (blank = auto)", layout=lay(width="230px"))
        self.zone = W.Text(placeholder="match zone override, arcmin", layout=lay(width="230px"))
        self.go = W.Button(description="Analyse field", button_style="info", layout=lay(width="140px"))
        self.bar = W.HTML()                       # Session 21b: smooth bar with percentage (progress_html)
        self.loader = W.HTML()                    # Session 21b: your emblem while Aletheia works
        self._pct, self._loader_html = 0.0, loader_html(self.base)
        self.status = W.HTML()
        self.status.add_class("al-status")
        self.summary = W.HTML()
        # --- your own image (Session 17)
        self.health_btn = W.Button(description="Health check", layout=lay(width="120px"),
                                   tooltip="Test every part of Aletheia on known objects (about 1 minute)")
        self.mine_btn = W.Button(description="Your image", layout=lay(width="140px"),
                                 tooltip="Plate solve your own image, read RA/Dec under the mouse, click to analyse")
        self.mine_files = W.Dropdown(options=[], description="Image", layout=lay(width="420px"))
        self.mine_refresh = W.Button(description="Refresh list", layout=lay(width="110px"))
        self.mine_paths = {}
        self.mine_uploader = W.Output()
        self.mine_auto = W.Checkbox(value=True, indent=False, layout=lay(width="620px"),
                                    description="Search the surveys automatically (image centre, then each click)")
        self.mine_solve = W.Button(description="Plate solve", button_style="info", layout=lay(width="120px"))
        self.mine_hint = W.Text(placeholder="Rough position (optional): object name or RA Dec, e.g. StDr 47",
                                layout=lay(width="420px"),
                                tooltip="With a rough position the image is solved in seconds, here in Colab")
        self.mine_points, self.mine_known, self.mine_sol, self.mine_circle = [], [], None, None
        self.web = False   # Session 22: True in the web edition (no Drive, no Colab; files offered as downloads)
        self.mine_status = W.HTML()
        self.mine_out = W.Output()
        self.mine = W.VBox([W.HTML("<div style='color:#8E8E93;font-family:monospace;font-size:12px;line-height:1.5'>"
                                   "Add an image from your computer below (any size; it is saved in My Drive &gt; "
                                   "Aletheia &gt; uploads), or put it in that Drive folder yourself and press Refresh "
                                   "list. FITS, JPG, PNG, TIFF.</div>"),
                            self.mine_uploader,
                            W.HBox([self.mine_files, self.mine_refresh]), W.HBox([self.mine_hint, self.mine_solve]),
                            self.mine_auto,
                            self.mine_status, self.mine_out])
        self.mine.layout.display = "none"
        # --- inspect view
        self.pick = W.Dropdown(options=[], description="Image", layout=lay(width="330px"))
        self.prev = W.Button(description="◀", layout=lay(width="40px"))
        self.next = W.Button(description="▶", layout=lay(width="40px"))
        self.big = W.Image(value=placeholder_png(self.BIG, "Analyse a field to see images here"), format="png",
                           layout=lay(width=f"{self.BIG}px", height=f"{self.BIG}px"))
        self.big_note = W.HTML()
        self.black = W.FloatSlider(value=2, min=0, max=8, step=0.25, description="Black point",
                                   continuous_update=True, style={"description_width": "80px"}, layout=lay(width="300px"))
        self.white = W.FloatSlider(value=10, min=2, max=60, step=0.5, description="White point",
                                   continuous_update=True, style={"description_width": "80px"}, layout=lay(width="300px"))
        self.smooth = W.Dropdown(options=[("No smoothing", "none"), ("Bin 2x2", "bin2"), ("Bin 4x4", "bin4"),
                                          ("Gaussian 1.5 px", "gauss")], value="none", description="Smoothing",
                                 style={"description_width": "80px"}, layout=lay(width="300px"))
        self.muting = W.Dropdown(options=[("Stars: original", "off"), ("Stars: Gentle mute", "gentle"),
                                          ("Stars: Strong mute", "strong")], value="off", description="Stars",
                                 style={"description_width": "80px"}, layout=lay(width="260px"))
        self.mute_info = W.Button(description="i", tooltip="About star muting", layout=lay(width="32px"))
        self.hist = W.Image(value=placeholder_png(HIST_W, "", HIST_H), format="png",
                            layout=lay(width=f"{HIST_W}px", height=f"{HIST_H}px"))
        self.apply_all = W.Button(description="Apply to all images", layout=lay(width="170px"))
        self.reset = W.Button(description="Reset image", layout=lay(width="120px"))
        self.about = W.HTML(layout=lay(width="300px"))
        self.hires_btn = W.Button(description="High resolution (zoom)", button_style="info",
                                  layout=lay(width="200px"))
        self.hires_out = W.Output()
        controls = W.VBox([self.black, self.white, self.hist, self.smooth, W.HBox([self.muting, self.mute_info]),
                           W.HBox([self.apply_all, self.reset]), self.hires_btn, self.about],
                          layout=lay(padding="0 0 0 12px"))
        self.strip = W.HBox(layout=lay(overflow_x="auto", width="900px", padding="6px 0"))
        self.inspect = W.VBox([W.HBox([self.pick, self.prev, self.next]),
                               W.HBox([W.VBox([self.big, self.big_note]), controls]), self.strip, self.hires_out])
        # --- grid view
        self.grid = W.GridBox(children=[W.HTML("<div style='color:#8E8E93;font-family:monospace;padding:20px'>"
                                               "Type a position above and press Analyse field.</div>")],
                              layout=lay(grid_template_columns="repeat(4, 250px)", grid_gap="10px 10px"))
        # view switch: two plain buttons (Colab does not always draw Tab widgets)
        self.live_btn = W.Button(description="Live sky", button_style="info", layout=lay(width="110px"),
                                 tooltip="Full-resolution surveys streamed from CDS, adjusted live in the browser")
        self.grid_btn = W.Button(description="Grid view", layout=lay(width="110px"))
        self.insp_btn = W.Button(description="Processing", layout=lay(width="110px"),
                                 tooltip="Star muting, binning, H-alpha minus r, high-resolution download")
        self.live_out = W.Output()
        self.live = W.VBox([self.live_out])
        self.inspect.layout.display = "none"
        self.grid.layout.display = "none"
        # --- sidebar
        self.boxes, rows = {}, []
        entries = [(r["key"], r["label"], r["group"]) for r in self.registry] + [("diff", "H-alpha minus r", "H-alpha")]
        for g in pn.GROUPS:
            group_items = [e for e in entries if e[2] == g]
            if not group_items:
                continue
            rows.append(W.HTML(f"<div style='margin-top:8px;color:#8E8E93;font-family:monospace;font-size:11px;"
                               f"letter-spacing:.12em'>{g.upper()}</div>"))
            for key, label, _ in group_items:
                cb = W.Checkbox(value=key in DEFAULT_ON, description=label, indent=False, layout=lay(width="215px"))
                ib = W.Button(description="i", tooltip=f"About {label}", layout=lay(width="30px", min_width="30px"))
                ib.on_click(self._safe(lambda _b, k=key, lab=label: self._show_info(k, lab)))
                cb.observe(self._safe(lambda _c: self._rebuild()), names="value")
                self.boxes[key] = cb
                rows.append(W.HBox([cb, ib], layout=lay(flex_flow="row nowrap", align_items="center")))
        self.show_all = W.Button(description="Show all", layout=lay(width="90px"))
        self.clear = W.Button(description="Clear", layout=lay(width="70px"))
        self.download = W.Button(description="Download cutouts (zip)", layout=lay(width="190px"))
        self.report_btn = W.Button(description="Candidate report (PDF)", layout=lay(width="190px"),
                                   tooltip="Summary, finding chart, survey overview and credits in one PDF")
        self.show_marks = W.Checkbox(value=True, description="Mark central-star candidates", indent=False,
                                     layout=lay(width="230px"))
        self.table_btn = W.Button(description="Show full star table", layout=lay(width="170px"))
        self.table = W.HTML(layout=lay(display="none"))
        self.info = W.HTML()
        self.sidebar = W.VBox([W.HBox([self.show_all, self.clear])] + rows,
                              layout=lay(width="270px", min_width="270px", padding="0 12px 0 0"))
        # --- wiring (every handler is wrapped so an error shows instead of freezing)
        self.go.on_click(self._safe(lambda _b: self._background(self.run, self.go, "Analysing...")))
        self.black.observe(self._safe(lambda _c: self._levels_changed()), names="value")
        self.white.observe(self._safe(lambda _c: self._levels_changed()), names="value")
        self.smooth.observe(self._safe(lambda _c: self._processing_changed()), names="value")
        self.muting.observe(self._safe(lambda _c: self._processing_changed()), names="value")
        self.pick.observe(self._safe(lambda _c: self._picked()), names="value")
        self.prev.on_click(self._safe(lambda _b: self._step(-1)))
        self.next.on_click(self._safe(lambda _b: self._step(1)))
        self.apply_all.on_click(self._safe(lambda _b: self._apply_all()))
        self.reset.on_click(self._safe(lambda _b: self._reset()))
        self.show_all.on_click(self._safe(lambda _b: self._set_all(True)))
        self.clear.on_click(self._safe(lambda _b: self._set_all(False)))
        self.mute_info.on_click(self._safe(lambda _b: self._set_info("Star muting", MUTING_INFO)))
        self.download.on_click(self._safe(lambda _b: self._download()))
        self.report_btn.on_click(self._safe(lambda _b: self._report()))
        self.hires_btn.on_click(self._safe(lambda _b: self._background(self._hires, self.hires_btn,
                                                                         "Downloading...")))
        self.live_btn.on_click(self._safe(lambda _b: self._view("live")))
        self.grid_btn.on_click(self._safe(lambda _b: self._view("grid")))
        self.insp_btn.on_click(self._safe(lambda _b: self._view("inspect")))
        # --- Session 19 layout: header, two modes (Search a position | Your image), one status line, results
        header = W.HTML(logo_html(self.base))
        self.mode_search = W.Button(description="Search a position", layout=lay(width="170px"),
                                    tooltip="Type RA/Dec in any format")
        self.mode_search.add_class("al-active")
        self.mine_btn.description, self.mine_btn.layout.width = "Your image", "170px"
        self.more_btn = W.Button(description="More options", layout=lay(width="130px"),
                                 tooltip="Cutout size and match zone (normally automatic)")
        self.options = W.HBox([self.cut, self.zone])
        self.options.layout.display = "none"
        self.mode_bar = W.HBox([self.mode_search, self.mine_btn])
        self.mode_bar.add_class("al-seg")
        self.search_panel = W.VBox([W.HBox([self.pos, self.size, self.go]), W.HBox([self.more_btn]), self.options])
        self.search_panel.add_class("al-panel")
        self.mine.add_class("al-panel")
        top = W.VBox([W.HBox([header, self.health_btn], layout=lay(justify_content="space-between",
                                                                   align_items="center", margin="0 0 10px 0")),
                      self.mode_bar, self.search_panel, self.mine,
                      W.HBox([self.loader, self.status, self.bar], layout=lay(align_items='center', min_height='50px'))])
        self.mode_search.on_click(self._safe(lambda _b: self._mode("search")))
        self.mine_btn.on_click(self._safe(lambda _b: self._mode("image")))
        self.more_btn.on_click(self._safe(lambda _b: self._toggle_options()))
        self.health_btn.on_click(self._safe(lambda _b: self._background(self._health, self.health_btn, "Checking...")))
        self.mine_refresh.on_click(self._safe(lambda _b: self._mine_list()))
        self.mine_solve.on_click(self._safe(lambda _b: self._mine_run()))
        self.callbacks = {"aletheia.pick": self._logged(self._mine_picked, "your image: click"),
                          "aletheia.chunk": self._logged(self._mine_chunk, "your image: upload"),
                          "aletheia.circle": self._logged(self._mine_circle, "your image: circle")}
        try:   # a click on your solved image sends its position back here (Colab; the web edition uses a bridge)
            from google.colab import output as _colab_output
            _colab_output.register_callback("aletheia.pick", self._logged(self._mine_picked, "your image: click"))
            _colab_output.register_callback("aletheia.chunk", self._logged(self._mine_chunk, "your image: upload"))
            _colab_output.register_callback("aletheia.circle", self._logged(self._mine_circle, "your image: circle"))
        except Exception:
            pass
        self.show_marks.observe(self._safe(lambda _c: self._rebuild()), names="value")
        self.table_btn.on_click(self._safe(lambda _b: self._toggle_table()))
        self.surveys_btn = W.Button(description="Surveys", layout=lay(width="130px"),
                                    tooltip="Choose which surveys to show (the usual ones are already ticked)")
        self.surveys_btn.on_click(self._safe(lambda _b: self._toggle_surveys()))
        self.sidebar.layout.display = "none"
        self.sidebar.add_class("al-panel")
        self.main = W.VBox([self.summary,
                            W.HBox([self.live_btn, self.grid_btn, self.insp_btn, self.surveys_btn],
                                   layout=lay(flex_flow="row wrap")),
                            W.HBox([self.report_btn, self.download, self.table_btn, self.show_marks],
                                   layout=lay(flex_flow="row wrap", align_items="center")),
                            self.table, self.info, self.live, self.grid, self.inspect],
                           layout=lay(flex="1 1 auto"))
        self.ui = W.VBox([W.HTML(APP_CSS), top, W.HBox([self.sidebar, self.main])])
        self.ui.add_class("aletheia")
        self._surveys_label()
        self.summary.value = ("<div class='al-cards'><div class='al-card wide al-muted'>Type a position above (any format) "
                              "and press Analyse field, or choose Your image to plate solve one of your own pictures. "
                              "Results appear here as cards.</div></div>")

    # ----------------------------------------------------------------- helpers
    def _safe(self, fn):
        def wrapped(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # show the problem instead of silently freezing
                self._batch = False
                hl.log_problem("interface", exc)
                self.status.value = f"<span style='color:#FF9F0A'>Problem: {type(exc).__name__}: " \
                                    f"{' '.join(str(exc).split())[:140]}</span>"
        return wrapped

    def _set_progress(self, pct, step=""):
        pct = max(self._pct, float(pct))           # never runs backwards
        self.bar.value = progress_html(self._pct, pct, step)
        self._pct = pct

    def _health_step(self, message):
        self.status.value = "&nbsp;" + message
        try:   # "health check 7/20: ..." -> 35%
            i, n = message.split(":")[0].split()[-1].split("/")
            self._set_progress(100 * (int(i) - 1) / int(n), message.split(":", 1)[1].strip(" ."))
        except Exception:
            pass

    def _working(self, on):
        """Show or hide the breathing emblem and the progress bar."""
        self.loader.value = self._loader_html if on else ""
        if on:
            self._pct = 0.0
        else:
            self.bar.value = ""

    def _mode(self, which):
        """Search a position, or work on your own image: one panel at a time (Session 19)."""
        image = which == "image"
        self.search_panel.layout.display = "none" if image else "flex"
        if image and self.mine.layout.display == "none":
            self._mine_toggle()            # shows the panel, the uploader and the file list
        elif not image:
            self.mine.layout.display = "none"
        for b, on in ((self.mine_btn, image), (self.mode_search, not image)):
            (b.add_class if on else b.remove_class)("al-active")

    def _toggle_options(self):
        show = self.options.layout.display == "none"
        self.options.layout.display = "flex" if show else "none"
        self.more_btn.description = "Fewer options" if show else "More options"

    def _surveys_label(self):
        n = sum(1 for cb in self.boxes.values() if cb.value)
        shown = self.sidebar.layout.display != "none"
        self.surveys_btn.description = f"Hide surveys" if shown else f"Surveys ({n} on)"

    def _toggle_surveys(self):
        self.sidebar.layout.display = "flex" if self.sidebar.layout.display == "none" else "none"
        self._surveys_label()

    @staticmethod
    def _logged(fn, where):
        """For calls coming from the browser: the problem is logged, then passed on to the page."""
        def wrapped(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:
                hl.log_problem(where, exc)
                raise
        return wrapped

    def _health(self):
        """Session 18: the built-in health check, results shown under the buttons and saved."""
        import sys
        if self._busy:
            self.status.value = "&nbsp;Busy with another job: please wait for it to finish"
            return
        self._busy = True
        self._working(True)
        try:
            mods = [m for m in sys.modules if m.startswith("aletheia_")]
            res, summ = hl.run_all(self.base, self.own, modules=mods,
                                   progress=self._health_step)
            path = hl.save(self.base, res, summ)
            self.info.value = hl.report_html(res, summ) + \
                f"<div style='font-family:monospace;font-size:11px;color:#6E6E73'>saved: Aletheia/results/health/{path.name}</div>"
            self.status.value = f"&nbsp;Health check: {summ}"
        finally:
            self._working(False)
            self._busy = False

    def show(self):
        from IPython.display import display
        display(self.ui)

    def _set_info(self, title, text):
        self.info.value = (f"<div style='border-left:2px solid #64D2FF;padding:4px 10px;margin:6px 0;"
                           f"font-size:13px'><b>{title}</b><br>{text}</div>")

    def _show_info(self, key, label):
        short, more = SURVEY_INFO.get(key, ("", ""))
        self._set_info(label, f"{short}<br><span style='color:#6E6E73'>{more}</span>")

    @staticmethod
    def _num(text):
        try:
            v = float(str(text).strip())
            return v if v > 0 else None
        except ValueError:
            return None

    def _progress(self, done, total, label):
        self._set_progress(10 + 70 * done / max(1, total), f"{label} ({done}/{total})")
        self.status.value = f"&nbsp;Downloading surveys {done}/{total}: {label}"

    # ----------------------------------------------------------------- analysis
    def run(self):
        import time
        from concurrent.futures import ThreadPoolExecutor
        if self._busy:
            return
        size, cut, zone = self._num(self.size.value), self._num(self.cut.value), self._num(self.zone.value)
        try:
            coord = ac.parse_position(self.pos.value)
        except Exception as exc:
            self.status.value = f"<span style='color:#FF9F0A'>Could not read that position: {exc}</span>"
            return
        self._busy, t0 = True, time.time()
        try:
            self.fov = cut or pn.field_of_view(size)
            self._working(True)
            self._set_progress(3, "starting")
            self.status.value = "&nbsp;Checking catalogues and downloading surveys at the same time..."
            self.cache, self.settings, self.active, self.gaia, self._last = {}, {}, None, None, None
            self.status.value = "&nbsp;Asking CDS which surveys cover this position..."
            self.covered = pn.covering_ids(coord, self.fov / 60.0 / 2, self.coverage_lister)
            self._set_progress(10, "survey footprints checked")
            self.panel = {}
            wanted = self._wanted_keys()
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(analyse, self.pos.value, size, self.own, zone)
                todo = [x for x in self.registry if x["key"] in wanted]
                self.panel.update(pn.fetch_panel_parallel(coord, self.fov, todo, 400, self.fetcher,
                                                          self.workers, self._progress))
                self.status.value = "&nbsp;Images done; finishing catalogue checks..."
                self._set_progress(84, "catalogue checks")
                self.result = fa.result()
            self._set_progress(92, "evidence scorecard")
            self.diffs = pn.difference_images(self.panel)
            try:
                self.card = sc.build_scorecard(self.result, self.panel, self.diffs, self.fov)
                card_html = sc.scorecard_html(self.card)
            except Exception as exc:  # the scorecard must never block the rest of the screen
                hl.log_problem("scorecard", exc)
                self.card, card_html = None, f"<br><span style='color:#FF9F0A'>Scorecard failed: {exc}</span>"
            note = self._coverage_text()
            self._sync_boxes()
            r = self.result
            auto = " (auto from object size)" if not zone else " (your override)"
            self.summary.value = result_cards_html(r, self.fov, auto, note, card_html, cspn_html(r))
            self.table.value = cspn_table_html(r)
            self._set_progress(100, "done")
            self.status.value = f"&nbsp;Done in {time.time() - t0:.0f} s."
            self._live_ready = False
            self._rebuild()
            self._view("live")
        finally:
            self._working(False)
            self._busy = False

    # ----------------------------------------------------------------- images
    PAIRS = {"diff": ("iphas_ha", "iphas_r", "vphas_ha", "vphas_r", "shs_ha", "shs_sr")}

    def _covered(self, key):
        x = next((r for r in self.registry if r["key"] == key), None)
        if x is None or not x["hips_id"]:
            return False
        return self.covered is None or x["hips_id"] in self.covered

    def _wanted_keys(self):
        keys = set()
        for k, cb in self.boxes.items():
            if not cb.value:
                continue
            keys |= set(self.PAIRS.get(k, (k,)))
        return {k for k in keys if self._covered(k)}

    def _sync_boxes(self):
        """Grey out surveys with no data here; keep ticks only where an image exists or can be fetched."""
        self._batch = True
        for k, cb in self.boxes.items():
            if k == "diff":
                ok = bool(self.diffs) or any(self._covered(x) for x in self.PAIRS["diff"][::2])
            elif k in self.panel:
                ok = self.panel[k]["status"] == "ok"
            else:
                ok = self._covered(k)
            cb.disabled = not ok
            if not ok and cb.value:
                cb.value = False
        self._batch = False

    def _coverage_text(self):
        names = {r["key"]: r["label"] for r in self.registry}
        have = [names[k] for k in names if self._covered(k) and not (k in self.panel and self.panel[k]["status"] != "ok")]
        missing = [names[k] for k in names if names[k] not in have]
        src = "CDS survey footprints" if self.covered is not None else "downloads so far"
        return (f"This field is covered by {len(have)} of {len(names)} surveys ({src}). "
                + (f"Not covered here: {', '.join(missing)}." if missing else ""))

    def _fetch_missing(self):
        """A survey was ticked after the analysis: download just what is now needed."""
        need = [k for k in self._wanted_keys() if k not in self.panel]
        if not need or self.result is None:
            return
        self.status.value = f"&nbsp;Downloading {len(need)} more survey(s)..."
        todo = [x for x in self.registry if x["key"] in need]
        self.panel.update(pn.fetch_panel_parallel(self.result["coord"], self.fov, todo, 400, self.fetcher,
                                                  self.workers, self._progress))
        self.diffs = pn.difference_images(self.panel)
        self.status.value = "&nbsp;Done."

    def _collect(self):
        out = {}
        for k, cb in self.boxes.items():
            if not cb.value or cb.disabled:
                continue
            if k != "diff" and k not in self.panel:
                continue
            if k == "diff":
                for name, dd in self.diffs.items():
                    out[f"diff:{name}"] = dict(dd, label=dd["label"], is_diff=True, group="H-alpha", kind="grey",
                                               info_key="diff")
            else:
                out[k] = dict(self.panel[k], is_diff=False, info_key=k)
        return out

    def _st(self, key):
        return self.settings.setdefault(key, dict(black=2.0, white=10.0, smooth="none", mute="off"))

    def _prepared(self, key):
        """Slow part, cached per (image, smoothing, muting): processed image, z image, histogram base."""
        st, item = self._st(key), self.items[key]
        ck = (key, st["smooth"], st["mute"])
        if ck not in self.cache:
            if st["mute"] == "strong" and self.gaia is None:
                self.status.value = "&nbsp;Fetching Gaia DR3 stars for Strong muting..."
                self.gaia = self.gaia_getter(self.result["coord"], self.fov)
                self.status.value = ""
            wcs = pn.grid_wcs(self.result["coord"], self.fov, 400)
            prot = pn.protect_radius_px(self.result["size"], self.fov, 400)
            work, note = prepare(item, st["smooth"], st["mute"], self.gaia, wcs, prot)
            z = normalise(work)
            self.cache[ck] = dict(work=work, z=z, note=note, hist=histogram_base(z))
        return self.cache[ck]

    def _png(self, key, size, highlight=False):
        st, item = self._st(key), self.items[key]
        c = self._prepared(key)
        img = c["work"] if c["z"] is None else levels_z(c["z"], st["black"], st["white"], item.get("is_diff"))
        ring = (self.result["size"] / self.fov) if self.result["size"] else None
        marks = cspn_marks(self.result, self.fov) if self.show_marks.value else None
        return to_png(img, ring, size=size, highlight=highlight, marks=marks)

    def _rebuild(self):
        """Ticked surveys changed (or a new field): rebuild grid, strip and image menu."""
        if self.panel is None or self._batch:
            return
        self._fetch_missing()
        self._sync_boxes()
        W = self.W
        self.items = self._collect()
        self.grid_imgs, self.thumbs, cells, strip = {}, {}, [], []
        for key, item in self.items.items():
            img = W.Image(value=self._png(key, 240), format="png", layout=W.Layout(width="240px", height="240px"))
            hint = self._prepared(key)["note"] or ("faint here: try Bin 2x2/4x4" if not item.get("is_diff")
                                                   and faint_hint(item, self.result["size"], self.fov) else "")
            adj = W.Button(description="Adjust", layout=W.Layout(width="80px"))
            adj.on_click(self._safe(lambda _b, k=key: self._open(k)))
            cells.append(W.VBox([W.HTML(f"<div style='font-family:monospace;font-size:12px'>{item['label']}</div>"),
                                 img, W.HBox([adj, W.HTML(f"<div style='color:#FF9F0A;font-family:monospace;"
                                                          f"font-size:10px'>{hint}</div>")])]))
            self.grid_imgs[key] = img
            th = W.Image(value=self._png(key, self.THUMB), format="png",
                         layout=W.Layout(width=f"{self.THUMB}px", height=f"{self.THUMB}px"))
            tb = W.Button(description=item["label"][:16], tooltip=item["label"],
                          layout=W.Layout(width=f"{self.THUMB}px"))
            tb.on_click(self._safe(lambda _b, k=key: self._activate(k)))
            self.thumbs[key] = th
            strip.append(W.VBox([th, tb], layout=W.Layout(min_width=f"{self.THUMB + 6}px")))
        self.grid.children = cells or [W.HTML("Tick a survey on the left, or press Show all.")]
        self._live_ready = False  # ticks changed: Live sky must be redrawn when next shown
        if getattr(self, "current_view", None) == "live" and self.result is not None:
            self._render_live()
        self.strip.children = strip
        self._batch = True
        self.pick.options = [(it["label"], k) for k, it in self.items.items()]
        self._batch = False
        if self.items:
            self._activate(self.active if self.active in self.items else next(iter(self.items)))
        else:
            self.active = None

    def _background(self, job, button, busy_label):
        """Run a slow job while the cell is active (Colab only shows updates made this way).
        The button is greyed out at the moment of the click, so repeated clicks cannot pile up."""
        with self._job_lock:
            if self._job_running or self._busy:
                self.status.value = "&nbsp;Still working on the previous request..."
                return None
            self._job_running = True
        label = button.description
        button.disabled, button.description = True, busy_label
        try:
            self._safe(job)()
        finally:
            button.disabled, button.description = False, label
            self._job_running = False
        return True

    def _toggle_table(self):
        show = self.table.layout.display == "none"
        self.table.layout.display = "block" if show else "none"
        self.table_btn.description = "Hide full star table" if show else "Show full star table"

    def _view(self, which):
        self.current_view = which
        self.live.layout.display = "flex" if which == "live" else "none"
        self.grid.layout.display = "grid" if which == "grid" else "none"
        self.inspect.layout.display = "flex" if which == "inspect" else "none"
        for b, w in ((self.live_btn, "live"), (self.grid_btn, "grid"), (self.insp_btn, "inspect")):
            b.button_style = "info" if which == w else ""
        if which == "live" and self.result is not None and not getattr(self, "_live_ready", False):
            self._render_live()

    def _live_layers(self):
        layers = []
        for k, cb in self.boxes.items():
            if k == "diff" or not cb.value or cb.disabled:
                continue
            reg = next((r for r in self.registry if r["key"] == k), None)
            if not reg or not reg["hips_id"]:
                continue
            p = self.panel.get(k) or {}
            stats = survey_stats(p) if p.get("data") is not None and reg["kind"] == "grey" else None
            layers.append(dict(key=k, id=reg["hips_id"], label=reg["label"], fits=reg["kind"] == "grey",
                               colour=reg["kind"] == "colour", stats=stats))
        return layers

    def _render_live(self, start_key=None):
        """Streamed full-resolution view of the ticked surveys; all adjustments run in the browser."""
        import time
        from IPython.display import HTML, display
        layers = self._live_layers()
        self.live_out.clear_output(wait=True)
        with self.live_out:
            if not layers:
                print("Tick at least one survey on the left (H-alpha minus r is in the Processing view).")
                return
            start = next((i for i, L in enumerate(layers) if L["key"] == start_key), 0)
            stars = (self.result.get("cspn") or {}).get("stars", [])[:5] if self.show_marks.value else []
            marks = [dict(ra=float(s["coord"].ra.deg), dec=float(s["coord"].dec.deg), n=str(s["rank"])) for s in stars]
            uid = abs(hash((self.result["desc"]["pn_g"], time.time()))) % 10 ** 8
            display(HTML(live_viewer_html(uid, self.result["coord"], self.fov,
                                          [{k: v for k, v in L.items() if k != "key"} for L in layers], start,
                                          self.result["size"], marks)))
        self._live_ready = True

    def _open(self, key):
        """Adjust under a grid image: streamed surveys open in Live sky; H-alpha minus r in Processing."""
        if not key.startswith("diff:") and any(L["key"] == key for L in self._live_layers()):
            self._render_live(key)
            self._view("live")
        else:
            self._activate(key)
            self._view("inspect")

    def _activate(self, key):
        prev, self.active = self.active, key
        st = self._st(key)
        self._batch = True
        self.pick.value = key
        self.black.value, self.white.value = st["black"], st["white"]
        self.smooth.value, self.muting.value = st["smooth"], st["mute"]
        self._batch = False
        self._last = None
        self._draw_active(full=True)
        if prev in self.thumbs and prev != key:
            self.thumbs[prev].value = self._png(prev, self.THUMB)
        short, _ = SURVEY_INFO.get(self.items[key].get("info_key"), ("", ""))
        self.about.value = f"<div style='color:#8E8E93;font-size:12px'>{short}</div>"

    def _draw_active(self, full=False):
        key = self.active
        if key not in self.items:
            return
        st = self._st(key)
        state = (key, st["black"], st["white"], st["smooth"], st["mute"])
        if state == self._last and not full:
            return  # queued slider events with the same value: skip the work
        self._last = state
        self.big.value = self._png(key, self.BIG, highlight=False)
        c = self._prepared(key)
        self.hist.value = histogram_png_fast(c["hist"], st["black"], st["white"])
        self.big_note.value = f"<div style='color:#FF9F0A;font-family:monospace;font-size:11px'>{c['note']}</div>"
        if full:
            self.thumbs[key].value = self._png(key, self.THUMB, highlight=True)
            self.grid_imgs[key].value = self._png(key, 240)

    def _levels_changed(self):
        if self._batch or self.active is None:
            return
        st = self._st(self.active)
        st["black"], st["white"] = self.black.value, self.white.value
        self._draw_active()

    def _processing_changed(self):
        if self._batch or self.active is None:
            return
        st = self._st(self.active)
        st["smooth"], st["mute"] = self.smooth.value, self.muting.value
        self._draw_active(full=True)

    def _picked(self):
        if not self._batch and self.pick.value and self.pick.value != self.active:
            self._activate(self.pick.value)

    def _step(self, d):
        keys = list(self.items)
        if keys and self.active in keys:
            self._activate(keys[(keys.index(self.active) + d) % len(keys)])

    def _hires(self):
        """Fetch the selected image near its native resolution, apply its settings, open the zoom viewer."""
        import time
        from astropy.io import fits
        key = self.active
        if key is None or self._busy:
            return
        self._busy = True
        try:
            self._hires_inner(key)
        finally:
            self._busy = False

    def _hires_inner(self, key):
        import time
        from astropy.io import fits
        item, st, r = self.items[key], self._st(key), self.result
        t0 = time.time()
        if key.startswith("diff:"):
            name = key.split(":", 1)[1]
            ha_key, r_key = {"IPHAS": ("iphas_ha", "iphas_r"), "VPHAS+": ("vphas_ha", "vphas_r"),
                             "SHS": ("shs_ha", "shs_sr")}[name]
            npx = hires_pixels(ha_key, self.fov)
        else:
            ha_key = r_key = None
            npx = hires_pixels(key, self.fov)
        self.status.value = f"&nbsp;Downloading {item['label']} at {npx} x {npx} pixels " \
                            f"({self.fov * 60 / npx:.2f}\"/px)..."
        wanted = [x for x in self.registry if x["key"] in ({ha_key, r_key} if ha_key else {key})]
        hp = pn.fetch_panel_parallel(r["coord"], self.fov, wanted, npx, self.fetcher, workers=2)
        if ha_key:
            dd = pn.difference_images(hp).get(name)
            if dd is None:
                self.status.value = "&nbsp;High-resolution H-alpha minus r not available here."
                return
            hi = dict(dd, label=dd["label"], is_diff=True, group="H-alpha", kind="grey")
        else:
            hi = dict(hp[key], is_diff=False)
            if hi["status"] != "ok":
                self.status.value = f"&nbsp;High-resolution {item['label']}: {hi['status']}"
                return
        wcs = pn.grid_wcs(r["coord"], self.fov, npx)
        if st["mute"] == "strong" and self.gaia is None:
            self.gaia = self.gaia_getter(r["coord"], self.fov)
        work, note = prepare(hi, st["smooth"], st["mute"], self.gaia, wcs,
                             pn.protect_radius_px(r["size"], self.fov, npx))
        z = normalise(work)
        img = work if z is None else levels_z(z, st["black"], st["white"], hi.get("is_diff"))
        ring = (r["size"] / self.fov) if r["size"] else None
        marks = cspn_marks(r, self.fov) if self.show_marks.value else None
        jpg = to_jpeg(img, ring, marks)
        folder = self.base / "results" / "highres"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{r['desc']['pn_g'].replace(' ', '')}_{item['label'].replace(' ', '_').replace('+', 'p')}_{npx}px"
        (folder / f"{stem}.jpg").write_bytes(jpg)
        if hi["data"].ndim == 2:
            hdu = fits.PrimaryHDU(np.asarray(hi["data"], dtype=np.float32), header=wcs.to_header())
            hdu.header["ALETHOBJ"] = (r["desc"]["pn_g"], "Aletheia field")
            hdu.header["SURVEY"] = (item["label"][:68], "Survey")
            hdu.writeto(folder / f"{stem}.fits", overwrite=True)
        from IPython.display import HTML, display
        uid = abs(hash((stem, time.time()))) % 10 ** 8
        self.hires_out.clear_output(wait=True)
        with self.hires_out:
            display(HTML(zoom_viewer_html(jpg, wcs, uid, f"{item['label']} · {npx} x {npx} px · "
                                                        f"{self.fov * 60 / npx:.2f}\"/pixel" +
                                          (f" · {note}" if note else ""))))
        ext = "fits and .jpg" if hi["data"].ndim == 2 else "jpg (colour surveys have no FITS)"
        self.status.value = f"&nbsp;High resolution ready in {time.time() - t0:.0f} s; saved to " \
                            f"Aletheia/results/highres/{stem}.{ext}"

    def _apply_all(self):
        if self.active is None:
            return
        src = dict(self._st(self.active))
        for k in self.items:
            self.settings[k] = dict(src)
            self.grid_imgs[k].value = self._png(k, 240)
            self.thumbs[k].value = self._png(k, self.THUMB, highlight=(k == self.active))

    def _reset(self):
        if self.active is None:
            return
        self.settings.pop(self.active, None)
        self._activate(self.active)

    def _set_all(self, value):
        self._batch = True
        for cb in self.boxes.values():
            if not cb.disabled:
                cb.value = value
        self._batch = False
        self._rebuild()

    # ----------------------------------------------------------------- download
    def _download(self):
        if self.panel is None:
            return
        r = self.result
        name = r["desc"]["pn_g"].replace(" ", "")
        folder = self.base / "results" / "cutouts" / f"{name}_{self.fov:.0f}arcmin"
        folder.mkdir(parents=True, exist_ok=True)
        wcs = pn.grid_wcs(r["coord"], self.fov, 400)
        title = f"{r['desc']['pn_g']} · {self.fov:.1f}' field"
        self.status.value = "&nbsp;Writing annotated cutouts..."
        for p in self.panel.values():
            if p["status"] != "ok":
                continue
            stem = p["label"].replace(" ", "_").replace("+", "p")
            pn.make_annotated_png(p, wcs, folder / f"{stem}.png", f"{p['label']} · {title}", r["size"])
            if p["kind"] == "grey":
                pn.save_grey_fits(p, wcs, folder / f"{stem}.fits", name)
        for k, d in self.diffs.items():
            pn.make_annotated_png(dict(d, fov=self.fov), wcs, folder / f"{k}_Halpha_minus_r.png",
                                  f"{d['label']} · {title}", r["size"], is_diff=True)
        cands = [cs.describe_star(st) for st in (r.get("cspn") or {}).get("stars", [])[:10]]
        (folder / "00_summary.txt").write_text("\n".join(summary_lines(r) + sc.scorecard_text(self.card) + ["", "Central star candidates:"] + cands)
                                               + "\n", encoding="utf-8")
        z = shutil.make_archive(str(folder), "zip", folder)
        self.status.value = f"&nbsp;Saved {Path(z).name} in Aletheia/results/cutouts"
        if self.web:
            self._offer(z, "application/zip")
            return
        try:
            from google.colab import files
            files.download(z)
        except Exception:
            pass


    # ----------------------------------------------------------------- your own image (Session 17)
    def _uploads(self):
        d = self.base / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _mine_toggle(self):
        show = self.mine.layout.display == "none"
        self.mine.layout.display = "flex" if show else "none"
        if show:
            if not getattr(self, "_uploader_shown", False):
                self._uploader_shown = True
                with self.mine_uploader:
                    from IPython.display import HTML, display
                    display(HTML(up.uploader_html(where="a private folder on this server, deleted after a day")
                                 if self.web else up.uploader_html()))
            self._mine_list()

    def _mine_list(self):
        """Images in Aletheia/uploads, and also loose ones in the Aletheia folder or at the top of My Drive."""
        places = [("uploads", self._uploads())] + ([] if self.web else [("Aletheia", self.base), ("My Drive", self.base.parent)])
        self.mine_paths = {}
        for label, folder in places:
            try:
                for p in sorted(folder.iterdir()):
                    if p.is_file() and p.suffix.lower() in up.IMAGE_TYPES:
                        self.mine_paths.setdefault(p.name if label == "uploads" else f"{label}/{p.name}", p)
            except OSError:
                continue
        self.mine_files.options = list(self.mine_paths)
        if self.mine_paths and self.mine_files.value not in self.mine_paths:
            self.mine_files.value = next(iter(self.mine_paths))   # never leave a list shown but nothing chosen
        self.mine_status.value = (f"&nbsp;{len(self.mine_paths)} image(s) found" if self.mine_paths else
                                  ("&nbsp;No images yet: drop one on the box above." if self.web else
                                   f"&nbsp;No images found in {self._uploads()} (or in Aletheia / My Drive). Drive can "
                                   "take a minute to show new files: press Refresh list again."))

    def _mine_chunk(self, name, i, n, b64):
        """Receives one piece of a file sent by the uploader (pieces arrive in order) and saves it to uploads."""
        import base64
        name = Path(str(name)).name
        if Path(name).suffix.lower() not in up.IMAGE_TYPES:
            self.mine_status.value = f"<span style='color:#FF9F0A'>&nbsp;{name}: not an image type Aletheia reads</span>"
            return "refused"
        part = self._uploads() / (name + ".part")
        with open(part, "wb" if int(i) == 0 else "ab") as fh:
            fh.write(base64.b64decode(b64))
        if int(i) == int(n) - 1:
            part.replace(self._uploads() / name)
            self._mine_list()
            self.mine_files.value = name
            self.mine_status.value = f"&nbsp;Saved {name} in Aletheia/uploads: press Plate solve"
        return "ok"

    def _api_key(self):
        import os
        env = self.base / "config" / "aletheia.env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("ASTROMETRY_NET_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return os.environ.get("ASTROMETRY_NET_API_KEY") or None   # web edition: a server secret

    def _offer(self, path, mime):
        """Web edition: the file appears as a download button (and starts downloading by itself)."""
        import base64
        p = Path(path)
        data = base64.b64encode(p.read_bytes()).decode()
        uid = abs(hash((str(p), p.stat().st_mtime))) % 10 ** 8
        self.info.value = (f"<div style='margin:10px 0'><a id='dl{uid}' download='{p.name}' "
                           f"href='data:{mime};base64,{data}' style='display:inline-block;padding:8px 16px;"
                           "border-radius:999px;background:#0A84FF;color:#fff;text-decoration:none;font-weight:500'>"
                           f"Download {p.name}</a> <span class='al-muted' style='margin-left:8px'>"
                           f"{p.stat().st_size / 1e6:.1f} MB</span></div>")

    def _mine_run(self):
        if self._busy:
            self.mine_status.value = "&nbsp;Busy with another job: please wait for it to finish"
            return
        if not self.mine_files.value:
            self.mine_status.value = "<span style='color:#FF9F0A'>&nbsp;Choose an image first. The list is empty: put " \
                                     "your image in Aletheia/uploads in your Drive and press Refresh list.</span>"
            return
        import time
        self._busy, self.mine_solve.disabled = True, True
        self._working(True)
        try:
            def say(m):
                self.mine_status.value = "&nbsp;" + m
            hint = self._hint_coord(say)
            sol = up.plate_solve(self.mine_paths.get(self.mine_files.value, self._uploads() / self.mine_files.value),
                                 self._api_key(), say, cache_dir=self.base / "results" / "solved", hint=hint,
                                 scales_file=self.base / "config" / "plate_scales.json")
            up.save_solution(sol, self.base / "results" / "solved")
            self.mine_sol, self.mine_points, self.mine_known, self.mine_circle = sol, [], [], None
            say(up.summary(sol))
            self.mine_out.clear_output()
            try:   # catalogued nebulae inside the image, drawn as labelled markers
                known = up.known_in_field(sol, self.own, xm.EXTRA)
                self.mine_known = known
                if known:   # Session 19b: name them in words too
                    names = "; ".join(f"{o['name']} ({o['kind']})" for o in known[:8])
                    self.mine_status.value += (f"<br>&nbsp;<span style='color:#FF9F0A'>Catalogued in your image: {names}"
                                               f"{' ...' if len(known) > 8 else ''}</span>")
            except Exception as exc:
                hl.log_problem("your image: catalogued objects", exc)
                known = []
                self.mine_status.value += f"<br>&nbsp;(catalogued objects could not be looked up: {' '.join(str(exc).split())[:80]})"
        except Exception as exc:   # shown in this panel, never a silent stop
            hl.log_problem("your image: plate solve", exc)
            self.mine_status.value = f"<span style='color:#FF9F0A'>&nbsp;Could not solve this image: " \
                                     f"{' '.join(str(exc).split())[:160]}</span>"
            return
        finally:
            self._working(False)
            self._busy, self.mine_solve.disabled = False, False
        try:
            with self.mine_out:
                from IPython.display import HTML, display
                display(HTML(up.viewer_html(sol, abs(hash((sol["path"], time.time()))) % 10 ** 8, known=known)))
        except Exception as exc:
            hl.log_problem("your image: viewer", exc)
            self.mine_status.value += f"<br><span style='color:#FF9F0A'>&nbsp;Solved, but the viewer failed: {exc}</span>"
        if self.mine_auto.value:   # whole image: centre, cutout as large as the image (at most 60')
            c = sol["centre"]
            self._analyse_at(c.ra.deg, c.dec.deg, min(60.0, max(sol["width_arcmin"], sol["height_arcmin"])),
                             "your image's centre")

    def _hint_coord(self, say):
        """Rough position typed by you: RA/Dec in any format, a name from the planetarynebulae.net list, or any name
        SIMBAD knows. Returns a SkyCoord or None (then the solve simply goes without it)."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        text = (self.mine_hint.value or "").strip()
        if not text:
            return None
        try:
            return ac.parse_position(text)
        except Exception:
            pass
        key = "".join(text.split()).lower()
        hit = next((e for e in xm.EXTRA if "".join(e["name"].split()).lower() == key), None)
        if hit:
            return hit["coord"]
        try:
            from astroquery.simbad import Simbad
            t = Simbad.query_object(text)
            if t is not None and len(t):
                cols = {c.lower(): c for c in t.colnames}
                return SkyCoord(float(t[cols["ra"]][0]) * u.deg, float(t[cols["dec"]][0]) * u.deg)
        except Exception as exc:
            hl.log_problem("rough position lookup", exc)
        say(f"'{text}' was not recognised as a position or a name; solving without it")
        return None

    def _mine_circle(self, ra, dec, radius_arcmin):
        """Session 21: the circle you drew on your image. Its centre is analysed with the circle's diameter as the
        object size, and the report rings exactly your circle."""
        ra, dec, r = float(ra), float(dec), float(radius_arcmin)
        self.mine_circle = (ra, dec, r)
        self.mine_points = (self.mine_points + [(ra, dec)])[-20:]
        self.size.value = f"{2 * r:.2f}"
        self._analyse_at(ra, dec, None, f"your circle ({2 * r:.2f}' across)", keep_size=True)

    def _mine_picked(self, ra, dec):
        self.mine_points = (self.mine_points + [(float(ra), float(dec))])[-20:]
        if self.mine_auto.value:
            self._analyse_at(ra, dec, None, "the point you clicked")
            return
        d = ac.describe(ac.parse_position(f"{float(ra):.6f} {float(dec):+.6f}"))
        self.pos.value = f"{d['ra_hms']} {d['dec_dms']}"
        self.status.value = f"&nbsp;Position set from your image ({d['pn_g']}): give a size if you know it, then " \
                            "press Analyse field"

    def _analyse_at(self, ra, dec, cut_arcmin, what, keep_size=False):
        """Fills the search boxes and runs the full analysis (catalogues, imposters, central star, surveys)."""
        if self._busy:
            self.status.value = "&nbsp;Still busy with the previous search: click again when it has finished"
            return
        d = ac.describe(ac.parse_position(f"{float(ra):.6f} {float(dec):+.6f}"))
        self.pos.value = f"{d['ra_hms']} {d['dec_dms']}"
        self.cut.value = f"{cut_arcmin:.0f}" if cut_arcmin else ""
        if not keep_size and self.mine_circle is not None:   # a plain click after a circle: the circle's size no longer applies
            self.size.value = ""
        self.status.value = f"&nbsp;Searching the surveys at {what} ({d['pn_g']})..."
        try:
            self.run()
        except Exception as exc:   # never leave the screen locked
            hl.log_problem(f"search at {what}", exc)
            self._busy = False
            self.status.value = f"<span style='color:#FF9F0A'>&nbsp;Search at {what} failed: " \
                                f"{' '.join(str(exc).split())[:140]}</span>"

    def _report(self):
        """Session 14: one PDF with summary, finding chart, survey overview and credits."""
        if self.result is None or self._busy:
            return
        r = self.result
        self._working(True)
        try:
            self._report_inner(r)
        finally:
            self._working(False)

    def _report_inner(self, r):
        if not any(self.panel.get(k, {}).get("status") == "ok" for k in rp.FINDER_KEYS):
            self.status.value = "&nbsp;Fetching an optical image for the finding chart..."
            todo = [x for x in self.registry if x["key"] in rp.FINDER_KEYS and self._covered(x["key"])][:1]
            if todo:
                self.panel.update(pn.fetch_panel_parallel(r["coord"], self.fov, todo, 400, self.fetcher, 1))
        self.status.value = "&nbsp;Writing the candidate report..."
        name = r["desc"]["pn_g"].replace(" ", "")
        mine = dict(sol=self.mine_sol, points=self.mine_points, known=self.mine_known,
                    circle=self.mine_circle) if self.mine_sol else None
        out = rp.build_report(self.base / "results" / "reports" / f"{name}.pdf", r, self.card, self.panel,
                              self.diffs, self.fov, summary_lines(r), RANK_ACCURACY, user_image=mine)
        self.status.value = (f"&nbsp;Saved {name}.pdf ({out['pages']} pages) in Aletheia/results/reports"
                             + (" with your image on page 2" if out.get("user_image") else
                                (" (your image does not contain this position, so it is not included)" if mine else "")))
        if self.web:
            self._offer(out["path"], "application/pdf")
            return
        try:
            from google.colab import files
            files.download(out["path"])
        except Exception:
            pass


# --------------------------------------------------------------------------- live sky viewer (Session 10c)
# Pinned stable version (Session 16b): "latest" moved to a 3.9 beta that broke the live view.
# Tried in order; the label of the one that loaded is shown under the viewer.
ALADIN_JS = [("Aladin Lite 3.8.2 (CDS)", "https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js"),
             ("Aladin Lite 3.8.2 (jsDelivr)", "https://cdn.jsdelivr.net/npm/aladin-lite@3.8.2/dist/aladin.js"),
             ("Aladin Lite latest (CDS)", "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js")]


def survey_stats(item, bins=120, zlo=-6.0, zhi=60.0):
    """Sky level, noise and a small histogram from the preview cutout (sets the live cuts in data units)."""
    d = item.get("data")
    if d is None or d.ndim != 2:
        return None
    f = np.isfinite(d)
    if not f.any():
        return None
    med = float(np.median(d[f]))
    sig = float(pn._robust_sigma(d[f] - med)) or 1.0
    h, _ = np.histogram(np.clip((d[f] - med) / sig, zlo, zhi), bins=bins, range=(zlo, zhi))
    return dict(med=med, sig=sig, hist=[int(v) for v in h], zlo=zlo, zhi=zhi)


def live_viewer_html(uid, coord, fov_arcmin, layers, start_index=0, ring_arcmin=None, marks=None, width=760):
    """Aladin Lite v3 embedded with all of its own controls hidden. Everything runs in the browser:
    black/white point (in sky-noise units), midtones (gamma), stretch, colour map, invert, blur,
    per-survey settings, survey menu with previous/next, RA/Dec readout, ring and candidate markers.
    layers: list of dict(id, label, fits, colour, stats or None)."""
    import json
    cfg = dict(ra=float(coord.ra.deg), dec=float(coord.dec.deg), fov=float(fov_arcmin) / 60.0, layers=layers,
               start=int(start_index), ring=(float(ring_arcmin) / 120.0 if ring_arcmin else None),
               marks=marks or [], js=[u for _, u in ALADIN_JS], jsl=[l for l, _ in ALADIN_JS])
    return """
<style>
 #W{uid} .aladin-logo-container, #W{uid} .aladin-logo {{ display:none !important; }}
 #W{uid} label {{ color:#8E8E93; font-size:12px; display:block; margin-top:10px; }}
 #W{uid} input[type=range] {{ width:230px; accent-color:#0A84FF; }}
 #W{uid} select {{ background:#1C1C1E; color:#F5F5F7; border:1px solid rgba(255,255,255,0.12); border-radius:10px;
   padding:6px 10px; color-scheme:dark; }}
 #W{uid} select option {{ background:#1C1C1E; color:#F5F5F7; }}
 #W{uid} .aladin-input-text {{ background:rgba(28,28,30,0.78) !important; color:#F5F5F7 !important; }}
</style>
<div id="W{uid}" style="display:flex;gap:14px;align-items:flex-start">
 <div>
  <div style="margin-bottom:6px">
   <button id="P{uid}">&#9664;</button>
   <select id="S{uid}"></select>
   <button id="N{uid}">&#9654;</button>
   <span style="color:#8E8E93;font-family:monospace;font-size:11px">&nbsp;wheel = zoom · drag = pan</span>
  </div>
  <div id="A{uid}" style="width:{w}px;height:{w}px;background:#000"></div>
  <div id="C{uid}" style="font-family:monospace;color:#64D2FF;margin-top:4px">RA --  Dec --</div>
  <div id="E{uid}" style="font-family:monospace;color:#FF9F0A;font-size:11px"></div>
  <div id="V{uid}" style="font-family:monospace;color:#555;font-size:10px"></div>
 </div>
 <div style="min-width:250px">
  <label>Black point (sky sigma) <span id="bv{uid}"></span></label><input id="b{uid}" type="range" min="0" max="8" step="0.25">
  <label>White point (sky sigma) <span id="wv{uid}"></span></label><input id="w{uid}" type="range" min="2" max="80" step="0.5">
  <label>Midtones (gamma) <span id="gv{uid}"></span></label><input id="g{uid}" type="range" min="0.2" max="3" step="0.05">
  <canvas id="H{uid}" width="240" height="56" style="display:block;margin-top:8px;background:rgba(28,28,30,0.7)"></canvas>
  <label>Stretch</label><select id="t{uid}"><option>asinh</option><option>sqrt</option><option>log</option><option>linear</option><option>pow2</option></select>
  <label>Colour map</label><select id="m{uid}"><option>grayscale</option><option>native</option><option>viridis</option><option>inferno</option><option>cubehelix</option><option>redtemperature</option></select>
  <label><input id="r{uid}" type="checkbox" style="width:auto"> Invert</label>
  <label>Blur (viewing aid) <span id="lv{uid}"></span></label><input id="l{uid}" type="range" min="0" max="4" step="0.25">
  <div style="margin-top:8px"><button id="R{uid}">Reset this survey</button></div>
  <div id="I{uid}" style="color:#6E6E73;font-family:monospace;font-size:11px;margin-top:8px;max-width:250px"></div>
 </div>
</div>
<script>
(function() {{
 const cfg = {cfg};
 const $ = (k) => document.getElementById(k + "{uid}");
 const err = (m) => {{ $("E").textContent = m; }};
 const settings = {{}};
 const defaults = (L) => ({{black: 2, white: 10, gamma: 1, stretch: L.colour ? "linear" : "asinh",
                            cmap: L.colour ? "native" : "grayscale", invert: false, blur: 0}});
 let al = null, idx = cfg.start, cache = {{}};
 function fmtRA(d) {{ let h = d / 15, H = Math.floor(h), m = (h - H) * 60, M = Math.floor(m), S = (m - M) * 60;
   if (S >= 59.995) {{ S = 0; M += 1; }} if (M >= 60) {{ M = 0; H = (H + 1) % 24; }}
   return String(H).padStart(2, "0") + ":" + String(M).padStart(2, "0") + ":" + S.toFixed(2).padStart(5, "0"); }}
 function fmtDec(d) {{ const s = d < 0 ? "-" : "+"; d = Math.abs(d); let D = Math.floor(d), m = (d - D) * 60, M = Math.floor(m), S = (m - M) * 60;
   if (S >= 59.95) {{ S = 0; M += 1; }} if (M >= 60) {{ M = 0; D += 1; }}
   return s + String(D).padStart(2, "0") + ":" + String(M).padStart(2, "0") + ":" + S.toFixed(1).padStart(4, "0"); }}
 function cuts(L, st) {{  // black/white point in sky-noise units -> data units (needs preview stats)
   if (!L.stats) return null;
   return [L.stats.med - st.black * L.stats.sig, L.stats.med + st.white * L.stats.sig];
 }}
 function drawHist(L, st) {{
   const c = $("H"), g = c.getContext("2d"); g.fillStyle = "#0a0a0a"; g.fillRect(0, 0, c.width, c.height);
   if (!L.stats) {{ g.fillStyle = "#555"; g.fillText("no histogram for colour surveys", 10, 30); return; }}
   const h = L.stats.hist, mx = Math.max(...h.map(v => Math.log1p(v))) || 1, bw = c.width / h.length;
   g.fillStyle = "#8c8c8c";
   h.forEach((v, i) => {{ const y = Math.log1p(v) / mx * (c.height - 4); g.fillRect(i * bw, c.height - y, Math.max(1, bw), y); }});
   const X = (z) => (Math.min(Math.max(z, L.stats.zlo), L.stats.zhi) - L.stats.zlo) / (L.stats.zhi - L.stats.zlo) * c.width;
   g.fillStyle = "#64D2FF"; g.fillRect(X(-st.black), 0, 2, c.height);
   g.fillStyle = "#FF9F0A"; g.fillRect(X(st.white), 0, 2, c.height);
 }}
 function layerFor(L) {{
   if (cache[L.id]) return cache[L.id];
   let s;
   if (typeof A.imageHiPS === "function") s = A.imageHiPS(L.id, {{imgFormat: L.fits ? "fits" : "jpeg"}});
   else {{ s = al.newImageSurvey(L.id); if (L.fits && s.setImageFormat) {{ try {{ s.setImageFormat("fits"); }} catch (e) {{}} }} }}
   cache[L.id] = s; return s;
 }}
 function apply() {{
   const L = cfg.layers[idx], st = settings[L.id] || (settings[L.id] = defaults(L));
   $("b").value = st.black; $("w").value = st.white; $("g").value = st.gamma; $("t").value = st.stretch;
   $("m").value = st.cmap; $("r").checked = st.invert; $("l").value = st.blur;
   $("bv").textContent = st.black; $("wv").textContent = st.white; $("gv").textContent = st.gamma; $("lv").textContent = st.blur + " px";
   ["b", "w", "g"].forEach(k => $(k).disabled = !L.stats);
   const s = al ? al.getBaseImageLayer() : null;
   if (s) {{
     try {{ s.setColormap(st.cmap, {{stretch: st.stretch, reversed: st.invert}}); }} catch (e) {{ err("colour map: " + e); }}
     const c = cuts(L, st); if (c) {{ try {{ s.setCuts(c[0], c[1]); }} catch (e) {{ err("cuts: " + e); }} }}
     if (typeof s.setGamma === "function") {{ try {{ s.setGamma(st.gamma); }} catch (e) {{}} }}
   }}
   $("A").style.filter = st.blur > 0 ? `blur(${{st.blur}}px)` : "none";
   $("I").textContent = L.label + (L.stats ? "" : " (colour survey: black/white/midtones not used)") + (L.fits ? " · full-depth FITS tiles" : "");
   drawHist(L, st);
 }}
 function show(i) {{
   idx = (i + cfg.layers.length) % cfg.layers.length; $("S").value = String(idx);
   const L = cfg.layers[idx];
   try {{ al.setImageLayer(layerFor(L)); }} catch (e) {{ err("could not load " + L.label + ": " + e); }}
   apply();
 }}
 function wire() {{
   cfg.layers.forEach((L, i) => {{ const o = document.createElement("option"); o.value = String(i); o.textContent = L.label; $("S").appendChild(o); }});
   $("S").onchange = () => show(parseInt($("S").value));
   $("P").onclick = () => show(idx - 1); $("N").onclick = () => show(idx + 1);
   const upd = (k, key, conv) => {{ $(k).oninput = () => {{ const L = cfg.layers[idx]; settings[L.id][key] = conv($(k)); apply(); }}; }};
   upd("b", "black", e => parseFloat(e.value)); upd("w", "white", e => parseFloat(e.value)); upd("g", "gamma", e => parseFloat(e.value));
   upd("t", "stretch", e => e.value); upd("m", "cmap", e => e.value); upd("r", "invert", e => e.checked); upd("l", "blur", e => parseFloat(e.value));
   $("R").onclick = () => {{ const L = cfg.layers[idx]; settings[L.id] = defaults(L); apply(); }};
   $("A").addEventListener("mousemove", (ev) => {{
     const r = $("A").getBoundingClientRect(); let p = null;
     try {{ p = al.pix2world(ev.clientX - r.left, ev.clientY - r.top); }} catch (e) {{}}
     $("C").textContent = p ? "RA " + fmtRA(p[0]) + "   Dec " + fmtDec(p[1]) + "   (" + p[0].toFixed(5) + ", " + p[1].toFixed(5) + ")" : "RA --  Dec --";
   }});
 }}
 function start() {{
   A.init.then(() => {{
     al = A.aladin("#A{uid}", {{target: cfg.ra + " " + cfg.dec, fov: cfg.fov, projection: "TAN",
       showLayersControl: false, showGotoControl: false, showZoomControl: false, showFullscreenControl: false,
       showProjectionControl: false, showCooGridControl: false, showSimbadPointerControl: false,
       showShareControl: false, showContextMenu: false, showFrame: false, showReticle: false}});
     if (cfg.ring) {{ const ov = A.graphicOverlay({{color: "#64D2FF", lineWidth: 1}}); al.addOverlay(ov); ov.add(A.circle(cfg.ra, cfg.dec, cfg.ring)); }}
     if (cfg.marks.length) {{
       const cat = A.catalog({{name: "candidates", shape: "circle", color: "#FF9F0A", sourceSize: 16, labelColumn: "n", labelColor: "#FF9F0A"}});
       al.addCatalog(cat); cat.addSources(cfg.marks.map(m => A.source(m.ra, m.dec, {{n: m.n}})));
     }}
     show(cfg.start);
   }}).catch(e => err("sky viewer failed to start: " + e));
 }}
 wire();
 function load(i) {{   // pinned version first; next source only if one fails
   if (i >= cfg.js.length) {{ err("could not load the sky viewer (no internet?)"); return; }}
   const sc = document.createElement("script"); sc.src = cfg.js[i]; sc.charset = "utf-8";
   sc.onload = () => {{ if (window.A && window.A.init) {{ $("V").textContent = "viewer: " + cfg.jsl[i]; start(); }}
                       else load(i + 1); }};
   sc.onerror = () => load(i + 1); document.head.appendChild(sc);
 }}
 if (window.A && window.A.init) start(); else load(0);
}})();
</script>""".format(uid=uid, w=width, cfg=json.dumps(cfg))
