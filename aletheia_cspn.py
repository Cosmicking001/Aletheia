"""
aletheia_cspn: white dwarf and central star finder (Session 9)

find_central_stars(coord, size_arcmin=None) -> dict
    stars:   Gaia DR3 stars within the search radius, each with
             offset, G, BP-RP, parallax, RUWE (catalogue values, Gaia DR3),
             P_WD (catalogue value, Gentile Fusillo et al. 2021, if listed),
             CSPN flags (catalogue values, Chornay & Walton 2021 / Gonzalez-Santamaria et al. 2021),
             a blind score and a final rank.
    radius:  search radius used (arcsec)

Blind score (does NOT use the CSPN catalogues, so it can be tested against them):
    score = 0.5 x centrality + 0.3 x blueness + 0.2 x P_WD
    centrality = exp(-0.5 (d / s)^2), s = max(4", 0.30 x nebula radius)
    blueness   = share of stars in the search area that are redder (BP-RP), 0..1
    P_WD       = white dwarf probability, 0 if the star is not in the catalogue
Final order: catalogued central stars first, then by blind score.
Every value is a catalogue value or a computed ranking; nothing here is a measurement.
"""
import math

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

GAIA = "I/355/gaiadr3"
GF21 = "J/MNRAS/508/3877/maincat"
CW21 = "J/A+A/656/A110"
GS21 = "J/A+A/656/A51"
BJ21 = "I/352/gedr3dis"   # Bailer-Jones et al. 2021, Gaia EDR3 distances
GUV = "II/335/galex_ais"  # Bianchi et al. 2017, GALEX unique UV sources (GUVcat_AIS GR6+7)
UV_MATCH_ARCSEC = 3.0     # GALEX resolution is about 5 arcsec
PC_PER_ARCSEC_KPC = 1000.0 / 206264.806  # pc per arcsec at 1 kpc
MIN_RADIUS_ARCSEC, MAX_RADIUS_ARCSEC = 30.0, 300.0
GAIA_ROW_LIMIT = 20000
CENTRALITY_FRACTION = 0.30   # Session 11b: window = 30% of the nebula radius (was 15%)
LEARN_FEATURES = ["offset_over_radius", "blueness", "bp_rp", "p_wd", "in_wd_catalogue",
                  "nuv_minus_g", "fuv_minus_nuv", "uv_detected"]  # no brightness or parallax-quality bias


def search_radius_arcsec(size_arcmin):
    """Central stars sit near the middle: search the inner half of the nebula (30" to 5')."""
    if size_arcmin in (None, "") or (isinstance(size_arcmin, float) and math.isnan(size_arcmin)):
        return 60.0
    return float(min(MAX_RADIUS_ARCSEC, max(MIN_RADIUS_ARCSEC, 0.5 * float(size_arcmin) * 60.0)))


def _col(table, *names, contains=None):
    for n in names:  # exact match first: b_rgeo and B_rgeo are different columns
        if n in table.colnames:
            return n
    low = {c.lower(): c for c in table.colnames}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if contains:
        for c in table.colnames:
            if contains.lower() in c.lower():
                return c
    return None


def _float(v):
    if np.ma.is_masked(v):   # empty catalogue cell: missing, without the flood of warnings
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _query(catalog, coord, radius_arcsec, columns=("*",), row_limit=2000):
    from astroquery.vizier import Vizier
    v = Vizier(columns=list(columns) + ["_RAJ2000", "_DEJ2000", "+_r"], row_limit=row_limit)  # nearest first
    v.TIMEOUT = 120
    return v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=catalog)


def _rows_with_pos(tablelist):
    """(table, row, SkyCoord) for every row of every table returned."""
    out = []
    if not tablelist:
        return out
    for key in tablelist.keys():
        t = tablelist[key]
        for row in t:
            ra, de = _float(row["_RAJ2000"]), _float(row["_DEJ2000"])
            if ra is None or de is None:
                continue
            out.append((key, t, row, SkyCoord(ra * u.deg, de * u.deg)))
    return out


def gaia_stars(coord, radius_arcsec, querier=None):
    q = querier or _query
    tl = q(GAIA, coord, radius_arcsec, columns=("Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "Plx",
                                                 "e_Plx", "RUWE", "pmRA", "pmDE"), row_limit=GAIA_ROW_LIMIT)
    stars = []
    for _, t, row, c in _rows_with_pos(tl):
        sid = row[_col(t, "Source")] if _col(t, "Source") else None
        stars.append(dict(source=str(sid), coord=c, sep=coord.separation(c).arcsec,
                          G=_float(row[_col(t, "Gmag")]) if _col(t, "Gmag") else None,
                          bp_rp=_float(row[_col(t, "BP-RP")]) if _col(t, "BP-RP") else None,
                          plx=_float(row[_col(t, "Plx")]) if _col(t, "Plx") else None,
                          e_plx=_float(row[_col(t, "e_Plx")]) if _col(t, "e_Plx") else None,
                          ruwe=_float(row[_col(t, "RUWE")]) if _col(t, "RUWE") else None,
                          p_wd=None, cspn=[], cspn_note=""))
    return stars


def _match(stars, entries, max_sep=1.5):
    """Attach catalogue entries to Gaia stars by source id, else by position (1.5")."""
    by_id = {s["source"]: s for s in stars}
    for e in entries:
        s = by_id.get(e["source"]) if e.get("source") else None
        if s is None and stars:
            seps = [e["coord"].separation(x["coord"]).arcsec for x in stars]
            i = int(np.argmin(seps))
            s = stars[i] if seps[i] <= max_sep else None
        if s is not None:
            yield s, e


def add_white_dwarfs(stars, coord, radius_arcsec, querier=None):
    q = querier or _query
    tl = q(GF21, coord, radius_arcsec + 2)
    entries = []
    for _, t, row, c in _rows_with_pos(tl):
        pcol = _col(t, "Pwd", contains="pwd")
        idcol = _col(t, "GaiaEDR3", "Source", "EDR3", contains="gaia")
        get = lambda *names, contains=None: (_float(row[_col(t, *names, contains=contains)])
                                             if _col(t, *names, contains=contains) else None)
        entries.append(dict(coord=c, p_wd=_float(row[pcol]) if pcol else None,
                            source=str(row[idcol]).strip() if idcol else None,
                            teff=get("TeffH", contains="teffh"), e_teff=get("e_TeffH", contains="e_teffh"),
                            logg=get("loggH", contains="loggh"), mass=get("MassH", contains="massh")))
    for s, e in _match(stars, entries):
        if e["p_wd"] is not None:
            s["p_wd"] = e["p_wd"]
        for k in ("teff", "e_teff", "logg", "mass"):
            if e.get(k) is not None:
                s[k] = e[k]


def count_cspn_entries(coord, radius_arcsec, querier=None):
    """How many CSPN catalogue entries lie within radius (used to tell 'missed' from 'not catalogued')."""
    q = querier or _query
    n = 0
    for cat in (CW21, GS21):
        try:
            n += len(_rows_with_pos(q(cat, coord, radius_arcsec)))
        except Exception:
            continue
    return n


def add_cspn_catalogues(stars, coord, radius_arcsec, querier=None):
    q = querier or _query
    for cat, label in ((CW21, "Chornay & Walton 2021"), (GS21, "Gonzalez-Santamaria et al. 2021")):
        try:
            tl = q(cat, coord, radius_arcsec + 2)
        except Exception:
            continue
        entries = []
        for _, t, row, c in _rows_with_pos(tl):
            idcol = _col(t, "GaiaEDR3", "Source", "EDR3", "Gaia", contains="gaia")
            name_col = _col(t, "Name", "PNG", "PN", contains="name")
            dist_col = _col(t, "Dist", "dist", "Dpc", "Dkpc", "D", contains="dist")
            note = str(row[name_col]).strip() if name_col else ""
            cat_dist = _float(row[dist_col]) if dist_col else None
            if cat_dist is not None and cat_dist < 30:   # given in kpc
                cat_dist *= 1000.0
            if cat_dist is not None:
                note += f", distance {cat_dist:.0f} pc (catalogue value)"
            png_col = _col(t, "PNG")
            entries.append(dict(coord=c, source=str(row[idcol]).strip() if idcol else None, label=label, note=note,
                                cat_dist=cat_dist, png=str(row[png_col]).strip() if png_col else None))
        for s, e in _match(stars, entries):
            if e.get("cat_dist") is not None:
                s.setdefault("cat_dist", {})[e["label"]] = e["cat_dist"]
            if e["label"].startswith("Gonzalez") and e.get("png"):
                s["gs_png"] = e["png"]
            if e["label"] not in s["cspn"]:
                s["cspn"].append(e["label"])
                s["cspn_note"] = (s["cspn_note"] + "; " if s["cspn_note"] else "") + f"{e['label']}: {e['note']}"


def add_distances(stars, coord, radius_arcsec, querier=None):
    """Bailer-Jones et al. 2021 geometric distances (catalogue values), matched by Gaia source id."""
    q = querier or _query
    try:
        tl = q(BJ21, coord, radius_arcsec + 2, columns=("Source", "rgeo", "b_rgeo", "B_rgeo"))
    except Exception:
        return
    by_id = {s["source"]: s for s in stars}
    for _, t, row, c in _rows_with_pos(tl):
        sid = str(row[_col(t, "Source")]).strip() if _col(t, "Source") else None
        s = by_id.get(sid)
        if s is None:
            continue
        s["dist"] = _float(row[_col(t, "rgeo")]) if _col(t, "rgeo") else None
        s["dist_lo"] = _float(row[_col(t, "b_rgeo")]) if _col(t, "b_rgeo") else None
        s["dist_hi"] = _float(row[_col(t, "B_rgeo")]) if _col(t, "B_rgeo") else None


def add_ultraviolet(stars, coord, radius_arcsec, querier=None):
    """GALEX NUV/FUV (catalogue values, AB mag), each UV source given to its nearest Gaia star within 3".
    Values of -999 (no measurement) are ignored."""
    q = querier or _query
    try:
        tl = q(GUV, coord, radius_arcsec + UV_MATCH_ARCSEC, columns=("FUVmag", "NUVmag"))
    except Exception:
        return
    rows = _rows_with_pos(tl)
    if not rows or not stars:
        return
    gc = SkyCoord([s["coord"] for s in stars])
    for _, t, row, c in rows:
        nuv = _float(row[_col(t, "NUVmag", contains="nuv")]) if _col(t, "NUVmag", contains="nuv") else None
        fuv = _float(row[_col(t, "FUVmag", contains="fuv")]) if _col(t, "FUVmag", contains="fuv") else None
        nuv = None if (nuv is None or nuv < -90) else nuv
        fuv = None if (fuv is None or fuv < -90) else fuv
        if nuv is None:
            continue
        sep = c.separation(gc).arcsec
        i = int(np.argmin(sep))
        if sep[i] <= UV_MATCH_ARCSEC and (stars[i].get("nuv") is None or sep[i] < stars[i].get("uv_sep", 99)):
            stars[i]["nuv"], stars[i]["fuv"], stars[i]["uv_sep"] = nuv, fuv, float(sep[i])


def plausibility(s, size_arcmin):
    """Computed estimates from catalogue values (no extinction correction)."""
    notes = []
    d = s.get("dist")
    if s.get("plx") is not None and s.get("e_plx"):
        poe = s["plx"] / s["e_plx"]
        if poe < 3:
            notes.append("distance poorly constrained (parallax/error < 3)")
    if d:
        if size_arcmin:
            r_pc = 0.5 * float(size_arcmin) * 60.0 * d / 206264.806
            s["radius_pc"] = r_pc
            if r_pc > 2.0:
                notes.append(f"nebula would be {r_pc:.1f} pc in radius at this distance: larger than almost all PNe")
            elif r_pc > 1.0:
                notes.append(f"nebula radius {r_pc:.2f} pc at this distance: large, like an old PN")
            else:
                notes.append(f"nebula radius {r_pc:.2f} pc at this distance: typical of PNe")
        if s.get("G") is not None:
            s["abs_G"] = s["G"] - 5 * math.log10(d / 10.0)
    if s.get("teff") and s["teff"] >= 30000:
        notes.append("very hot: colours barely change above about 30,000 K, so this temperature is a lower limit")
    s["plausibility"] = notes


FEATURES = ["offset_over_radius", "blueness", "bp_rp", "p_wd", "in_wd_catalogue", "G", "parallax_snr",
            "nuv_minus_g", "fuv_minus_nuv", "uv_detected"]
RANK_MODEL = None   # learned in Session 11 (logistic regression); None = Session 9 formula


def features(stars, size_arcmin):
    """One row per star, columns = FEATURES. Missing values get neutral defaults."""
    radius = 0.5 * float(size_arcmin) * 60 if size_arcmin else 30.0
    colours = np.array([s["bp_rp"] for s in stars if s["bp_rp"] is not None])
    rows = []
    for s in stars:
        blue = float(np.mean(colours > s["bp_rp"])) if (s["bp_rp"] is not None and colours.size > 1) else 0.5
        snr = (s["plx"] / s["e_plx"]) if (s.get("plx") is not None and s.get("e_plx")) else 0.0
        nuv_g = (s["nuv"] - s["G"]) if (s.get("nuv") is not None and s.get("G") is not None) else 5.0
        fuv_nuv = (s["fuv"] - s["nuv"]) if (s.get("fuv") is not None and s.get("nuv") is not None) else 1.5
        rows.append([s["sep"] / max(radius, 1.0), blue, s["bp_rp"] if s["bp_rp"] is not None else 1.0,
                     s["p_wd"] or 0.0, 1.0 if s["p_wd"] is not None else 0.0,
                     s["G"] if s["G"] is not None else 19.0, float(np.clip(snr, -5, 50)),
                     float(np.clip(nuv_g, -4, 10)), float(np.clip(fuv_nuv, -3, 5)),
                     1.0 if s.get("nuv") is not None else 0.0])
    return np.array(rows, dtype=float)


def model_probability(X, model):
    cols = [FEATURES.index(f) for f in model.get("features", FEATURES)]
    X = X[:, cols]
    z = (X - np.array(model["mean"])) / np.array(model["std"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))


def score(stars, size_arcmin):
    radius_arcsec = 0.5 * float(size_arcmin) * 60 if size_arcmin else 30.0
    sigma = max(4.0, CENTRALITY_FRACTION * radius_arcsec)
    colours = np.array([s["bp_rp"] for s in stars if s["bp_rp"] is not None])
    for s in stars:
        centrality = math.exp(-0.5 * (s["sep"] / sigma) ** 2)
        if s["bp_rp"] is not None and colours.size > 1:
            blueness = float(np.mean(colours > s["bp_rp"]))
        else:
            blueness = 0.5
        pwd = s["p_wd"] or 0.0
        s["parts"] = dict(centrality=round(centrality, 3), blueness=round(blueness, 3), p_wd=round(pwd, 3))
        s["score"] = round(0.5 * centrality + 0.3 * blueness + 0.2 * pwd, 3)
    if RANK_MODEL is not None and stars:
        probs = model_probability(features(stars, size_arcmin), RANK_MODEL)
        for st, pr in zip(stars, probs):
            st["score"] = round(float(pr), 3)
    blind = sorted(stars, key=lambda x: -x["score"])
    for i, s in enumerate(blind, 1):
        s["blind_rank"] = i
    final = sorted(stars, key=lambda x: (0 if x["cspn"] else 1, -x["score"]))
    for i, s in enumerate(final, 1):
        s["rank"] = i
    return final


def find_central_stars(coord, size_arcmin=None, querier=None):
    r = search_radius_arcsec(size_arcmin)
    stars = gaia_stars(coord, r, querier)
    truncated = len(stars) >= GAIA_ROW_LIMIT
    if stars:
        add_white_dwarfs(stars, coord, r, querier)
        add_cspn_catalogues(stars, coord, r, querier)
        add_distances(stars, coord, r, querier)
        add_ultraviolet(stars, coord, r, querier)
        for st in stars:
            plausibility(st, size_arcmin)
        stars = score(stars, size_arcmin)
    return dict(stars=stars, radius=r, truncated=truncated)


def describe_star(s):
    bits = [f"#{s['rank']} Gaia DR3 {s['source']}", f"{s['sep']:.1f}\" from centre"]
    if s["G"] is not None:
        bits.append(f"G {s['G']:.2f}")
    if s["bp_rp"] is not None:
        bits.append(f"BP-RP {s['bp_rp']:+.2f}")
    if s["plx"] is not None and s["e_plx"]:
        bits.append(f"parallax {s['plx']:.2f} +/- {s['e_plx']:.2f} mas")
    if s["ruwe"] is not None:
        bits.append(f"RUWE {s['ruwe']:.2f}" + (" (possible binary)" if s["ruwe"] > 1.4 else ""))
    if s["p_wd"] is not None:
        bits.append(f"P_WD {s['p_wd']:.2f}")
    if s["cspn"]:
        bits.append("CATALOGUED CENTRAL STAR (" + s["cspn_note"] + ")")
    bits.append(f"score {s['score']:.2f} (blind rank {s['blind_rank']})")
    return " · ".join(bits)



def plain_summary(s, n_stars):
    """One readable line for a candidate, plus the reason it ranks where it does."""
    bits = [f"#{s['rank']}", f"{s['sep']:.1f}\" from centre"]
    b = s["parts"]["blueness"]
    if s["bp_rp"] is not None:
        bits.append("blue" if b >= 0.9 else ("bluish" if b >= 0.7 else "red" if b <= 0.3 else "average colour"))
    if s.get("G") is not None:
        bits.append(f"G {s['G']:.1f}")
    if s.get("p_wd") is not None:
        bits.append(f"white dwarf probability {100 * s['p_wd']:.0f}%")
    if s.get("dist"):
        rng = (f" ({s['dist_lo']:.0f} to {s['dist_hi']:.0f})" if s.get("dist_lo") and s.get("dist_hi") else "")
        bits.append(f"distance ~{s['dist']:.0f} pc{rng}")
    if s.get("teff"):
        bits.append(f"Teff ~{s['teff']:,.0f} K" + (f", log g {s['logg']:.2f}" if s.get("logg") else "")
                    + (f", {s['mass']:.2f} Msun" if s.get("mass") else ""))
    if s.get("nuv") is not None and s.get("G") is not None:
        nuv_g = s["nuv"] - s["G"]
        bits.append(f"UV-bright (NUV-G {nuv_g:+.1f})" if nuv_g < 1.5 else f"NUV-G {nuv_g:+.1f}")
    if s.get("ruwe") and s["ruwe"] > 1.4:
        bits.append(f"RUWE {s['ruwe']:.1f}: possible binary")
    if s["cspn"]:
        bits.append("catalogued central star")
    reasons = []
    if s["cspn"]:
        reasons.append("listed in a central-star catalogue")
    if s["parts"]["centrality"] >= 0.6:
        reasons.append("close to the centre")
    if b >= 0.9:
        reasons.append(f"bluer than {100 * b:.0f}% of the {n_stars} stars nearby")
    if (s.get("p_wd") or 0) >= 0.75:
        reasons.append("high white dwarf probability")
    if s.get("nuv") is not None and s.get("G") is not None and s["nuv"] - s["G"] < 1.5:
        reasons.append("bright in the ultraviolet (hot)")
    return " · ".join(bits), ", ".join(reasons)
