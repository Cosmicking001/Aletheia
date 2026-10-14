"""
aletheia_crossmatch: Aletheia cross-match A (Session 4)

Is this position already a known or reported planetary nebula?
Checks SIMBAD, Acker/SECGPN (true, possible, rejected), Kohoutek 2000, MASH,
French amateur lists, and the user's own catalogue. Matching is by POSITION
with a zone scaled to the object's size, never by name.

check_known(coord, size_arcmin=None, own=None, exclude_own_id=None) -> dict
"""
import math
import re

import astropy.units as u
from astropy.coordinates import SkyCoord

# VizieR table -> (label, category). Category decides the verdict.
VIZIER_TABLES = {
    "V/84/main": ("Acker SECGPN: true/probable PN", "KNOWN PN"),
    "V/84/pospn": ("Acker SECGPN: possible PN", "KNOWN CANDIDATE"),
    "V/84/notpn": ("Acker SECGPN: rejected (not PN)", "PREVIOUSLY REJECTED"),
    "IV/24/table2": ("Kohoutek CGPN 2000: PN", "KNOWN PN"),
    "IV/24/table3": ("Kohoutek CGPN 2000: misclassified", "PREVIOUSLY REJECTED"),
    "IV/24/table5": ("Kohoutek: possible pre-PN", "KNOWN CANDIDATE"),
    "IV/24/table6": ("Kohoutek: possible post-PN", "KNOWN CANDIDATE"),
    "V/127A": ("MASH (Parker+ 2006-2008)", "KNOWN PN"),
    "J/other/LAstr/80.42": ("French amateur PN list (L'Astronomie 80, Ferrero+ 2015)", "KNOWN CANDIDATE"),
    "J/other/LAstr/102.46": ("French amateur PN list (L'Astronomie 102, Le Du)", "KNOWN CANDIDATE"),
    "J/other/LAstr/125.54": ("French amateur PN list (L'Astronomie 125, Le Du 2019)", "KNOWN CANDIDATE"),
    "J/other/LAstr/136.58": ("French amateur PN list (L'Astronomie 136, Le Du 2020)", "KNOWN CANDIDATE"),
}
EXTRA = []   # local copies of other lists, e.g. planetarynebulae.net (Session 16): dicts name, coord, source


STATUS_EN = [("vraie", "true PN"), ("probable", "probable PN"), ("possible", "possible PN"),
             ("nouvelle candidate", "new PN candidate"), ("nature inconnue", "object of unknown nature"),
             ("pagb", "post-AGB / pre-PN"), ("pre-np", "post-AGB / pre-PN")]


# Every status used on planetarynebulae.net (read from the full copy, Session 18c): English, Aletheia category
C, P, R = "KNOWN CANDIDATE", "KNOWN PN", "PREVIOUSLY REJECTED"
STATUS_TABLE = {
    "vraie np": ("true PN", P), "np probable": ("probable PN", P), "possible np": ("possible PN", C),
    "nouvelle candidate": ("new PN candidate", C), "objet de nature inconnue": ("object of unknown nature", C),
    "possible proto np": ("possible proto-PN", C), "pagb/pre-np": ("post-AGB / pre-PN", C),
    "objet interessant": ("interesting object", C), "objet en emission": ("emission object", C),
    "": ("no status given", C),
    "non np": ("not a PN", R), "etoile avec raies en emission": ("emission-line star", R),
    "etoile avec raies en emission possible": ("possible emission-line star", R), "region hii": ("H II region", R),
    "etoile": ("star", R), "galaxie": ("galaxy", R), "galaxie ?": ("possible galaxy", R),
    "ism ionise": ("ionised interstellar gas", R), "candidat remanent de supernova": ("SNR candidate", R),
    "remanent de supernova": ("supernova remnant", R), "objet herbig-haro": ("Herbig-Haro object", R),
    "candidate etoile symbiotique": ("symbiotic star candidate", R), "etoile symbiotique": ("symbiotic star", R),
    "amas d'etoiles": ("star cluster", R), "matiere circumstellaire": ("circumstellar material", R),
    "naine blanche": ("white dwarf", R), "globule cometaire": ("cometary globule", R),
    "variable cataclysmique": ("cataclysmic variable", R),
}


def _status_key(status):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(status or "")).encode("ascii", "ignore").decode()
    return " ".join(t.lower().replace("\u2019", "'").split())


def status_english(status):
    hit = STATUS_TABLE.get(_status_key(status))
    if hit:
        return hit[0]
    low = (status or "").lower()
    return next((en for fr, en in STATUS_EN if fr in low), (status or "").strip())


def status_category(status):
    """planetarynebulae.net status -> Aletheia verdict category."""
    hit = STATUS_TABLE.get(_status_key(status))
    if hit:
        return hit[1]
    low = (status or "").lower()
    if not low.strip() or any(w in low for w in ("possible", "candidate", "inconnue", "pagb", "pre-np")):
        return "KNOWN CANDIDATE"
    if any(w in low for w in ("vraie", "probable", "true")):
        return "KNOWN PN"
    return "PREVIOUSLY REJECTED"   # listed there as another kind of object


NEARBY_ARCMIN = 10.0   # Session 19b: known nebulae this close are always named, even when your point is outside them


def size_arcmin_of(text):
    """Largest dimension in arcmin from a size as written on planetarynebulae.net ("4,5 min d'arc",
    "12 x 9 min d'arc", "40 sec d'arc"); None when there is no unit or no number (we never guess the unit)."""
    t = str(text or "").replace(",", ".")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if not nums:
        return None
    if "sec" in t or "arcsec" in t or '"' in t:
        return max(nums) / 60.0
    if "min" in t or "arcmin" in t or "'" in t:
        return max(nums)
    return None


def load_extra(path, source="planetarynebulae.net (local copy)"):
    """Reads a CSV with a name column and RA/Dec columns (any format). Returns how many rows were loaded."""
    import csv
    from pathlib import Path
    import aletheia_coords as ac
    global EXTRA
    EXTRA = []
    p = Path(path)
    if not p.exists():
        return 0
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    if not rows:
        return 0
    low = {k.lower().strip(): k for k in rows[0]}
    pick = lambda *names: next((low[n] for n in names if n in low), None)
    n_col, r_col, d_col = pick("name", "designation", "nom"), pick("ra", "ra_deg", "raj2000"), \
        pick("dec", "dec_deg", "dej2000")
    s_col, g_col = pick("status", "statut"), pick("pn_g", "png", "coord. gal.")
    z_col, u_col = pick("size", "dimensions"), pick("unpublished")
    if r_col is None or d_col is None:
        return 0
    for r in rows:
        try:
            c = ac.parse_position(f"{r[r_col]} {r[d_col]}")
        except Exception:
            continue
        name = (r.get(n_col) or "").strip().rstrip("* ").strip() if n_col else ""   # "StDr 47 *" -> "StDr 47"
        status = (r.get(s_col) or "").strip() if s_col else ""
        en = status_english(status)
        size = (r.get(z_col) or "").strip() if z_col else ""
        major = size_arcmin_of(size)
        EXTRA.append(dict(name=name, coord=c, source=source, status=en, category=status_category(status),
                          pn_g=(r.get(g_col) or "").strip() if g_col else "", status_fr=status, size=size,
                          radius_arcmin=major / 2.0 if major else None,
                          unpublished=(r.get(u_col) or "").strip() == "yes" if u_col else False,
                          label=f"{name} [{en}]" if en else name))
    return len(EXTRA)
RANK = {"KNOWN PN": 4, "KNOWN CANDIDATE": 3, "PREVIOUSLY REJECTED": 2, "INSIDE EXTERNAL GALAXY": 1,
        "EXTRAGALACTIC PN": 0, "NO MATCH": 0}
GALAXY_TYPES = ("G", "G?", "GiC", "GiG", "GiP", "BiC", "IG", "PaG", "SBG", "SyG", "Sy1", "Sy2", "AGN",
                "LIN", "rG", "H2G", "LSB", "EmG", "QSO", "Bla", "BLL")
DEFAULT_ZONE_ARCMIN = 1.0      # used when the size is unknown
POSITION_TOLERANCE_ARCMIN = 0.5
MIN_SEARCH_ARCMIN, MAX_SEARCH_ARCMIN = 2.0, 30.0


def zones(size_arcmin):
    """(match zone, search radius) in arcmin. Match zone = object radius + tolerance."""
    if size_arcmin in (None, "") or (isinstance(size_arcmin, float) and math.isnan(size_arcmin)):
        return DEFAULT_ZONE_ARCMIN, MIN_SEARCH_ARCMIN
    s = float(size_arcmin)
    match = max(DEFAULT_ZONE_ARCMIN, 0.5 * s + POSITION_TOLERANCE_ARCMIN)
    search = min(MAX_SEARCH_ARCMIN, max(MIN_SEARCH_ARCMIN, match * 1.5))
    return min(match, MAX_SEARCH_ARCMIN), search


def _table_info(key):
    for prefix, info in VIZIER_TABLES.items():
        if key == prefix or key.startswith(prefix + "/") or key.startswith(prefix):
            return info
    return (key, "KNOWN CANDIDATE")


def _name_of(row, table):
    for c in ("PNG", "Name", "MASH", "Main", "Ident", "ID", "Seq"):
        if c in table.colnames and str(row[c]).strip() not in ("", "--"):
            return f"{c} {str(row[c]).strip()}" if c in ("Seq",) else str(row[c]).strip()
    for c in table.colnames:
        if not c.startswith("_") and table[c].dtype.kind in "OUS" and str(row[c]).strip():
            return str(row[c]).strip()
    return "(unnamed)"


def _row_coord(row):
    ra, dec = row["_RAJ2000"], row["_DEJ2000"]
    try:
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    except (TypeError, ValueError):
        return SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))


def query_vizier(coord, radius_arcmin):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=50)
    v.TIMEOUT = 90
    out = []
    try:
        tablelists = [v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=list(VIZIER_TABLES))]
    except Exception:
        tablelists = []
        for cat in VIZIER_TABLES:  # slower fallback, one catalogue at a time
            try:
                tablelists.append(v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=cat))
            except Exception:
                continue
    for tl in tablelists:
        for key in tl.keys():
            t = tl[key]
            label, cat = _table_info(key)
            for row in t:
                try:
                    c = _row_coord(row)
                except Exception:
                    continue
                out.append(dict(source=label, category=cat, name=_name_of(row, t),
                                sep_arcmin=coord.separation(c).arcmin))
    return out


def query_simbad_pn(coord, radius_arcmin):
    from astroquery.simbad import Simbad
    r_deg = radius_arcmin / 60.0
    adql = (
        "SELECT main_id, ra, dec, otype FROM basic WHERE "
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {r_deg})) = 1 "
        "AND otype IN ('PN', 'PN?')"
    )
    try:
        t = Simbad.query_tap(adql)
    except Exception:
        s = Simbad()
        s.add_votable_fields("otype")
        t = s.query_region(coord, radius=radius_arcmin * u.arcmin)
    out = []
    if t is None:
        return out
    cols = {c.lower(): c for c in t.colnames}
    for row in t:
        otype = str(row[cols["otype"]]).strip()
        if otype not in ("PN", "PN?"):
            continue
        c = SkyCoord(float(row[cols["ra"]]) * u.deg, float(row[cols["dec"]]) * u.deg)
        out.append(dict(source=f"SIMBAD ({otype})", category="KNOWN PN" if otype == "PN" else "KNOWN CANDIDATE",
                        name=" ".join(str(row[cols["main_id"]]).split()), sep_arcmin=coord.separation(c).arcmin))
    return out


def query_simbad_galaxies(coord, radius_arcmin):
    """Galaxies whose catalogued extent contains this position (SIMBAD galdim_majaxis)."""
    from astroquery.simbad import Simbad
    r_deg = max(radius_arcmin, 30.0) / 60.0
    types = ", ".join(f"'{t}'" for t in GALAXY_TYPES)
    adql = (
        "SELECT main_id, ra, dec, otype, galdim_majaxis FROM basic WHERE "
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {r_deg})) = 1 "
        f"AND otype IN ({types}) AND galdim_majaxis IS NOT NULL"
    )
    try:
        t = Simbad.query_tap(adql)
    except Exception:
        return []
    out = []
    if t is None:
        return out
    cols = {c.lower(): c for c in t.colnames}
    for row in t:
        try:
            major = float(row[cols["galdim_majaxis"]])
        except (TypeError, ValueError):
            continue
        c = SkyCoord(float(row[cols["ra"]]) * u.deg, float(row[cols["dec"]]) * u.deg)
        sep = coord.separation(c).arcmin
        if sep <= 0.5 * major:
            out.append(dict(name=" ".join(str(row[cols["main_id"]]).split()), sep_arcmin=sep, major_arcmin=major))
    return sorted(out, key=lambda g: g["sep_arcmin"])


def match_own(coord, zone_arcmin, own, exclude_own_id=None):
    hits = []
    for o in own or []:
        if exclude_own_id and o["own_id"] == exclude_own_id:
            continue
        sep = coord.separation(SkyCoord(float(o["ra_deg"]) * u.deg, float(o["dec_deg"]) * u.deg)).arcmin
        if sep <= zone_arcmin:
            label = o["own_id"] + (f" ({o['designation']})" if o.get("designation") else "")
            hits.append(dict(name=label, sep_arcmin=sep))
    return sorted(hits, key=lambda h: h["sep_arcmin"])


def check_known(coord, size_arcmin=None, own=None, exclude_own_id=None, zone_override=None):
    zone, search = zones(size_arcmin)
    if zone_override:
        zone = float(zone_override)
        search = min(MAX_SEARCH_ARCMIN, max(MIN_SEARCH_ARCMIN, 1.5 * zone))
    galaxies = query_simbad_galaxies(coord, search)
    try:
        simbad_pn = query_simbad_pn(coord, max(search, NEARBY_ARCMIN))
    except Exception:   # SIMBAD sometimes refuses one request ("Unable to access the capabilities"): retry once
        import time
        time.sleep(3)
        simbad_pn = query_simbad_pn(coord, max(search, NEARBY_ARCMIN))
    if galaxies:  # PNe that SIMBAD lists inside an external galaxy are not Galactic PNe
        host = galaxies[0]["name"]
        for f in simbad_pn:
            f["category"] = "EXTRAGALACTIC PN"
            f["source"] = f"{f['source']} in {host}"
    found = query_vizier(coord, search) + simbad_pn + [
        dict(source=e["source"], category=e.get("category", "KNOWN CANDIDATE"), name=e["name"],
             status=e.get("status", ""), pn_g=e.get("pn_g", ""), sep_arcmin=coord.separation(e["coord"]).arcmin,
             radius_arcmin=e.get("radius_arcmin"), entry=e)
        for e in EXTRA if coord.separation(e["coord"]).arcmin <= max(search, NEARBY_ARCMIN)]

    def is_inside(f):   # within the match zone, or within the object's own catalogued size (Session 19b)
        return f["sep_arcmin"] <= zone or (f.get("radius_arcmin") is not None
                                           and f["sep_arcmin"] <= f["radius_arcmin"] + POSITION_TOLERANCE_ARCMIN)
    inside = sorted([f for f in found if is_inside(f)], key=lambda f: f["sep_arcmin"])
    nearby = sorted([f for f in found if not is_inside(f)], key=lambda f: f["sep_arcmin"])
    verdict = "NO MATCH"
    for f in inside:
        if RANK[f["category"]] > RANK[verdict]:
            verdict = f["category"]
    if verdict == "NO MATCH" and galaxies:
        verdict = "INSIDE EXTERNAL GALAXY"
    return dict(verdict=verdict, galaxies=galaxies, zone_arcmin=zone, search_arcmin=search, inside=inside, nearby=nearby,
                own=match_own(coord, zone, own, exclude_own_id))


def identify(known, official=None):
    """Plain identity of a known object at this position: its name, classification, PN G name and SIMBAD
    entry. Returns (title, lines); title is None when nothing is catalogued in the zone."""
    inside = known.get("inside") or []
    if not inside:
        return None, []
    best = sorted(inside, key=lambda f: (-RANK.get(f["category"], 0), "entry" not in f, f["sep_arcmin"]))[0]
    cls = best.get("status") or {"KNOWN PN": "known PN", "KNOWN CANDIDATE": "known PN candidate",
                                  "PREVIOUSLY REJECTED": "listed, but not as a PN",
                                  "EXTRAGALACTIC PN": "PN in another galaxy"}.get(best["category"], best["category"])
    png = official or next((f["pn_g"] for f in inside if f.get("pn_g")), None)
    simbad = [f for f in inside if f["source"].startswith("SIMBAD")]
    lines = [f"Known object: {best['name']} ({cls}), listed in {best['source']}, {best['sep_arcmin'] * 60:.1f}\" "
             "from your position",
             "PN G name: " + (png.replace("PN-G", "PN G") if png else "none catalogued"),
             "SIMBAD: " + ("; ".join(f"{f['name']} (type {f['source'].split('(')[-1].split(')')[0]})" for f in simbad[:3])
                           if simbad else
                           "no PN entry here")]
    others = []
    for f in inside:
        if f is best or f in simbad or f["name"] in [o.split(" (")[0] for o in others]:
            continue
        others.append(f"{f['name']} ({f.get('status') or f['source']})")
    if others:
        lines.append("Also listed as: " + "; ".join(others[:4]))
    if best.get("sep_arcmin", 0) > known.get("zone_arcmin", 1e9) and best.get("radius_arcmin"):
        lines.insert(1, f"Your position is {best['sep_arcmin']:.1f}' from its centre, inside its catalogued size "
                        f"({best['entry'].get('size')})")
    for f in inside:
        if f.get("entry"):
            lines.append(quote_entry(f["entry"]))
    title = f"{best['name']} ({cls})"
    return title, lines


def quote_entry(e):
    """The planetarynebulae.net entry exactly as the site gives it (Session 19b)."""
    c = e["coord"]
    pos = f"{c.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)} " \
          f"{c.dec.to_string(sep=':', precision=1, alwayssign=True, pad=True)}"
    parts = [e["name"], e.get("pn_g") or "", pos, e.get("size") or "no size given",
             (e.get("status_fr") or "no status") + (f" ({e['status']})" if e.get("status_fr") else "")]
    return "planetarynebulae.net entry: " + " | ".join(p for p in parts if p) + \
        (" | marked * on the site" if e.get("unpublished") else "")


def nearest_known(known):
    """The closest catalogued object outside the zone (within NEARBY_ARCMIN), for 'near StDr 47' messages."""
    near = [f for f in known.get("nearby") or [] if f["sep_arcmin"] <= NEARBY_ARCMIN
            and f.get("category") in ("KNOWN PN", "KNOWN CANDIDATE", "PREVIOUSLY REJECTED")]
    if not near:
        return None
    f = min(near, key=lambda g: g["sep_arcmin"] - (0.5 if g.get("entry") else 0))   # same object: prefer the list's name
    cls = f.get("status") or {"KNOWN PN": "known PN", "KNOWN CANDIDATE": "known PN candidate"}.get(f["category"], "listed")
    size = f" (its catalogued size {f['entry']['size']})" if f.get("entry") and f["entry"].get("size") else ""
    return dict(name=f["name"], cls=cls, sep=f["sep_arcmin"], source=f["source"], entry=f.get("entry"),
                text=f"{f['name']} ({cls}), {f['sep_arcmin']:.1f}' from your position, outside the object{size}; "
                     f"listed in {f['source']}")


def summarise(result, limit=3):
    def fmt(items):
        return "; ".join(f"{i['source']}: {i['name']} ({i['sep_arcmin']:.2f}')" for i in items[:limit])
    return fmt(result["inside"]), fmt(result["nearby"])
