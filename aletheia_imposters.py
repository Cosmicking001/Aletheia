"""
aletheia_imposters: Aletheia cross-match B (Session 5)

Is this position inside or overlapping something that often imitates a PN?
H II regions (WISE H II catalogue, Sharpless), supernova remnants (Green 2024),
bright nebulae (Lynds LBN), reflection nebulae (Magakian 2003 merged list,
van den Bergh 1966), external galaxies (SIMBAD) and variable stars (VSX).

check_imposters(coord, size_arcmin=None) -> dict(primary, flags, notes)
A flag is "inside" when the position lies within the catalogued object's radius,
"overlaps" when the two extents touch. Flags are evidence for the scorecard,
not verdicts: a PN catalogue match (Session 4) always takes priority.
"""
import astropy.units as u
from astropy.coordinates import SkyCoord

import aletheia_crossmatch as xm

# Catalogues that can hold very large objects: searched out to 3 degrees
WIDE = {
    "VII/297": ("Green SNR catalogue 2024", "SNR"),
    "VII/20": ("Sharpless H II regions", "H II region"),
    "VII/9": ("Lynds Bright Nebulae", "Bright nebula (type not given)"),
}
# Catalogues of smaller objects: searched out to 30 arcmin
NARROW = {
    "J/ApJS/212/1": ("WISE catalogue of Galactic H II regions", "H II region"),
    "J/A+A/399/141": ("Magakian 2003 reflection nebulae", "Reflection nebula"),
    "VII/21": ("van den Bergh 1966 reflection nebulae", "Reflection nebula"),
}
VSX = "B/vsx"
PRIORITY = ["External galaxy", "SNR", "H II region", "Reflection nebula", "Bright nebula (type not given)"]
SIZE_COLUMNS = ["Rad", "Radius", "MajDiam", "Dmaj", "MajAxis", "Diam", "Diam1", "Size", "Dim", "D"]


def _label(key, table_map):
    for prefix, info in table_map.items():
        if key.startswith(prefix):
            return info
    return (key, "Other")


def _radius_arcmin(row, table):
    """Catalogued radius of the entry in arcmin, from its size column and unit."""
    for name in SIZE_COLUMNS:
        col = next((c for c in table.colnames if c.lower() == name.lower()), None)
        if col is None:
            continue
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            continue
        if val != val or val <= 0:  # NaN or empty
            continue
        unit = table[col].unit
        try:
            arcmin = (val * u.Unit(str(unit))).to(u.arcmin).value if unit else val
        except Exception:
            arcmin = val
        is_radius = name.lower().startswith("rad")
        return arcmin if is_radius else arcmin / 2.0
    return None


def _query(coord, table_map, radius_arcmin, row_limit):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=row_limit)
    v.TIMEOUT = 90
    try:
        lists = [v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=list(table_map))]
    except Exception:
        lists = []
        for cat in table_map:
            try:
                lists.append(v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=cat))
            except Exception:
                continue
    out = []
    for tl in lists:
        for key in tl.keys():
            t = tl[key]
            label, kind = _label(key, table_map)
            for row in t:
                try:
                    c = xm._row_coord(row)
                except Exception:
                    continue
                out.append(dict(source=label, kind=kind, name=xm._name_of(row, t),
                                sep_arcmin=coord.separation(c).arcmin, radius_arcmin=_radius_arcmin(row, t)))
    return out


def _vsx(coord, radius_arcsec=30):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=10)
    v.TIMEOUT = 60
    try:
        tl = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=VSX)
    except Exception:
        return []
    out = []
    for key in tl.keys():
        t = tl[key]
        for row in t:
            typ = str(row["Type"]).strip() if "Type" in t.colnames else "?"
            name = str(row["Name"]).strip() if "Name" in t.colnames else xm._name_of(row, t)
            out.append(f"{name} ({typ})")
    return out


def check_imposters(coord, size_arcmin=None, zone_override=None):
    zone, _ = xm.zones(size_arcmin)  # object radius + tolerance, arcmin
    if zone_override:
        zone = float(zone_override)
    found = _query(coord, WIDE, 180.0, 200) + _query(coord, NARROW, 30.0, 200)
    flags = []
    for f in found:
        r = f["radius_arcmin"]
        if r is not None and f["sep_arcmin"] <= r:
            f["relation"] = "inside"
        elif r is not None and f["sep_arcmin"] <= r + zone:
            f["relation"] = "overlaps"
        elif r is None and f["sep_arcmin"] <= zone:
            f["relation"] = "at position"
        else:
            continue
        flags.append(f)
    for g in xm.query_simbad_galaxies(coord, 30.0):
        flags.append(dict(source="SIMBAD galaxy", kind="External galaxy", name=g["name"],
                          sep_arcmin=g["sep_arcmin"], radius_arcmin=g["major_arcmin"] / 2, relation="inside"))
    flags.sort(key=lambda f: (PRIORITY.index(f["kind"]) if f["kind"] in PRIORITY else 99, f["sep_arcmin"]))
    strong = [f for f in flags if f["relation"] in ("inside", "at position")]
    primary = strong[0]["kind"] if strong else (flags[0]["kind"] + " (edge)" if flags else "none")
    return dict(primary=primary, flags=flags, variables=_vsx(coord), zone_arcmin=zone)


def summarise(result, limit=3):
    parts = []
    for f in result["flags"][:limit]:
        rad = f" r={f['radius_arcmin']:.1f}'" if f.get("radius_arcmin") else ""
        parts.append(f"{f['kind']}: {f['name']} [{f['source']}] {f['relation']}, {f['sep_arcmin']:.1f}' away{rad}")
    return "; ".join(parts)
