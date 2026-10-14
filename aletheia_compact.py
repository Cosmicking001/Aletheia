"""
aletheia_compact: evidence for COMPACT objects (Session 13)

Two published tests, applied only when the object is compact (size at most 0.5 arcmin, or unknown):

ir_test     2MASS + WISE colours of the counterpart (AllWISE, which carries 2MASS J, H, Ks).
            Akras et al. 2019 (MNRAS 488, 3238), classification-tree criteria for compact PNe:
              W1-W4 >= 7.87 and J-H < 1.10,  or  H-W2 >= 2.24 and J-H < 0.50,  or  Ks-W3 >= 6.42 and J-H < 1.31.
            Quoted false-positive rate 10 to 15 per cent.
ha_test     H-alpha excess of the point source in the (r-i, r-Halpha) diagram, measured against the stellar
            locus of the surrounding field, as in the IPHAS emission-line catalogue (Witham et al. 2008):
            IPHAS DR2 in the north, VPHAS+ DR2 in the south. Field stars 13 < r < 19.5 with errors < 0.1 mag.

Columns are looked up by EXACT name only (no loose matching: Session 12d lesson). If a name is missing the
test is switched off with a message, never guessed.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

COMPACT_MAX_ARCMIN = 0.5
IR_MATCH_ARCSEC = 3.0
HA_MATCH_ARCSEC = 2.0
HA_FIELD_ARCMIN = 3.0
HA_MIN_STARS = 30
HA_SIGMA, HA_MIN_EXCESS = 4.0, 0.2
ALLWISE = "II/328/allwise"
HA_CATALOGUES = [  # (catalogue, name, r, i, ha, error columns)
    ("II/321/iphas2", "IPHAS DR2", ("r", "rmag"), ("i", "imag"), ("ha", "Hamag", "hamag"),
     (("rErr", "e_r", "e_rmag"), ("iErr", "e_i", "e_imag"), ("haErr", "e_ha", "e_Hamag"))),
    ("II/341/vphasp", "VPHAS+ DR2", ("rmag", "r", "r2mag"), ("imag", "i"), ("Hamag", "ha", "Ha"),
     (("e_rmag", "rErr", "e_r2mag"), ("e_imag", "iErr"), ("e_Hamag", "haErr", "e_ha"))),
]


def _exact(t, names):
    low = {c.lower(): c for c in t.colnames}
    for n in names:
        if n in t.colnames:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None


def _query(coord, catalog, radius_arcsec, querier=None):
    if querier:
        return querier(catalog, coord, radius_arcsec)
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "+_r", "_RAJ2000", "_DEJ2000"], row_limit=5000)
    v.TIMEOUT = 60
    tl = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=catalog)
    return tl[0] if len(tl) else None


def _num(t, col):
    return np.array(np.ma.filled(np.ma.asarray(t[col], dtype=float), np.nan), dtype=float)


def _sep_arcsec(t, coord):
    c = SkyCoord(_num(t, "_RAJ2000") * u.deg, _num(t, "_DEJ2000") * u.deg)
    return coord.separation(c).arcsec


def is_compact(size_arcmin):
    return size_arcmin is None or float(size_arcmin) <= COMPACT_MAX_ARCMIN


def ir_test(coord, querier=None):
    t = _query(coord, ALLWISE, IR_MATCH_ARCSEC, querier)
    if t is None or len(t) == 0:
        return dict(status="no AllWISE source within 3\"")
    cols = {k: _exact(t, (k,)) for k in ("W1mag", "W2mag", "W3mag", "W4mag", "Jmag", "Hmag", "Kmag", "qph")}
    if None in cols.values():
        return dict(status="AllWISE columns not recognised: " + ", ".join(k for k, v in cols.items() if v is None))
    sep = _sep_arcsec(t, coord)
    row = t[int(np.nanargmin(sep))]
    m = {k: float(row[c]) if not np.ma.is_masked(row[c]) else np.nan for k, c in cols.items() if k != "qph"}
    q = str(row[cols["qph"]])
    ok = {f"W{i}mag": len(q) >= i and q[i - 1] in "ABC" for i in range(1, 5)}   # U = upper limit only
    jh = m["Jmag"] - m["Hmag"]
    tests = [("W1-W4", m["W1mag"] - m["W4mag"], 7.87, ok["W1mag"] and ok["W4mag"], 1.10),
             ("H-W2", m["Hmag"] - m["W2mag"], 2.24, ok["W2mag"], 0.50),
             ("Ks-W3", m["Kmag"] - m["W3mag"], 6.42, ok["W3mag"], 1.31)]
    usable = [(n, v, lim, jh_max) for n, v, lim, good, jh_max in tests if good and np.isfinite(v) and np.isfinite(jh)]
    if not usable:
        return dict(status="counterpart too faint in the needed bands (upper limits or no 2MASS)")
    hits = [f"{n} = {v:.2f} (needs >= {lim}) and J-H = {jh:.2f} (needs < {jh_max})"
            for n, v, lim, jh_max in usable if v >= lim and jh < jh_max]
    return dict(status="ok", pn_like=bool(hits), detail=hits[0] if hits else
                "colours do not meet the compact-PN criteria: " + ", ".join(f"{n} {v:.2f}" for n, v, _, _ in usable))


def ha_test(coord, querier=None):
    for cat, name, rc, ic, hc, ec in HA_CATALOGUES:
        try:
            t = _query(coord, cat, HA_FIELD_ARCMIN * 60, querier)
        except Exception:
            t = None
        if t is None or len(t) == 0:
            continue
        cols = [_exact(t, rc), _exact(t, ic), _exact(t, hc)] + [_exact(t, e) for e in ec]
        if None in cols:
            return dict(status=f"{name} columns not recognised (found: {', '.join(t.colnames[:25])})")
        r, i, ha, er, ei, eh = (_num(t, c) for c in cols)
        sep = _sep_arcsec(t, coord)
        x, y = r - i, r - ha
        good = (r > 13) & (r < 19.5) & (er < 0.1) & (ei < 0.1) & (eh < 0.1) & np.isfinite(x) & np.isfinite(y)
        tgt = np.where((sep <= HA_MATCH_ARCSEC) & good)[0]
        if len(tgt) == 0:
            return dict(status=f"{name}: no reliable point source within 2\" of the centre")
        k = tgt[np.argmin(sep[tgt])]
        field = good.copy()
        field[k] = False
        if field.sum() < HA_MIN_STARS:
            return dict(status=f"{name}: too few field stars ({int(field.sum())}) to define the stellar locus")
        keep = field.copy()
        for _ in range(5):   # robust straight-line fit to the stellar locus, outliers clipped
            b, a = np.polyfit(x[keep], y[keep], 1)
            res = y - (a + b * x)
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            keep = field & (np.abs(res) < 2.5 * max(s, 0.01))
        excess = float(res[k])
        sig = excess / np.sqrt(s ** 2 + er[k] ** 2 + eh[k] ** 2)
        return dict(status="ok", catalogue=name, excess=excess, sigma=float(sig),
                    emitter=bool(sig >= HA_SIGMA and excess >= HA_MIN_EXCESS),
                    detail=f"r-Halpha excess {excess:+.2f} mag ({sig:+.1f} sigma) above the field stellar locus "
                           f"in {name}, r = {r[k]:.2f}")
    return dict(status="no IPHAS or VPHAS+ coverage at this position")


def compact_evidence(coord, size_arcmin, querier=None):
    """Both tests, or a note saying why they do not apply."""
    if not is_compact(size_arcmin):
        return dict(applies=False, note=f"compact-object tests not used (size {size_arcmin}' > {COMPACT_MAX_ARCMIN}')")
    out = dict(applies=True)
    for key, fn in (("ir", ir_test), ("ha", ha_test)):
        try:
            out[key] = fn(coord, querier)
        except Exception as exc:
            out[key] = dict(status=f"query failed: {' '.join(str(exc).split())[:60]}")
    return out
