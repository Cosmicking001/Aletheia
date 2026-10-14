"""
aletheia_ranking: Session 11, testing and learning the central-star ranking

build_test_set(...)   spectroscopically confirmed central stars (Weidmann et al. 2020) placed in their
                      nebulae (centre and size from SIMBAD), with the centre shifted by a fraction of the radius.
pn_sizes()            known PNe with a catalogued size (SIMBAD). Session 12d: replaces the Chornay & Walton
                      list, whose "radius" was being read from its RA column (loose column-name match).
evaluate(...)         how often the confirmed star ranks #1 / top 3 among Gaia-detected cases.
learn(...)            logistic regression on the features, 5-fold cross-validation grouped by PN.
"""
import math

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

import aletheia_cspn as cs

WEIDMANN = "J/A+A/640/A10/catalog"


def _all_rows(catalog, getter=None):
    if getter:
        return getter(catalog)
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=-1)
    v.TIMEOUT = 300
    return v.get_catalogs(catalog)


def pn_sizes(getter=None):
    """Known PNe with a catalogued size: SIMBAD basic table, otype PN, galdim_majaxis (arcmin).
    Returns a list of (SkyCoord, radius_arcsec, name)."""
    if getter:
        return getter("SIMBAD_PN_SIZES")
    from astroquery.simbad import Simbad
    q = "SELECT main_id, ra, dec, galdim_majaxis FROM basic WHERE otype = 'PN' AND galdim_majaxis > 0"
    try:
        t = Simbad.query_tap(q, maxrec=20000)
    except TypeError:   # older astroquery without maxrec
        t = Simbad.query_tap(q)
    cols = {c.lower(): c for c in t.colnames}
    out = []
    for row in t:
        ra, de, maj = (cs._float(row[cols[k]]) for k in ("ra", "dec", "galdim_majaxis"))
        if None not in (ra, de, maj):
            out.append((SkyCoord(ra * u.deg, de * u.deg), 30.0 * maj, " ".join(str(row[cols["main_id"]]).split())))
    return out


def build_test_set(n_max=100, shift_frac=0.15, seed=11, getter=None):
    """Returns (cases, notes). Each case: name, true_coord, centre (shifted), size_arcmin."""
    rng = np.random.default_rng(seed)
    w = _all_rows(WEIDMANN, getter)
    wt = w[list(w.keys())[0]]  # first table (works for astroquery TableList and plain dicts)
    pns = pn_sizes(getter)  # (coord, radius_arcsec, name)
    notes = [f"Weidmann central stars: {len(wt)}; PNe with centre and size (SIMBAD): {len(pns)}",
             "Weidmann columns: " + ", ".join(wt.colnames)]
    if not pns:
        return [], notes + ["no PN radii found in the PN list"]
    pc = SkyCoord([p[0] for p in pns])
    true = SkyCoord(np.array(wt["_RAJ2000"], float) * u.deg, np.array(wt["_DEJ2000"], float) * u.deg)
    idx, sep, _ = true.match_to_catalog_sky(pc)
    cases = []
    for i in range(len(wt)):
        c, rad, name = pns[idx[i]]
        if sep[i].arcsec > max(30.0, rad) or rad < 3.0 or rad > 600:
            continue
        ang = rng.uniform(0, 2 * math.pi)
        centre = c.directional_offset_by(ang * u.rad, shift_frac * rad * u.arcsec)
        cases.append(dict(name=name or f"Weidmann row {i}", true_coord=true[i], centre=centre,
                          size_arcmin=2 * rad / 60.0))
    rng.shuffle(cases)
    notes.append(f"usable (radius 3-600 arcsec, central star inside the nebula): {len(cases)}; tested: "
                 f"{min(n_max, len(cases))}")
    return cases[:n_max], notes


MATCH_ARCSEC = 2.5


def run_case(case, querier=None):
    """The answer is accepted only if ONE Gaia star matches the confirmed position unambiguously:
    the nearest within 2.5 arcsec, and no other star nearly as close (second nearest at least
    4 arcsec and at least twice as far)."""
    res = cs.find_central_stars(case["centre"], case["size_arcmin"], querier)
    stars = res["stars"]
    truth, ambiguous = None, False
    if stars:
        seps = np.array([s["coord"].separation(case["true_coord"]).arcsec for s in stars])
        order = np.argsort(seps)
        d1 = seps[order[0]]
        d2 = seps[order[1]] if len(order) > 1 else np.inf
        if d1 <= MATCH_ARCSEC:
            if d2 >= max(4.0, 2 * d1):
                truth = stars[order[0]]
            else:
                ambiguous = True
    return dict(case=case, stars=stars, truth=truth, ambiguous=ambiguous, truncated=res.get("truncated", False))


def ranks(results, scorer):
    """Rank of the true star under a scoring function (higher = better). None if not in the Gaia list."""
    out = []
    for r in results:
        if r["truth"] is None or not r["stars"]:
            out.append(None)
            continue
        sc = scorer(r)
        order = np.argsort(-sc)
        pos = [i for i, s in enumerate(r["stars"]) if s is r["truth"]][0]
        out.append(int(np.where(order == pos)[0][0]) + 1)
    return out


def formula_scores(r):
    return np.array([s["parts"]["centrality"] * 0.5 + s["parts"]["blueness"] * 0.3 + s["parts"]["p_wd"] * 0.2
                     for s in r["stars"]])


def why_missing(results):
    """For confirmed central stars not in the Gaia list: outside the search circle, small position
    mismatch (a Gaia star 1.5-5 arcsec away), or no Gaia star within 5 arcsec (likely too faint)."""
    out = dict(outside_search=0, ambiguous=0, position_mismatch=0, too_faint_or_blended=0, list_cut_off=0)
    for r in results:
        if r["truth"] is not None:
            continue
        c = r["case"]
        if r.get("ambiguous"):
            out["ambiguous"] += 1
        elif r.get("truncated"):
            out["list_cut_off"] += 1
        elif c["true_coord"].separation(c["centre"]).arcsec > cs.search_radius_arcsec(c["size_arcmin"]):
            out["outside_search"] += 1
        elif any(MATCH_ARCSEC < s["coord"].separation(c["true_coord"]).arcsec <= 5 for s in r["stars"]):
            out["position_mismatch"] += 1
        else:
            out["too_faint_or_blended"] += 1
    return out


def summary(rk):
    tested = [x for x in rk if x is not None]
    n = len(tested)
    if n == 0:
        return dict(n=0, top1=0.0, top3=0.0, missing=len(rk))
    return dict(n=n, top1=sum(x == 1 for x in tested) / n, top3=sum(x <= 3 for x in tested) / n,
                missing=len(rk) - n)


def learn(results, folds=5, seed=11):
    """Logistic regression, cross-validated by PN. Returns (cv_ranks, final_model)."""
    usable = [r for r in results if r["truth"] is not None and len(r["stars"]) > 1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(usable))
    fold_of = {id(usable[i]): k % folds for k, i in enumerate(order)}

    cols = [cs.FEATURES.index(f) for f in cs.LEARN_FEATURES]

    def xy(rs):
        X, y = [], []
        for r in rs:
            F = cs.features(r["stars"], r["case"]["size_arcmin"])[:, cols]
            X.append(F)
            y.append(np.array([1.0 if s is r["truth"] else 0.0 for s in r["stars"]]))
        return np.vstack(X), np.concatenate(y)

    def fit(rs):
        X, y = xy(rs)
        mean, std = X.mean(0), X.std(0)
        std[std == 0] = 1.0
        Z = (X - mean) / std
        w, b = np.zeros(Z.shape[1]), 0.0
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1)  # balance the rare true stars
        for _ in range(3000):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            wt = np.where(y == 1, pos_w, 1.0)
            g = (p - y) * wt
            w -= 0.05 * (Z.T @ g / len(y) + 0.01 * w)
            b -= 0.05 * g.mean()
        return dict(type="logistic", features=cs.LEARN_FEATURES, coef=w.tolist(), intercept=float(b),
                    mean=mean.tolist(), std=std.tolist())

    cv_rank = {}
    for k in range(folds):
        train = [r for r in usable if fold_of[id(r)] != k]
        test = [r for r in usable if fold_of[id(r)] == k]
        if not train or not test:
            continue
        m = fit(train)
        for r, rk in zip(test, ranks(test, lambda rr: cs.model_probability(
                cs.features(rr["stars"], rr["case"]["size_arcmin"]), m))):
            cv_rank[id(r)] = rk
    cv = [cv_rank.get(id(r)) if r["truth"] is not None and len(r["stars"]) > 1 else
          (1 if r["truth"] is not None else None) for r in results]
    return cv, fit(usable)
