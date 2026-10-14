"""
aletheia_scorecard: evidence scorecard (Session 12, rules revised in 12c)

12c changes (written down before the fresh test): imposter catalogues are known to contain real PNe
(Sharpless: Sh 2-188, Abell 21; WISE H II: NGC 6302), so a catalogue label now counts +1 with no PN
penalty; inside a catalogued galaxy, H II and reflection entries belong to that galaxy; "decisive"
needs a lead of 2 AND support from at least two independent sources.

Turns the evidence already gathered (imposter catalogues, central-star candidates, H-alpha minus r)
into points for each possible explanation. Every point has a written reason and a source, so the
verdict can always be traced. Rules are fixed in advance; they are NOT tuned on the test objects.

Deliberately NOT scored: whether the position is already a catalogued PN (that is the known-object
check, shown separately) and whether a catalogued central star sits there (that only says the PN is
known). The scorecard is meant for objects nobody has catalogued.

build_scorecard(result, panel, diffs, fov) -> dict(scores, lines, best, status, lead)
scorecard_html(card) -> short HTML block for the screen
"""
import numpy as np

import aletheia_panels as pn

CLASSES = ["PN candidate", "H II region", "SNR", "Reflection nebula", "Galaxy", "Young stellar object",
           "Nova shell"]
EMISSION_SIGMA = 3.0          # smoothed, star-masked H-alpha minus r excess needed to count as emission
COINCIDENT_SIZE_RATIO = 3.0   # a catalogued nebula at most 3x our object's radius "is" our object
YSO_TYPES = ("YSO", "TTS", "CTTS", "WTTS", "INS", "UXOR", "FU", "EXOR", "ORION")
COLOUR_MAX_G = 19.0   # Gaia BP/RP colours of fainter stars are unreliable (background, crowding: Riello et al. 2021)
CV_TYPES = ("UG", "NL", "CV", "NA", "NB", "NC", "NR", "ZAND", "N:")
R_KEY = {name: r_key for name, _, r_key in pn.DIFF_PAIRS}


def _add(lines, cls, pts, text, source):
    lines.append(dict(cls=cls, pts=pts, text=text, source=source))


def _object_radius_arcmin(result):
    return 0.5 * result["size"] if result.get("size") else result["imposters"].get("zone_arcmin", 1.0)


def _imposter_lines(result, lines, context):
    imp = result["imposters"]
    r_obj = _object_radius_arcmin(result)
    used = set()
    compact = bool(result.get("size")) and float(result["size"]) <= 0.5
    host = next((f["name"] for f in imp["flags"] if f["kind"] == "External galaxy" and f["relation"] == "inside"), None)
    for f in imp["flags"]:
        kind, rel = f["kind"], f["relation"]
        tag = f"{f['name']} ({f['source']}, {rel}"
        tag += f", r={f['radius_arcmin']:.1f}')" if f.get("radius_arcmin") else ")"
        if kind == "External galaxy" and rel == "inside" and "Galaxy" not in used:
            used.add("Galaxy")
            _add(lines, "Galaxy", 3, f"position lies inside the catalogued extent of galaxy {tag}", f["source"])
            _add(lines, "PN candidate", -2, "a Galactic PN seen through another galaxy is unlikely", f["source"])
        elif kind == "SNR" and rel == "inside" and compact:
            context.append(f"lies inside supernova remnant {tag}; a compact object there is most likely only "
                           "seen in projection (Session 14: not scored)")
        elif kind == "SNR" and rel == "inside" and "SNR" not in used:
            used.add("SNR")
            _add(lines, "SNR", 1, f"inside supernova remnant {tag}; filaments can look like PN arcs", f["source"])
        elif kind in ("H II region", "Reflection nebula") and host:
            context.append(f"{kind} entry {f['name']} lies inside galaxy {host}: counted as part of the galaxy")
        elif kind in ("H II region", "Reflection nebula") and kind not in used:
            r = f.get("radius_arcmin")
            same = rel in ("inside", "at position") and (r is None or r <= COINCIDENT_SIZE_RATIO * r_obj)
            if same:
                used.add(kind)
                word = "H II region" if kind == "H II region" else "reflection nebula"
                _add(lines, kind, 1, f"catalogued {word} of similar size at this position: {tag} (these catalogues "
                     "also contain some real PNe, so this is a label, not proof)", f["source"])
            else:
                context.append(f"environment only (much larger or offset): {kind} {tag}")
    for v in imp.get("variables", []):
        typ = v.rsplit("(", 1)[-1].rstrip(")").upper()
        if any(typ.startswith(t) for t in YSO_TYPES) and "YSO" not in used:
            used.add("YSO")
            _add(lines, "Young stellar object", 2, f"young variable star within 30\": {v}", "VSX")
        elif (typ == "N" or any(typ.startswith(t) for t in CV_TYPES)) and "CV" not in used:
            used.add("CV")
            _add(lines, "Nova shell", 2, f"cataclysmic variable or nova within 30\": {v}", "VSX")


def _star_lines(result, lines, context):
    c = result.get("cspn") or {}
    stars = c.get("stars") or []
    if not stars:
        context.append("no Gaia star in the central-star search area")
        return
    # the BLIND best candidate: the on-screen order puts catalogued central stars first, which would
    # leak "this PN is already known" into the evidence
    s = min(stars, key=lambda x: x.get("blind_rank", x.get("rank", 1)))
    radius_arcsec = 30.0 * (result["size"] if result.get("size") else 1.0)
    window = max(4.0, 0.30 * radius_arcsec)
    if s["sep"] > window:
        context.append(f"best central-star candidate is {s['sep']:.0f}\" from centre (outside the "
                       f"{window:.0f}\" central window): not used as evidence")
        return
    if s.get("cspn"):
        context.append("a catalogued central star sits here (the PN is probably known; not scored)")
    name = f"Gaia DR3 {s['source']}"
    if s.get("p_wd") is not None and s["p_wd"] >= 0.75:
        _add(lines, "PN candidate", 2, f"central star {name} is a likely white dwarf or hot pre-white dwarf "
             f"(P_WD {s['p_wd']:.2f})", "Gentile Fusillo et al. 2021")
    nuv_g = (s["nuv"] - s["G"]) if (s.get("nuv") is not None and s.get("G") is not None) else None
    if nuv_g is not None and nuv_g < 1.5:
        _add(lines, "PN candidate", 1, f"central star is ultraviolet-bright (NUV - G = {nuv_g:+.1f})", "GALEX")
    elif (s.get("p_wd") is None or s["p_wd"] < 0.75) and s.get("bp_rp") is not None and s["bp_rp"] < -0.2 \
            and (s.get("abs_G") is None or s["abs_G"] > -2) and s.get("G") is not None \
            and s["G"] <= COLOUR_MAX_G:
        _add(lines, "PN candidate", 1, f"central star is very blue (BP - RP = {s['bp_rp']:+.2f}) and not "
             "luminous enough to be a massive star", "Gaia DR3")
    if s.get("abs_G") is not None and s["abs_G"] < -2:
        _add(lines, "H II region", 1, f"central star is very luminous (absolute G {s['abs_G']:.1f}, no extinction "
             "correction): typical of the massive stars that light H II regions", "Gaia DR3, Bailer-Jones et al. 2021")
        _add(lines, "PN candidate", -1, "PN central stars are rarely this luminous", "Gaia DR3")
    snr = (s["plx"] / s["e_plx"]) if (s.get("plx") is not None and s.get("e_plx")) else 0.0
    if s.get("radius_pc") is not None and s["radius_pc"] > 2.0 and snr >= 5:
        _add(lines, "PN candidate", -1, f"at this star's well-measured distance the nebula would be "
             f"{s['radius_pc']:.1f} pc in radius, larger than almost all PNe", "Bailer-Jones et al. 2021")


def _emission_lines(result, panel, diffs, fov, lines, context):
    if not result.get("size"):
        context.append("H-alpha emission not measured: give an object size to measure it")
        return
    best, tried = None, 0
    for name, d in (diffs or {}).items():
        ref = (panel or {}).get(R_KEY.get(name), {})
        if ref.get("status") != "ok":
            continue
        tried += 1
        val = pn.emission_contrast_smoothed(d["data"], fov, result["size"], star_ref=ref["data"], protect_object=True)
        if np.isfinite(val) and (best is None or val > best[1]):
            best = (name, val)
    if best is None:
        context.append("emission not measured: the object is larger than the cutout (choose a wider view)"
                       if tried else "no H-alpha survey image here: emission not measured")
    elif best[1] >= EMISSION_SIGMA:
        txt = f"H-alpha excess inside the object: {best[1]:+.1f} sigma in {best[0]} H-alpha minus r"
        for cls in ("PN candidate", "H II region", "SNR"):
            _add(lines, cls, 1, txt, f"{best[0]} image")
        _add(lines, "Reflection nebula", -1, "reflection nebulae shine by scattered starlight, not H-alpha",
             f"{best[0]} image")
    else:
        context.append(f"no clear H-alpha excess ({best[1]:+.1f} sigma in {best[0]}; needs {EMISSION_SIGMA:.0f})")


def _compact_lines(result, lines, context):
    c = result.get("compact")
    if not c:
        return
    if not c.get("applies"):
        context.append(c.get("note", "compact-object tests not used"))
        return
    ir, ha = c.get("ir", {}), c.get("ha", {})
    if ir.get("status") == "ok" and ir["pn_like"]:
        _add(lines, "PN candidate", 2, "infrared colours match compact PNe (Akras et al. 2019; about 10-15% of "
             f"objects passing are not PNe): {ir['detail']}", "AllWISE + 2MASS")
    else:
        context.append("infrared test: " + (ir.get("detail") if ir.get("status") == "ok" else ir.get("status", "")))
    if ha.get("status") == "ok" and ha["emitter"]:
        _add(lines, "PN candidate", 1, f"point-like H-alpha emitter: {ha['detail']} (emission-line stars also do "
             "this)", ha["catalogue"])
    else:
        context.append("H-alpha point-source test: " + (ha.get("detail") if ha.get("status") == "ok"
                                                         else ha.get("status", "")))


def build_scorecard(result, panel=None, diffs=None, fov=None):
    lines, context = [], []
    _imposter_lines(result, lines, context)
    _compact_lines(result, lines, context)
    _star_lines(result, lines, context)
    if fov:
        _emission_lines(result, panel, diffs, fov, lines, context)
    scores = {c: 0 for c in CLASSES}
    for ln in lines:
        scores[ln["cls"]] += ln["pts"]
    order = sorted(CLASSES, key=lambda c: -scores[c])
    best, lead = order[0], scores[order[0]] - scores[order[1]]
    sources = {ln["source"] for ln in lines if ln["cls"] == best and ln["pts"] > 0}
    if scores[best] <= 0 or lead <= 0:
        status, best = "undetermined", None
    else:
        status = "decisive" if lead >= 2 and len(sources) >= 2 else "leaning"
    if result["imposters"].get("not_checked"):   # without the imposter check no verdict is safe
        status, best = "undetermined", None
        context.insert(0, "imposter catalogues could not be checked, so no verdict is given: press Analyse again")
    verdict = result.get("known", {}).get("verdict", "")
    if verdict and verdict != "NO MATCH":
        context.append(f"known-object check: {verdict} (shown above; not scored)")
    context.append("artefacts (ghosts, reflections, plate flaws) are not assessed yet")
    return dict(scores=scores, lines=lines, context=context, best=best, status=status, lead=lead)


def scorecard_html(card):
    if card["best"]:
        word = "Most likely" if card["status"] == "decisive" else "Leaning towards"
        head = (f"<b>{word}: {card['best']}</b> <span style='color:#8E8E93'>({card['status']}, lead "
                f"{card['lead']} point{'s' if card['lead'] != 1 else ''}; a candidate assessment, not a "
                f"confirmation)</span>")
    else:
        head = ("<b>Explanation: undetermined</b> <span style='color:#8E8E93'>(the evidence available does not "
                "favour one explanation)</span>")
    rows = "".join(f"<div style='padding-left:10px'><span style='color:{'#64D2FF' if l['pts'] > 0 else '#FF9F0A'}'>"
                   f"{l['pts']:+d} {l['cls']}</span>: {l['text']} <span style='color:#6E6E73'>[{l['source']}]</span></div>"
                   for l in card["lines"])
    ctx = "".join(f"<div style='padding-left:10px;color:#8E8E93'>· {c}</div>" for c in card["context"])
    return f"<div style='margin-top:12px'><b>Evidence scorecard</b><br>{head}{rows}{ctx}</div>"


def scorecard_text(card):
    """Plain-text lines for the downloaded summary."""
    if not card:
        return []
    head = (f"{'Most likely' if card['status'] == 'decisive' else 'Leaning towards'}: {card['best']} "
            f"({card['status']}, lead {card['lead']}; candidate assessment, not a confirmation)"
            if card["best"] else "Explanation: undetermined")
    return (["", "Evidence scorecard: " + head]
            + [f"  {l['pts']:+d} {l['cls']}: {l['text']} [{l['source']}]" for l in card["lines"]]
            + [f"  - {c}" for c in card["context"]])
