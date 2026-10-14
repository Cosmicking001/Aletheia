"""
aletheia_report: candidate report as one PDF (Session 14)

Page 1  Summary: position, names, known-object check, imposter flags, evidence scorecard, central-star
        candidates, compact-object tests, suggested follow-up. Candidate language throughout.
Page 2  Finding chart: optical survey, inverted grey, North up, East left, RA/Dec grid, scale bar,
        object circle, numbered central-star candidates.
Page 3  Survey overview (the same panel figure as the cutout download).
Page 4  Data credits and references.
Text is written only from the evidence already gathered; nothing is invented here.
"""
import datetime as dt
import tempfile
import textwrap
from pathlib import Path

import numpy as np

import aletheia_cspn as cs
import aletheia_panels as pn

FINDER_KEYS = ("dss2_red", "ps_r", "decaps_r", "legacy_r")   # first one available is used
PAGE = (8.27, 11.69)   # A4 portrait, inches
REFERENCES = [
    "Gaia Collaboration 2023, A&A 674, A1 (Gaia DR3)",
    "Bailer-Jones et al. 2021, AJ 161, 147 (distances)",
    "Gentile Fusillo et al. 2021, MNRAS 508, 3877 (white dwarf candidates)",
    "Chornay & Walton 2021, A&A 656, A110; Gonzalez-Santamaria et al. 2021, A&A 656, A51 (central stars)",
    "Akras et al. 2019, MNRAS 488, 3238 (infrared criteria for compact PNe)",
    "Witham et al. 2008, MNRAS 384, 1277 (H-alpha excess selection)",
    "Acker et al. 1992; Kohoutek 2001; Parker et al. 2006 (MASH); Le Du et al. (French amateur list)",
    "Green 2024 (SNR catalogue); Sharpless 1959; Anderson et al. 2014 (WISE H II regions); Watson et al. 2006 (VSX)",
    "Bonnarel et al. 2000 (Aladin), Fernique et al. 2015 (HiPS), CDS hips2fits; SIMBAD (Wenger et al. 2000); "
    "VizieR (Ochsenbein et al. 2000)",
]


def follow_up(result, card):
    """Suggested next steps, each tied to a gap or a flag in the evidence."""
    out = []
    lines = card["lines"] if card else []
    srcs = " ".join(l["text"] for l in lines)
    if "H-alpha excess inside the object" not in srcs and "H-alpha emitter" not in srcs:
        out.append("Deep narrowband H-alpha and [O III] imaging: the surveys do not show clear emission here, so "
                   "the nebula needs to be confirmed in at least two independent data sets.")
    stars = (result.get("cspn") or {}).get("stars") or []
    if stars:
        s = min(stars, key=lambda x: x.get("blind_rank", x.get("rank", 99)))
        g = f", G = {s['G']:.1f}" if s.get("G") is not None else ""
        out.append(f"Spectrum of the best central-star candidate Gaia DR3 {s['source']}{g}: a hot, blue continuum "
                   "would support the PN interpretation.")
    for c in (card["context"] if card else []):
        if c.startswith("environment only") or "supernova remnant" in c:
            out.append("Check for association with the larger nebula noted above (line ratios such as "
                       "[S II]/H-alpha are high in supernova remnants and low in most PNe).")
            break
    if result.get("known", {}).get("verdict", "NO MATCH") != "NO MATCH":
        out.append("A catalogued object lies inside the match zone: compare positions and sizes before reporting.")
    out.append("An optical spectrum of the nebula (H-alpha, [N II], [O III], [S II]) is needed before the object "
               "can be called a planetary nebula. Until then it remains a candidate.")
    return out


def _page_text(pdf, blocks, title):
    """Writes (heading, [lines]) blocks onto as many A4 pages as needed."""
    import matplotlib.pyplot as plt
    width, y, fig = 95, None, None

    def new_page():
        f = plt.figure(figsize=PAGE)
        f.text(0.06, 0.965, title, fontsize=12, weight="bold")
        f.text(0.06, 0.948, f"Aletheia · generated {dt.date.today()} · candidate assessment, not a confirmation",
               fontsize=7, color="#555555")
        return f, 0.92
    fig, y = new_page()
    for head, lines in blocks:
        wrapped = [w for ln in lines for w in (textwrap.wrap(ln, width, subsequent_indent="   ") or [""])]
        if y - 0.022 * (len(wrapped) + 2) < 0.04 and y < 0.85:
            pdf.savefig(fig)
            plt.close(fig)
            fig, y = new_page()
        fig.text(0.06, y, head, fontsize=9.5, weight="bold")
        y -= 0.022
        for w in wrapped:
            if y < 0.04:
                pdf.savefig(fig)
                plt.close(fig)
                fig, y = new_page()
            fig.text(0.07, y, w, fontsize=7.6)
            y -= 0.0165
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def finding_chart(pdf, result, panel, fov, title):
    """Returns the survey label used, or None if no optical survey is available."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    item = next((panel[k] for k in FINDER_KEYS if k in (panel or {}) and panel[k]["status"] == "ok"), None)
    if item is None:
        return None
    n = item["data"].shape[0]
    wcs = pn.grid_wcs(result["coord"], fov, n)
    scale = fov / n
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.12, 0.2, 0.8, 0.66], projection=wcs)
    d = item["data"]
    lo, hi = np.nanpercentile(d, [5, 99.9])
    z = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    ax.imshow(np.arcsinh(10 * z) / np.arcsinh(10), cmap="gray_r", origin="lower", interpolation="nearest")
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    ax.coords[0].set_axislabel("RA (J2000)")
    ax.coords[1].set_axislabel("Dec (J2000)")
    ax.coords.grid(color="#3388aa", alpha=0.35, linestyle=":", linewidth=0.5)
    c = n / 2 - 0.5
    if result.get("size"):
        ax.add_patch(Circle((c, c), 0.5 * float(result["size"]) / scale, fill=False, ec="#d03030", lw=1.2))
    else:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # open cross marking the position
            ax.plot([c + dx * 0.03 * n, c + dx * 0.08 * n], [c + dy * 0.03 * n, c + dy * 0.08 * n], color="#d03030", lw=1.2)
    stars = sorted((result.get("cspn") or {}).get("stars") or [], key=lambda x: x.get("blind_rank", 99))[:3]
    offsets = ((0.025, 0.02), (-0.045, 0.02), (0.025, -0.04))   # labels spread so close stars stay readable
    for s, (ox, oy) in zip(stars, offsets):
        x, y = wcs.world_to_pixel(s["coord"])
        if 0 <= x < n and 0 <= y < n:
            ax.add_patch(Circle((x, y), max(4, 0.012 * n), fill=False, ec="#e08000", lw=1.0))
            ax.text(x + ox * n, y + oy * n, str(s.get("blind_rank", "")), color="#e08000", fontsize=9, weight="bold")
    bar = 1.0 if fov <= 10 else (5.0 if fov <= 40 else 10.0)
    ax.plot([0.06 * n, 0.06 * n + bar / scale], [0.06 * n, 0.06 * n], color="black", lw=2)
    ax.text(0.06 * n, 0.085 * n, f"{bar:g}'", color="black", fontsize=10)
    ax.annotate("", xy=(0.92 * n, 0.92 * n), xytext=(0.92 * n, 0.80 * n), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.80 * n, 0.80 * n), xytext=(0.92 * n, 0.80 * n), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(0.925 * n, 0.93 * n, "N", fontsize=10, ha="center")
    ax.text(0.775 * n, 0.795 * n, "E", fontsize=10, va="center")
    fig.text(0.06, 0.95, title, fontsize=12, weight="bold")
    fig.text(0.06, 0.93, f"Finding chart · {item['label']} · field {fov:.1f}' · North up, East left", fontsize=9)
    legend = "Red: object position" + (f" and size ({result['size']:g}')" if result.get("size") else "")
    if stars:
        legend += " · Orange: central-star candidates, numbered by rank"
    fig.text(0.06, 0.15, legend, fontsize=8)
    fig.text(0.06, 0.13, f"Image: {item['credit']} · cutout via CDS hips2fits", fontsize=7, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)
    return item["label"]


def _stretch_rgb(sol):
    d = sol["disp"]
    if sol.get("rgb") is not None:
        return sol["rgb"], None
    lo, hi = np.nanpercentile(d, [5, 99.7])
    return np.arcsinh(5 * np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)) / np.arcsinh(5), "gray"


def user_image_page(pdf, user, result, title):
    """Session 20: your own plate-solved image, with the analysed position ringed, your other clicks numbered,
    catalogued objects labelled, North/East arrows, a scale bar and an enlarged view of the target.
    Returns False (no page) when the analysed position is not inside your image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    sol = user["sol"]
    H, W = sol["disp"].shape
    c = result["coord"]
    tx, ty = (float(v[0]) for v in sol["to_pixel"](c.ra.deg, c.dec.deg))
    if not (0 <= tx < W and 0 <= ty < H):
        return False
    scale = sol["scale"]
    r_px = max(10.0, (0.5 * float(result["size"]) * 60 if result.get("size") else 60.0) / scale)
    ring = 1.35                                  # ring just outside the object
    circ = user.get("circle")
    mine_circle = False
    if circ is not None:                         # your own circle at this position: ring exactly your circle
        from astropy.coordinates import SkyCoord as _SC
        import astropy.units as _u
        if _SC(circ[0] * _u.deg, circ[1] * _u.deg).separation(c).arcsec < 1.0:
            r_px, ring, mine_circle = max(3.0, circ[2] * 60 / scale), 1.0, True
    img, cmap = _stretch_rgb(sol)
    others = []
    for ra, dec in user.get("points") or []:
        x, y = sol["to_pixel"](ra, dec)
        if np.hypot(float(x[0]) - tx, float(y[0]) - ty) > 3:
            others.append((float(x[0]), float(y[0])))
    known = [(o["name"], o["x"], H - 1 - o["y"]) for o in user.get("known") or []]

    def draw(ax, x0, x1, y0, y1, lw, fs, label_known):
        ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest" if (x1 - x0) < 300 else "antialiased")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.add_patch(Circle((tx, ty), ring * r_px, fill=False, ec="#ff3b30", lw=lw * 1.4,
                            ls=(0, (5, 3)) if mine_circle else "-"))
        ax.plot([tx - (ring + 1.05) * r_px, tx - (ring + 0.2) * r_px], [ty, ty], color="#ff3b30", lw=lw)
        ax.plot([tx + (ring + 0.2) * r_px, tx + (ring + 1.05) * r_px], [ty, ty], color="#ff3b30", lw=lw)
        for n, (x, y) in enumerate(others, 1):
            ax.add_patch(Circle((x, y), max(6, 0.01 * W), fill=False, ec="#64d2ff", lw=lw * 0.8))
            ax.text(x + 0.012 * W, y + 0.012 * W, str(n), color="#64d2ff", fontsize=fs)
        used = []
        for name, x, y in known:
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            ax.add_patch(Circle((x, y), max(5, 0.008 * W), fill=False, ec="#ffd60a", lw=lw * 0.8))
            if label_known:
                yy = y - 0.02 * (y1 - y0)
                while any(abs(yy - u_) < 0.03 * (y1 - y0) and abs(x - v_) < 0.25 * (x1 - x0) for v_, u_ in used):
                    yy -= 0.03 * (y1 - y0)
                used.append((x, yy))
                ax.text(x + 0.012 * (x1 - x0), yy, name, color="#ffd60a", fontsize=fs * 0.85)
        ax.set_xticks([])
        ax.set_yticks([])

    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.965, title, fontsize=12, weight="bold")
    fig.text(0.06, 0.945, f"Your image: {Path(sol['path']).name} · {sol['method']} · {scale:.2f}\"/px on the "
                          f"display copy · field {sol['width_arcmin']:.1f}' x {sol['height_arcmin']:.1f}'",
             fontsize=8, color="#444444")
    ch = sol.get("check") or {}
    if ch.get("median") is not None:
        fig.text(0.06, 0.931, f"Plate solution checked against Gaia DR3: {ch['n']} stars, typical offset "
                              f"{ch['median']:.2f}\"", fontsize=8, color="#444444")
    asp = H / W
    w_main = 0.88
    h_main = min(0.50, w_main * asp * PAGE[0] / PAGE[1])
    ax = fig.add_axes([0.06, 0.92 - h_main, w_main, h_main])
    draw(ax, -0.5, W - 0.5, -0.5, H - 0.5, 1.2, 9, True)
    # North and East arrows, from the solution itself (works for rotated or mirrored images)
    px, py = 0.88 * W, 0.15 * H if asp < 1 else 0.12 * H
    pra, pdec = sol["to_world"]([px], [py])
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    p0 = SkyCoord(pra[0] * u.deg, pdec[0] * u.deg)
    L = 0.08 * min(W, H) * scale
    for pa, lab in ((0, "N"), (90, "E")):
        q = p0.directional_offset_by(pa * u.deg, L * u.arcsec)
        qx, qy = sol["to_pixel"](q.ra.deg, q.dec.deg)
        ax.annotate("", xy=(float(qx[0]), float(qy[0])), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.3))
        ax.text(float(qx[0]) + 0.3 * (float(qx[0]) - px), float(qy[0]) + 0.3 * (float(qy[0]) - py), lab,
                color="white", fontsize=9, ha="center", va="center")
    bar = next(b for b in (60, 30, 10, 5, 2, 1) if b * 60 / scale <= 0.25 * W) if 60 / scale <= 0.25 * W else 1
    ax.plot([0.04 * W, 0.04 * W + bar * 60 / scale], [0.05 * H, 0.05 * H], color="white", lw=2)
    ax.text(0.04 * W, 0.075 * H, f"{bar}'", color="white", fontsize=9)
    # enlarged view of the target
    half = max(4 * r_px, 80)
    inset_h = 0.40 * PAGE[0] / PAGE[1]                  # square on the page, just below the full image
    y2 = max(0.05, 0.92 - h_main - 0.06 - inset_h)
    ax2 = fig.add_axes([0.06, y2, 0.40, inset_h])
    draw(ax2, tx - half, tx + half, ty - half, ty + half, 1.6, 10, True)
    ax2.set_title(f"Enlarged: {2 * half * scale / 60:.1f}' across", fontsize=8)
    d = result["desc"]
    lines = [("Red dashed circle: the area you marked" if mine_circle else "Red ring: the position analysed in this report"),
             f"RA {d['ra_hms']}  Dec {d['dec_dms']}",
             f"{d['pn_g']} · {d['constellation']}",
             (f"Your circle: radius {r_px * scale / 60:.2f}', diameter {2 * r_px * scale / 60:.2f}'" if mine_circle else
              f"Object size {float(result['size']):g}' (ring drawn just outside it)" if result.get("size") else
              "No object size given: the ring has a 1.35' radius"),
             "Blue numbers: your other clicked points",
             "Yellow: catalogued objects in the field"]
    for i, t in enumerate(lines):
        fig.text(0.52, y2 + inset_h - 0.03 - i * 0.022, t, fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)
    return True


def overview_page(pdf, result, panel, diffs, fov, title):
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as tmp:
        png = pn.make_panel_png(panel, diffs or {}, Path(tmp) / "overview.png", title, result.get("size"),
                                pn.coverage_note(panel)[0], fov)
        img = plt.imread(png)
    fig = plt.figure(figsize=PAGE, facecolor="black")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, facecolor="black")
    plt.close(fig)


def build_report(path, result, card, panel, diffs, fov, summary_lines, accuracy=None, name=None, user_image=None):
    """Writes the PDF and returns dict(path, pages, finder) for checking."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    d = result["desc"]
    label = name or d["pn_g"]
    title = f"Aletheia candidate report: {label}"
    if card and card["best"]:
        verdict = f"{'Most likely' if card['status'] == 'decisive' else 'Leaning towards'}: {card['best']} " \
                  f"({card['status']}, lead {card['lead']}). Candidate assessment, not a confirmation."
    else:
        verdict = "Undetermined: the evidence available does not favour one explanation."
    stars = sorted((result.get("cspn") or {}).get("stars") or [], key=lambda x: x.get("blind_rank", 99))[:3]
    star_lines = [cs.describe_star(s) for s in stars] or ["No Gaia DR3 star in the search area."]
    if accuracy:
        star_lines.append(f"Ranking accuracy in tests: right star #1 in {accuracy['top1']:.0%}, top 3 in "
                          f"{accuracy['top3']:.0%} ({accuracy['n']} confirmed central stars).")
    blocks = [
        ("Position", [f"RA {d['ra_hms']}  Dec {d['dec_dms']} (J2000)",
                      f"l {d['l_deg']:.4f}  b {d['b_deg']:+.4f}  ·  {d['constellation']}",
                      f"Name from position: {d['pn_g']}" + (f"  ·  catalogued as {result['official']}"
                                                            if result.get("official") else ""),
                      f"Object size used: {result['size']:g}'" if result.get("size") else "Object size: not given"]),
        ("Known-object and imposter checks", list(summary_lines)),
        ("Evidence scorecard", [verdict] + [f"{l['pts']:+d} {l['cls']}: {l['text']} [{l['source']}]"
                                            for l in (card["lines"] if card else [])]
         + [f"note: {c}" for c in (card["context"] if card else [])]),
        ("Central-star candidates (blind ranking)", star_lines),
        ("Suggested follow-up", [f"{i}. {t}" for i, t in enumerate(follow_up(result, card), 1)]),
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _page_text(pdf, blocks, title)
        mine = False
        if user_image and user_image.get("sol") is not None:
            mine = user_image_page(pdf, user_image, result, title)
        finder = finding_chart(pdf, result, panel, fov, title)
        if panel:
            overview_page(pdf, result, panel, diffs, fov, f"{label} · {fov:.1f}' field")
        credits = sorted({p["credit"] for p in (panel or {}).values() if p.get("status") == "ok"})
        _page_text(pdf, [("Survey images", credits or ["none"]), ("Catalogues and methods", REFERENCES),
                          ("About this report", [
                              "Generated by Aletheia from public survey and catalogue data. Every statement above "
                              "comes from the listed catalogues and the evidence scorecard; physical values are "
                              "catalogue values or estimates as labelled.",
                              "Tested performance (Sessions 13 and 14, 136 known PNe never seen before): the "
                              "scorecard called about 1 in 4 of them 'PN candidate', leaned wrongly on about 1 in "
                              "25, and was never confidently wrong. 'Undetermined' is common and is not evidence "
                              "against an object; it means the public surveys are not deep enough to decide.",
                              "The object is a candidate until an optical spectrum confirms its nature."])],
                   title)
        pages = pdf.get_pagecount()
    return dict(path=str(path), pages=pages, finder=finder, user_image=mine)
