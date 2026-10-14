"""
aletheia_health: built-in health check and problem log (Session 18)

run_all(base, ...)   tests every part of Aletheia on objects with known answers (the Ring Nebula M57, StDr 47 and
                     your designated objects) and on synthetic data. Every check has its own time limit, so a
                     service that does not answer is reported as a failure instead of freezing the screen.
                     Result per check: PASS, WARN (works, but something needs your attention), FAIL, or SKIP
                     (could not be tested because an earlier check failed).
log_problem(...)     every error the interface meets is written in full to Aletheia/results/problems.log, so a
                     problem that flashed on the screen can still be found and fixed later.
"""
import datetime as dt
import json
import re
import time
import traceback
from pathlib import Path

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
M57 = "18 53 35.08 +33 01 45.0"        # Ring Nebula (SIMBAD), PN G063.1+13.9, about 1.4' across
M57_PNG, M57_SIZE = "PN G063.1+13.9", 1.4
COPY_MIN_ROWS = 1000                   # planetarynebulae.net lists about 1,300 objects; 17e saved only 729
COPY_MUST_HAVE = ("StDr 47", "MoMi 1", "PaRasMoMi 1")
LOG_NAME = "results/problems.log"
LOG_MAX_BYTES = 400_000
SEP = "=" * 70
BASE = None                            # set by set_base(); used when log_problem is called without a base


# --------------------------------------------------------------------------- problem log
def set_base(base):
    global BASE
    BASE = Path(base)


def log_problem(where, exc=None, note="", base=None):
    """Appends one problem (time, where, error, full traceback) to Aletheia/results/problems.log.
    Never raises: logging must not cause a second problem."""
    try:
        root = Path(base) if base else BASE
        if root is None:
            return
        p = root / LOG_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > LOG_MAX_BYTES:   # keep the newest half
            text = p.read_text(encoding="utf-8", errors="replace")
            p.write_text(text[len(text) // 2:], encoding="utf-8")
        lines = [SEP, f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {where}"]
        if note:
            lines.append(note)
        if exc is not None:
            lines.append(f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}")
            lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception:
        pass


def read_problems(base, days=7):
    """Problems logged in the last `days` days: list of dict(time, where, message)."""
    p = Path(base) / LOG_NAME
    if not p.exists():
        return []
    since = dt.datetime.now() - dt.timedelta(days=days)
    out = []
    for block in p.read_text(encoding="utf-8", errors="replace").split(SEP + "\n"):
        head, _, rest = block.strip().partition("\n")
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| (.*)", head)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if t >= since:
            msg = next((ln for ln in rest.splitlines() if ln.strip()), "")
            out.append(dict(time=t, where=m.group(2), message=msg[:160]))
    return out


# --------------------------------------------------------------------------- running checks
def _run(fn, ctx, budget):
    """(status, detail, seconds). The check runs in its own thread; past the budget it is abandoned."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn, ctx)
    try:
        status, detail = fut.result(timeout=budget)
    except _Skip as skip:
        status, detail = SKIP, str(skip)
    except FutureTimeout:
        status, detail = FAIL, f"no answer within {budget} s (on the screen this would have meant a long wait)"
        log_problem(f"health check: {fn.__name__}", note=detail, base=ctx.get("base"))
    except Exception as exc:
        status, detail = FAIL, f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
        log_problem(f"health check: {fn.__name__}", exc, base=ctx.get("base"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return status, detail, time.time() - t0


def _need(ctx, key, what):
    if ctx.get(key) is None:
        raise _Skip(f"not tested: {what} did not work")
    return ctx[key]


class _Skip(Exception):
    pass


def _norm(name):
    return re.sub(r"[\s\-_*]", "", str(name)).lower()


# --------------------------------------------------------------------------- the checks (offline first)
def check_packages(ctx):
    import importlib
    import sys
    if sys.version_info < (3, 9):
        return FAIL, f"Python {sys.version.split()[0]} is too old (3.9 or newer needed)"
    have, missing = [], []
    for name in ("numpy", "scipy", "astropy", "astroquery", "requests", "bs4", "matplotlib", "PIL", "ipywidgets"):
        try:
            m = importlib.import_module(name)
            have.append(f"{name} {getattr(m, '__version__', '?')}")
        except Exception:
            missing.append(name)
    if missing:
        return FAIL, "missing: " + ", ".join(missing)
    return PASS, f"Python {sys.version.split()[0]}; " + ", ".join(have)


def check_folders(ctx):
    base = Path(ctx["base"])
    missing = [s for s in ("code", "data", "config", "results") if not (base / s).is_dir()]
    if missing:
        return FAIL, "folders missing in Aletheia: " + ", ".join(missing)
    probe = base / "results" / "health" / "_write_test.txt"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok", encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == "ok"
    probe.unlink()
    if not ok:
        return FAIL, "a test file could not be written and read back"
    if not ctx.get("on_drive", True):
        return WARN, "working in Colab's temporary space (Drive did not connect): results are lost when Colab closes"
    return PASS, f"all folders present and writable ({base})"


def check_modules(ctx):
    import importlib
    names = ctx.get("modules") or []
    bad = []
    for n in names:
        try:
            importlib.import_module(n)
        except Exception as exc:
            bad.append(f"{n} ({type(exc).__name__})")
    if bad:
        return FAIL, "could not load: " + ", ".join(bad)
    import aletheia_app as app
    import aletheia_panels as pn
    notes = []
    if not getattr(app, "CHECK_BUDGET_S", None) or app.CHECK_BUDGET_S > 180:
        notes.append("catalogue time limit missing")
    if not getattr(pn, "FETCH_BUDGET_S", None) or pn.FETCH_BUDGET_S > 180:
        notes.append("survey time limit missing")
    if "3.8.2" not in app.ALADIN_JS[0][1]:
        notes.append("live view is not pinned to Aladin Lite 3.8.2")
    if notes:
        return FAIL, "; ".join(notes)
    return PASS, f"{len(names)} parts loaded; time limits {app.CHECK_BUDGET_S} s (catalogues), " \
                 f"{pn.FETCH_BUDGET_S} s (surveys); live view pinned to Aladin Lite 3.8.2"


def check_coordinates(ctx):
    import aletheia_coords as ac
    forms = [M57, "18:53:35.08 +33:01:45.0", "18h53m35.08s +33d01m45.0s", "283.396167 33.029167",
             "283,396167 33,029167", "18 53.5847 +33 01.750"]
    ref = ac.parse_position(forms[0])
    worst = 0.0
    for f in forms:
        c = ac.parse_position(f)
        worst = max(worst, ref.separation(c).arcsec)
        if ac.pn_g_name(c) != M57_PNG:
            return FAIL, f"'{f}' gave {ac.pn_g_name(c)}, expected {M57_PNG}"
    if worst > 0.5:
        return FAIL, f"the {len(forms)} ways of writing M57 disagree by up to {worst:.2f}\""
    try:
        ac.parse_position("hello world")
        return FAIL, "nonsense text was accepted as a position"
    except Exception:
        pass
    return PASS, f"{len(forms)} ways of writing M57 agree within {worst:.2f}\" and give {M57_PNG}; nonsense refused"


def _copy_rows(ctx):
    import csv
    p = Path(ctx["base"]) / "data" / "planetarynebulae_net.csv"
    if not p.exists():
        return None, p
    return list(csv.DictReader(p.open(encoding="utf-8-sig"))), p


def check_pnnet_copy(ctx):
    import aletheia_crossmatch as xm
    rows, p = _copy_rows(ctx)
    if rows is None:
        return WARN, "no planetarynebulae.net copy yet: recent candidates are not checked (run this session again)"
    names = {_norm(r.get("name", "")): r for r in rows}
    missing = [n for n in COPY_MUST_HAVE if _norm(n) not in names]
    loaded = xm.load_extra(p)
    ctx["copy_rows"] = rows
    found = ", ".join(f"{n} ({names[_norm(n)].get('status', '')})" for n in COPY_MUST_HAVE if n not in missing)
    detail = f"{len(rows)} objects saved, {loaded} loaded"
    bad = garbled(rows)
    if bad:
        return WARN, detail + f"; {len(bad)} rows look misread (for example name '{bad[0].get('name', '')[:40]}', " \
                              f"status '{bad[0].get('status', '')[:40]}'): run this session again"
    if loaded < 0.95 * len(rows):
        return WARN, detail + f": {len(rows) - loaded} rows have positions Aletheia cannot read"
    if len(rows) < COPY_MIN_ROWS or missing:
        return WARN, detail + (f"; fewer than {COPY_MIN_ROWS}" if len(rows) < COPY_MIN_ROWS else "") + \
            (f"; NOT in the copy: {', '.join(missing)}" if missing else "")
    return PASS, detail + f"; found {found}"


MENU_WORDS = ("Rubriques", "Accueil", "Planetary Nebulae", "Coord. gal", "Statut", "Catalogue")


def garbled(rows):
    """Rows whose name or status contains menu or heading text, or is far too long (the Session 18 misreading)."""
    return [r for r in rows if len(str(r.get("name", ""))) > 30 or len(str(r.get("status", ""))) > 40
            or any(w in f"{r.get('name', '')} {r.get('status', '')}" for w in MENU_WORDS)
            or str(r.get("name", "")).rstrip().endswith("*")                         # Session 18c: "StDr 47 *"
            or "d'arc" in str(r.get("status", "")) or "|" in str(r.get("status", ""))]   # size left in the status


def check_designated(ctx):
    import csv
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rows = _need(ctx, "copy_rows", "the planetarynebulae.net copy")
    p = Path(ctx["base"]) / "data" / "designated_objects.csv"
    if not p.exists():
        return SKIP, "no designated_objects.csv in Aletheia/data"
    des = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    if not des or not rows:
        return SKIP, "nothing to compare"
    pos = []
    for r in rows:
        try:
            pos.append((float(r["ra_deg"]), float(r["dec_deg"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not pos:
        return WARN, "no readable positions in the copy"
    cat = SkyCoord([p[0] for p in pos] * u.deg, [p[1] for p in pos] * u.deg)
    lost = []
    for d in des:
        c = SkyCoord(float(d["ra_deg"]) * u.deg, float(d["dec_deg"]) * u.deg)
        _, sep, _ = c.match_to_catalog_sky(cat)
        if sep.arcsec[0] > 2.0:
            far = sep.arcsec[0]
            lost.append(f"{d['name']} (nearest entry {far:.0f}\" away)" if far < 120 else d["name"])
    if lost:
        return WARN, f"{len(des) - len(lost)}/{len(des)} of your designated objects found; missing: " + ", ".join(lost)
    return PASS, f"all {len(des)} of your designated objects are in the copy within 2\""


def check_pnnet_reader(ctx):
    """A page laid out like the real site (menu table, headings, linked names, bullets and <br> lines)."""
    import aletheia_pnnet as pp
    nav = ("<table><tr><td><a href='index.php'>Planetary Nebulae</a> <a href='r.php'>Rubriques</a> "
           "<a href='index.php'>Accueil</a> <a href='t.php'>Tables I, II, III</a></td></tr></table>")
    html = ("<html><body>" + nav + "<h2>Catalogue StDr</h2><p><b>Nom</b> <b>Coord. gal.</b> <b>AD</b> <b>DEC</b> "
            "<b>Dimensions</b> <b>Statut</b></p><ul>"
            "<li><a href='page_np.php?id=901'>StDr 47</a> PN-G 116.6+05.3 23:29:15.00 +66:54:00.00 "
            "12 x 9 min d'arc Vraie NP</li>"
            "<li><a href='page_np.php?id=903'>StDr 49</a> PN-G 118.0\u221202.0 00:10:10.00 "
            "\u221201\u00a002\u00a003,0 0,2 NP Probable</li></ul>"
            "<p>Nom Coord. gal. AD DEC Dimensions Statut</p>"
            "<a href='page_np.php?id=1201'>MoMi 1</a> PN-G 146.7+03.1 04:02:11.07 +56:54:28.93 0,1 min d'arc "
            "Nouvelle candidate<br><a href='page_np.php?id=1300'>PaRasMoMi 1</a> PN-G 100.1+02.2 21:58:00.00 "
            "+57:00:00.00 30<br><a href='page_np.php?id=1400'>Mul-IR 1</a> PN-G 010.0+01.0 18:00:00.00 "
            "-20:00:00.00 - min d'arc<br><a href='page_np.php?id=1401'>Pre 1 *</a> PN-G 011.0+01.0 18:10:00.00 "
            "-19:00:00.00 0.2 x 0. 3 min d'arc | Nouvelle candidate</body></html>")
    rows = pp.parse_page(html)
    got = {r["name"]: r for r in rows}
    want = {"StDr 47": (352.3125, 66.9, "Vraie NP"), "StDr 49": (2.541667, -1.034167, "NP Probable"),
            "MoMi 1": (60.546125, 56.908036, "Nouvelle candidate"), "PaRasMoMi 1": (329.5, 57.0, ""),
            "Mul-IR 1": (270.0, -20.0, ""), "Pre 1": (272.5, -19.0, "Nouvelle candidate")}
    wrong = [n for n, (ra, dec, st) in want.items() if n not in got or abs(got[n]["ra_deg"] - ra) > 1e-4
             or abs(got[n]["dec_deg"] - dec) > 1e-4 or got[n]["status"] != st]
    if wrong or len(rows) != len(want):
        return FAIL, f"the page reader misread a test page: read {sorted(got)}, wrong or missing {wrong}"
    if got["Pre 1"].get("unpublished") != "yes" or got["Pre 1"]["size"] != "0.2 x 0. 3 min d'arc":
        return FAIL, f"the page reader misread the star mark or the size: {got['Pre 1']}"
    return PASS, ("the page reader reads linked names, star marks, odd sizes, menus, headings, bullets, line breaks "
                  "and decimal commas correctly")


def _synthetic_image(path, n_stars=60, seed=3):
    """A 600 x 400 star field with a known solution in its header (2"/px, rotated 30 deg, centred on M57)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    import aletheia_coords as ac
    rng = np.random.default_rng(seed)
    c = ac.parse_position(M57)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [c.ra.deg, c.dec.deg]
    w.wcs.crpix = [300.5, 200.5]
    s, a = 2.0 / 3600, np.radians(30.0)
    w.wcs.cd = [[-s * np.cos(a), s * np.sin(a)], [s * np.sin(a), s * np.cos(a)]]
    yy, xx = np.mgrid[0:400, 0:600]
    img = rng.normal(100.0, 3.0, (400, 600))
    xs, ys = rng.uniform(20, 580, n_stars), rng.uniform(20, 380, n_stars)
    for x, y, f in zip(xs, ys, rng.uniform(300, 3000, n_stars)):
        img += f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2))
    fits.PrimaryHDU(img.astype("float32"), header=w.to_header()).writeto(path, overwrite=True)
    ra, dec = w.all_pix2world(xs, ys, 0)
    return w, ra, dec


def check_plate_solve(ctx):
    """The plate-solving path from a file that already holds its solution (no internet needed)."""
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import aletheia_upload as up
    folder = Path(ctx["base"]) / "results" / "health"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "synthetic_field.fits"
    w, ra, dec = _synthetic_image(path)
    stars = SkyCoord(ra * u.deg, dec * u.deg)
    sol = up.plate_solve(path, None, progress=lambda m: None, gaia=lambda c, r, g: stars)
    x = np.array([10.0, 300.0, 590.0])
    y = np.array([10.0, 200.0, 390.0])
    back = sol["to_pixel"](*sol["to_world"](x, y))
    trip = float(np.max(np.hypot(back[0] - x, back[1] - y)))
    true_ra, true_dec = w.all_pix2world(x, y, 0)
    got_ra, got_dec = sol["to_world"](x, y)
    err = float(np.max(SkyCoord(true_ra * u.deg, true_dec * u.deg).separation(
        SkyCoord(got_ra * u.deg, got_dec * u.deg)).arcsec))
    ch = sol["check"]
    problems = []
    if abs(sol["scale"] - 2.0) > 0.01:
        problems.append(f"scale {sol['scale']:.3f}\"/px (expected 2.000)")
    if err > 0.05:
        problems.append(f"positions off by {err:.2f}\"")
    if trip > 0.01:
        problems.append(f"sky -> pixel -> sky round trip off by {trip:.3f} px")
    if ch.get("median") is None or ch["median"] > 1.0:
        problems.append(f"star check: {ch.get('note') or ch}")
    if problems:
        return FAIL, "; ".join(problems)
    ctx["solution"] = sol
    return PASS, (f"synthetic image: scale {sol['scale']:.3f}\"/px, positions exact to {err:.3f}\", round trip "
                  f"{trip:.4f} px, {ch['n']} stars matched with typical offset {ch['median']:.2f}\"")


def check_quick_solve(ctx):
    """Session 20: the seconds-fast solve on a synthetic mirrored, rotated field (no internet needed)."""
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    import aletheia_coords as ac
    import aletheia_upload as up
    rng = np.random.default_rng(5)
    c = ac.parse_position(M57)
    n = 3000
    r = 0.9 * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    cat = c.spherical_offsets_by(r * np.cos(t) * u.deg, r * np.sin(t) * u.deg)
    g = rng.uniform(0, 1, n) ** 0.4 * 8 + 10
    W_, H_, s = 1200, 800, 2.0 / 3600
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [c.ra.deg, c.dec.deg]
    w.wcs.crpix = [W_ / 2, H_ / 2]
    a = np.radians(127.0)
    w.wcs.cd = [[s * np.cos(a), -s * np.sin(a)], [s * np.sin(a), s * np.cos(a)]]   # mirrored (no minus on RA)
    x, y = w.all_world2pix(cat.ra.deg, cat.dec.deg, 0)
    img = rng.normal(300, 8, (H_, W_))
    yy, xx = np.mgrid[-6:7, -6:7]
    for xi, yi, gi in zip(x, y, g):
        i, j = int(round(xi)), int(round(yi))
        if 6 <= i < W_ - 6 and 6 <= j < H_ - 6:
            img[j - 6:j + 7, i - 6:i + 7] += 10 ** (-0.4 * (gi - 12)) * 2500 * np.exp(
                -((xx - (xi - i)) ** 2 + (yy - (yi - j)) ** 2) / (2 * 1.8 ** 2))
    folder = Path(ctx["base"]) / "results" / "health"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "synthetic_unsolved.fits"
    fits.PrimaryHDU(np.clip(img, 0, 60000).astype("float32")).writeto(path, overwrite=True)
    scales = folder / "synthetic_scales.json"
    scales.write_text(json.dumps({f"{W_}x{H_}": 2.0 * 1.06}))           # remembered scale, 6% off
    order = np.argsort(g)
    quick = lambda cen, rad, k=6000: (lambda sel: (cat.ra.deg[sel], cat.dec.deg[sel], g[sel]))(
        order[cat[order].separation(cen).deg <= rad][:k])
    hint = SkyCoord(*w.all_pix2world(0.8 * W_, 0.25 * H_, 0), unit="deg")   # a target near a corner
    t0 = time.time()
    sol = up.plate_solve(path, None, progress=lambda m: None, gaia=lambda cen, rad, gl: cat[g < gl + 2],
                         hint=hint, scales_file=scales, quick_gaia=quick)
    secs = time.time() - t0
    ra_, dec_ = sol["to_world"](np.array([10.0, 600.0, 1190.0]), np.array([10.0, 400.0, 790.0]))
    tr, td = w.all_pix2world([10.0, 600.0, 1190.0], [10.0, 400.0, 790.0], 0)
    err = float(SkyCoord(ra_ * u.deg, dec_ * u.deg).separation(SkyCoord(tr * u.deg, td * u.deg)).arcsec.max())
    if not sol["method"].startswith("quick") or err > 1.0:
        return FAIL, f"quick solve gave {sol['method']} with errors up to {err:.2f}\""
    return PASS, f"mirrored, rotated synthetic field solved in {secs:.1f} s from a rough position, errors " \
                 f"below {err:.2f}\""


def _scripts(html):
    return [s for s in re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S) if s.strip()]


def check_viewer_scripts(ctx):
    """Builds the crosshair viewer, the uploader and the live view, and checks their JavaScript for syntax
    errors with Node.js when Colab has it (a syntax error would show an empty or frozen viewer)."""
    import shutil
    import subprocess
    import tempfile
    import aletheia_app as app
    import aletheia_coords as ac
    import aletheia_upload as up
    sol = _need(ctx, "solution", "the plate-solving check")
    pages = {"crosshair viewer": up.viewer_html(sol, 12345678, known=[]),
             "uploader": up.uploader_html(),
             "live view": app.live_viewer_html(87654321, ac.parse_position(M57), 6.0,
                                               [dict(key="dss2_red", id="CDS/P/DSS2/red", label="DSS2 red",
                                                     fits=True, colour=False, stats=None)])}
    codes = {k: _scripts(v) for k, v in pages.items()}
    empty = [k for k, v in codes.items() if not v]
    if empty:
        return FAIL, "no script found in: " + ", ".join(empty)
    node = shutil.which("node")
    if not node:
        return PASS, "pages built; JavaScript not syntax-checked (Node.js is not available here)"
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, scripts in codes.items():
            for i, s in enumerate(scripts):
                f = Path(tmp) / f"s{i}.js"
                f.write_text(s, encoding="utf-8")
                r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True, timeout=30)
                if r.returncode:
                    msg = next((ln for ln in r.stderr.splitlines() if "Error" in ln), r.stderr.strip()[:120])
                    bad.append(f"{k}: {msg}")
    if bad:
        return FAIL, "JavaScript error: " + "; ".join(bad)
    return PASS, f"crosshair viewer, uploader and live view built; {sum(map(len, codes.values()))} scripts " \
                 "pass a syntax check"


def check_problem_log(ctx):
    probs = [p for p in read_problems(ctx["base"]) if not p["where"].startswith("health check")]   # those are in this report
    if not probs:
        return PASS, "no problems logged in the last 7 days"
    by = {}
    for p in probs:
        by[p["where"]] = by.get(p["where"], 0) + 1
    last = probs[-1]
    return WARN, (f"{len(probs)} problem(s) in the last 7 days ("
                  + ", ".join(f"{w} x{n}" for w, n in sorted(by.items(), key=lambda kv: -kv[1])[:4])
                  + f"); latest {last['time']:%d %b %H:%M}: {last['message']}")


# --------------------------------------------------------------------------- online checks
def check_astrometry_key(ctx):
    import requests
    import aletheia_upload as up
    key = None
    env = Path(ctx["base"]) / "config" / "aletheia.env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ASTROMETRY_NET_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        return WARN, "no astrometry.net key saved: images without a solution in the file cannot be solved"
    http = ctx.get("http") or requests
    try:
        r = http.post(f"{up.NOVA}/api/login", data={"request-json": json.dumps({"apikey": key})}, timeout=40).json()
    except Exception as exc:   # site down or busy: not a problem with the key
        return WARN, f"astrometry.net did not answer ({' '.join(str(exc).split())[:80]}): try again later"
    if r.get("status") != "success":
        return FAIL, f"astrometry.net refused the saved key: {r.get('errormessage', r)}"
    return PASS, "astrometry.net accepted the saved key (login only; nothing uploaded)"


def check_analysis(ctx):
    """The full one-position analysis on M57: catalogues, imposters, central star, compact tests."""
    import aletheia_app as app
    t0 = time.time()
    r = (ctx.get("analyse") or app.analyse)(M57, M57_SIZE, ctx.get("own"))
    secs = time.time() - t0
    ctx["result"] = r
    gaps = [k for k, bad in (("catalogues", "NOT CHECKED" in r["known"]["verdict"]),
                             ("imposters", r["imposters"].get("not_checked")),
                             ("central star", r["cspn"].get("error"))) if bad]
    if gaps:
        return FAIL, "these parts gave no answer: " + ", ".join(gaps)
    if secs > 60:
        return WARN, f"all parts answered, but slowly ({secs:.0f} s; usually under 40 s)"
    return PASS, f"all parts answered in {secs:.0f} s"


def check_known_m57(ctx):
    import aletheia_crossmatch as xm
    r = _need(ctx, "result", "the M57 analysis")
    title, _ = xm.identify(r["known"], r["official"])
    if r["known"]["verdict"] != "KNOWN PN" or not title:
        return FAIL, f"M57 was not recognised as a known PN (verdict {r['known']['verdict']}, name {title})"
    if not any(k in _norm(title + str(r["official"])) for k in ("m57", "ngc6720", "ring")):
        return WARN, f"recognised as known, but named {title}"
    return PASS, f"M57 recognised: {title}; catalogued as {r['official']}"


def check_central_star_m57(ctx):
    r = _need(ctx, "result", "the M57 analysis")
    stars = r["cspn"].get("stars") or []
    if not stars:
        return FAIL, "no Gaia stars returned at M57"
    top = stars[0]
    if top["sep"] > 2.0:
        return FAIL, f"top-ranked star is {top['sep']:.1f}\" from the centre (the M57 central star is at the centre)"
    g = f", G {top['G']:.1f}" if top.get("G") is not None else ""
    return PASS, f"top-ranked star (Gaia DR3 {top['source']}{g}) is {top['sep']:.1f}\" from the " \
                 f"centre, as expected; {len(stars)} stars ranked"


def check_known_stdr47(ctx):
    """The problem you met: StDr 47 must be named, with its planetarynebulae.net status."""
    import aletheia_crossmatch as xm
    rows = _need(ctx, "copy_rows", "the planetarynebulae.net copy")
    row = next((x for x in rows if _norm(x.get("name", "")) == _norm("StDr 47")), None)
    if row is None:
        return SKIP, "StDr 47 is not in the copy yet, so its name cannot be shown"
    import aletheia_coords as ac
    coord = ac.parse_position(f"{row['ra_deg']} {row['dec_deg']}")
    try:
        known = (ctx.get("check_known") or xm.check_known)(coord, None, ctx.get("own"))
    except Exception as exc:   # a busy catalogue service is not an Aletheia fault
        return WARN, f"SIMBAD or VizieR did not answer ({' '.join(str(exc).split())[:80]}): try again later"
    title, _ = xm.identify(known)
    if not title or "stdr47" not in _norm(title + " ".join(f["name"] for f in known.get("inside", []))):
        return FAIL, f"StDr 47 is in the copy but the known-object check said: {title or known['verdict']}"
    # Session 19b: a point well inside the nebula's catalogued size is matched to it; a point 5' away names it as nearby
    import astropy.units as u
    check = ctx.get("check_known") or xm.check_known
    major = xm.size_arcmin_of(row.get("size"))
    if not major:
        return PASS, f"StDr 47 named correctly at its centre: {title} (no catalogued size, so no inside test)"
    step = 0.6 * major / 2.0
    try:
        inner = xm.identify(check(coord.directional_offset_by(0 * u.deg, step * u.arcmin), None, ctx.get("own")))[0]
        near = xm.nearest_known(check(coord.directional_offset_by(60 * u.deg, 5.0 * u.arcmin), None, ctx.get("own")))
    except Exception as exc:
        return WARN, f"centre named ({title}); SIMBAD or VizieR did not answer for the offset points: try again later"
    if not inner or "stdr47" not in _norm(inner):
        return FAIL, f"a point {step:.1f}' inside StDr 47 was not matched to it (got {inner})"
    if not near or _norm(near["name"]) != "stdr47":
        return FAIL, f"a point 5' from StDr 47 did not name it as the nearest known object (got {near and near['name']})"
    return PASS, f"StDr 47 named correctly at its centre ({title}), {step:.1f}' inside it, and as the nearest object from 5' away"


def check_surveys(ctx):
    import numpy as np
    import aletheia_coords as ac
    import aletheia_panels as pn
    reg = ctx.get("registry") or pn.resolve_surveys(ctx.get("lister"))
    ctx["registry"] = reg
    usable = [x for x in reg if x.get("hips_id")]
    coord = ac.parse_position(M57)
    covered = pn.covering_ids(coord, 0.05, ctx.get("coverage_lister"))
    todo = [x for x in usable if x["key"] in ("dss2_red", "ps_col")]
    panel = pn.fetch_panel_parallel(coord, 6.0, todo, 400, ctx.get("fetcher"), 2)
    ok = [k for k, v in panel.items() if v["status"] == "ok" and v["data"] is not None
          and np.isfinite(np.asarray(v["data"], float)).mean() > 0.9]
    ctx["panel"] = panel
    detail = f"{len(usable)} of {len(reg)} surveys found at CDS; coverage service " \
             f"{'answered' if covered else 'did not answer'}; test images: " + \
             ", ".join(f"{k} {v['status']}" for k, v in panel.items())
    if not ok:
        return FAIL, detail
    if len(usable) < 20 or not covered or len(ok) < len(todo):
        return WARN, detail
    return PASS, detail


def check_scorecard(ctx):
    import aletheia_panels as pn
    import aletheia_scorecard as sc
    r = _need(ctx, "result", "the M57 analysis")
    panel = ctx.get("panel") or {}
    card = sc.build_scorecard(r, panel, pn.difference_images(panel), 6.0)
    ctx["card"] = card
    text = " ".join(sc.scorecard_text(card)) + sc.scorecard_html(card)
    if re.search(r"(?<!not )(?<!never )\bconfirmed\b", text, flags=re.I):
        return FAIL, "the scorecard uses the word 'confirmed'"
    head = card["best"] or "undetermined"
    return PASS, f"scorecard built for M57: {head} ({card['status']}); candidate wording only"


def check_report(ctx):
    import aletheia_app as app
    import aletheia_panels as pn
    import aletheia_report as rp
    r = _need(ctx, "result", "the M57 analysis")
    panel = ctx.get("panel") or {}
    path = Path(ctx["base"]) / "results" / "health" / "health_test_M57.pdf"
    out = rp.build_report(path, r, ctx.get("card"), panel, pn.difference_images(panel), 6.0,
                          app.summary_lines(r), app.RANK_ACCURACY, name="M57 (health check)")
    size = Path(out["path"]).stat().st_size
    if out["pages"] != 4 or size < 20_000:
        return FAIL, f"report has {out['pages']} pages and {size // 1024} KB (expected 4 pages)"
    return PASS, f"4-page report written ({size // 1024} KB, finding chart from {out['finder'] or 'no survey'}), " \
                 "saved as results/health/health_test_M57.pdf"


def check_live_view_source(ctx):
    import requests
    import aletheia_app as app
    http = ctx.get("http") or requests
    label, url = app.ALADIN_JS[0]
    r = http.get(url, timeout=30)
    if r.status_code != 200 or len(r.content) < 10_000:
        return FAIL, f"{label} could not be downloaded (HTTP {r.status_code}): the live view would stay blank"
    return PASS, f"{label} downloads correctly ({len(r.content) // 1024} KB)"


# (name, function, time limit in s, what to do if it does not pass)
CHECKS = [
    ("Python and packages", check_packages, 60, "Restart Colab (Runtime > Restart session) and run the cell again."),
    ("Drive folders", check_folders, 30, "Allow Google Drive access when Colab asks, then run the cell again."),
    ("Aletheia parts load", check_modules, 60, "Send this report to Claude: a part of the code is broken."),
    ("Coordinates", check_coordinates, 30, "Send this report to Claude."),
    ("planetarynebulae.net copy", check_pnnet_copy, 60,
     "Run this session again to refresh the copy; if it stays short, send the report to Claude."),
    ("Designated objects in the copy", check_designated, 60, "Send the report to Claude (reader needs fixing)."),
    ("Page reader self-test", check_pnnet_reader, 30, "Send this report to Claude."),
    ("Plate solving (synthetic image)", check_plate_solve, 90, "Send this report to Claude."),
    ("Quick plate solve (synthetic)", check_quick_solve, 90, "Send this report to Claude."),
    ("Viewer scripts", check_viewer_scripts, 90, "Send this report to Claude."),
    ("Problem log", check_problem_log, 20, "Send the report (and Aletheia/results/problems.log) to Claude."),
    ("astrometry.net key", check_astrometry_key, 60,
     "Check the key in Aletheia/config/aletheia.env (your key is on nova.astrometry.net > Dashboard > API)."),
    ("Full analysis of M57", check_analysis, 160,
     "Usually a busy CDS service: try again in a few minutes. If it fails twice, send the report to Claude."),
    ("Known object: M57", check_known_m57, 20, "Send this report to Claude."),
    ("Central star: M57", check_central_star_m57, 20, "Send this report to Claude."),
    ("Known object: StDr 47", check_known_stdr47, 120,
     "If a service did not answer, try again later; otherwise send this report to Claude."),
    ("Survey images", check_surveys, 130, "Usually a busy CDS service: try again later."),
    ("Evidence scorecard", check_scorecard, 60, "Send this report to Claude."),
    ("PDF report", check_report, 120, "Send this report to Claude."),
    ("Live view source", check_live_view_source, 40,
     "The Aladin Lite site may be down: the other views still work. Try again later."),
]


def run_all(base, own=None, on_drive=True, modules=None, progress=print, only=None, **inject):
    """Runs every check in order. Returns (results, summary). `inject` replaces services for offline tests."""
    set_base(base)
    ctx = dict(base=Path(base), own=own or [], on_drive=on_drive, modules=modules or [], **inject)
    results = []
    for i, (name, fn, budget, fix) in enumerate(CHECKS, 1):
        if only and name not in only:
            continue
        progress(f"health check {i}/{len(CHECKS)}: {name}...")
        status, detail, secs = _run(fn, ctx, budget)
        results.append(dict(name=name, status=status, detail=detail, seconds=round(secs, 1),
                            fix=fix if status in (FAIL, WARN) else ""))
    count = {s: sum(r["status"] == s for r in results) for s in (PASS, WARN, FAIL, SKIP)}
    summary = (f"{count[PASS]} passed, {count[WARN]} warning(s), {count[FAIL]} failed, {count[SKIP]} not tested"
               f" (total {sum(r['seconds'] for r in results):.0f} s)")
    return results, summary


def report_lines(results, summary):
    out = [f"Health check: {summary}", ""]
    for r in results:
        out.append(f"{r['status']:<4}  {r['name']} ({r['seconds']:.1f} s): {r['detail']}")
        if r["fix"]:
            out.append(f"      what to do: {r['fix']}")
    return out


def report_html(results, summary):
    colour = {PASS: "#64D2FF", WARN: "#FF9F0A", FAIL: "#FF453A", SKIP: "#8E8E93"}
    rows = "".join(
        f"<div style='padding:2px 0'><b style='color:{colour[r['status']]}'>{r['status']}</b> &nbsp;{r['name']} "
        f"<span style='color:#6E6E73'>({r['seconds']:.1f} s)</span><br><span style='padding-left:42px;color:#C7C7CC'>"
        f"{_esc(r['detail'])}</span>"
        + (f"<br><span style='padding-left:42px;color:#FF9F0A'>what to do: {_esc(r['fix'])}</span>" if r["fix"] else "")
        + "</div>" for r in results)
    return f"<div style='font-family:monospace;font-size:12px'><b>Health check: {summary}</b><br>{rows}</div>"


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save(base, results, summary):
    folder = Path(base) / "results" / "health"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = f"{dt.datetime.now():%Y%m%d_%H%M}"
    (folder / f"health_{stamp}.txt").write_text("\n".join(report_lines(results, summary)) + "\n", encoding="utf-8")
    (folder / f"health_{stamp}.json").write_text(json.dumps(dict(summary=summary, results=results), indent=1),
                                                 encoding="utf-8")
    return folder / f"health_{stamp}.txt"
