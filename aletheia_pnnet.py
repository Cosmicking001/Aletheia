"""
aletheia_pnnet: private copy of the planetarynebulae.net object lists (Session 17e)

For the owner's own testing, with the site owner informed by the user. Reads the public list pages slowly
(one request every 1.5 s, robots.txt respected), saves name, position, size and status to
Aletheia/data/planetarynebulae_net.csv, which the known-object check and the image markers then use.
Nothing is published or shared.
"""
import re
import time

import astropy.units as u
from astropy.coordinates import SkyCoord

BASE = "https://planetarynebulae.net/FR/"
SEARCH_PAGE = BASE + "page_recherche.php"
EXTRA_PAGES = [BASE + "page_np_nonpubliees.php"]
LAST_EMPTY = []   # (url, html) of pages that gave no rows in the last fetch
AGENT = "Aletheia private research tool (testing; slow polite requests)"
RA_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?$")
DEC_RE = re.compile(r"^[+\-−]?\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?$")
PNG_RE = re.compile(r"PN[- ]?G\s*\d{3}\.\d\s*[+\-−]\s*\d{2}\.\d")


def list_pages(html):
    """Every catalogue list linked from the search page."""
    ids = sorted(set(re.findall(r"page_np_resultat\.php\?id=(\d+)", html)), key=int)
    return [BASE + f"page_np_resultat.php?id={i}" for i in ids]


_SP = re.compile(r"[\u00a0\u2009\u202f\s]+")
ROW_RE = re.compile(
    r"(?P<png>PN[- ]?G\s*\d{3}[.,]\d\s*[+\-]\s*\d{2}[.,]\d)[\s|]+"
    r"(?P<ra>\d{1,2}[: ]\d{2}[: ]\d{2}(?:[.,]\d+)?)[\s|]+"
    r"(?P<dec>[+\-]?\s?\d{1,2}[: ]\d{2}[: ]\d{2}(?:[.,]\d+)?)")


def _clean(t):
    return _SP.sub(" ", t.replace("\u2212", "-").replace("\u2013", "-")).strip()


def _row(name, png, ra, dec, rest):
    ra = ra.strip().replace(" ", ":").replace(",", ".")
    dec = re.sub(r"^([+-])\s+", r"\1", dec.strip()).replace(" ", ":").replace(",", ".")
    try:
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    except Exception:
        return None
    rest = rest.strip(" |").replace(" | ", " ")
    m = re.match(r"(.*?(?:min d'arc|arcmin|sec d'arc|arcsec|\"|'))\s*(.*)$", rest)
    size, status = (m.group(1).strip(), m.group(2).strip()) if m else ("", rest)
    return dict(name=name.strip(" -:|"), pn_g=_clean(png), ra_deg=round(c.ra.deg, 6), dec_deg=round(c.dec.deg, 6),
                size=size, status=status[:60])


# Session 18b: every object name on the site is a link to its own page (page_np.php?id=N). Reading the text
# that follows each such link, up to the next one, gives exactly one row per object, whatever the layout
# (bullets, <br> lines, tables), and the id makes duplicates across lists easy to remove.
OBJ_HREF = re.compile(r"page_np\.php\?id=(\d+)")
POS_RE = re.compile(r"(?P<ra>\b\d{1,2}[: ]\d{2}[: ]\d{2}(?:[.,]\d+)?)[\s|,;]+"
                    r"(?P<dec>[+\-]?\s?\d{1,2}[: ]\d{2}[: ]\d{2}(?:[.,]\d+)?)")
PNG_ANY = re.compile(r"PN[- ]?G\s*\d{3}[.,]\d\s*[+\-]\s*\d{2}[.,]\d")
SIZE_RE = re.compile(r"^[\s|,;]*(\d+(?:[.,]\d+)?(?:\s*[x\u00d7]\s*\d+(?:[.,]\d+)?)?"
                     r"(?:\s*(?:min d.arc|sec d.arc|arcmin|arcsec|[\"'\u2032\u2033]))?)")
UNIT_RE = re.compile(r"(?:min|sec)\.? d.arc|arcmin|arcsec")
LINE_TAGS = {"br", "p", "div", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "hr"}
NOT_STATUS = ("Nom ", "Coord", "Catalogue", "Planetary Nebulae", "Rubriques", "Accueil", "Table")


def _text_after(a, limit=600):
    """Text after one object link up to the next object link; block elements become line breaks."""
    from bs4 import Comment, NavigableString, Tag
    own = {id(x) for x in a.descendants}
    parts, n = [], 0
    for el in a.next_elements:
        if id(el) in own:
            continue
        if isinstance(el, Tag):
            if el.name == "a" and OBJ_HREF.search(el.get("href", "")):
                break
            if el.name in LINE_TAGS:
                parts.append("\n")
            elif el.name in ("td", "th"):
                parts.append(" | ")
        elif isinstance(el, NavigableString) and not isinstance(el, Comment) \
                and (el.parent is None or el.parent.name not in ("script", "style")):
            parts.append(str(el))
            n += len(el)
            if n > limit:
                break
    return "\n".join(_clean(x) for x in "".join(parts).split("\n"))


def parse_links(soup):
    """One row per object link: name, PN G, RA, Dec, size, status and the site's object id."""
    rows = {}
    for a in soup.find_all("a", href=OBJ_HREF):
        oid = OBJ_HREF.search(a["href"]).group(1)
        name = _clean(a.get_text(" "))
        star = name.endswith("*")          # the site marks unpublished objects with a star (Session 18c)
        name = name.rstrip("* ").strip()
        if not name or oid in rows:
            continue
        text = _text_after(a)
        m = POS_RE.search(text)
        if not m:
            continue
        png = PNG_ANY.search(text[:m.start()])
        rest = text[m.end():].split("\n")[0]
        um = UNIT_RE.search(rest)
        if um and um.start() <= 30:        # size = everything up to its unit, even "0.2 x 0. 3 min d'arc" or "? min d'arc"
            size, status = rest[:um.end()].strip(" |,;"), rest[um.end():]
            size = size if re.search(r"\d", size) else ""   # "- min d'arc": no size given
        else:
            sm = SIZE_RE.match(rest)
            size = sm.group(1).strip() if sm else ""
            status = rest[sm.end():] if sm else rest
        status = " ".join(status.replace("|", " ").split()).strip(" ,;-")
        for w in NOT_STATUS:   # never let headings or menus leak into the status
            if w in status:
                status = status[:status.index(w)].strip(" |,;-")
        r = _row(name, png.group(0) if png else "", m.group("ra"), m.group("dec"), "")
        if r is None:
            continue
        r.update(size=size, status=status[:40], obj_id=oid, unpublished="yes" if star else "")
        rows[oid] = r
    return list(rows.values())


def parse_page(html):
    """Rows (name, PN G, RA, Dec, size, status) from one list page. Object links first (Session 18b); only a page
    with no object links falls back to the older three readings (table rows, block elements, lines)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    linked = parse_links(soup)
    if linked:
        return linked
    texts = []   # (text, where the name sits): table rows put it in the first cell
    for tr in soup.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        texts.append((" | ".join(c for c in cells if c), "first"))
    texts += [(_clean(t), "last") for t in soup.get_text("\n").split("\n")]
    texts += [(_clean(el.get_text(" ")), "last") for el in soup.find_all(["li", "p", "div"]) if not el.find(["div", "table"])]
    rows, seen = [], set()
    for t, where in texts:
        for m in ROW_RE.finditer(t):
            name = t[:m.start()].strip(" |")
            parts = [x.strip() for x in re.split(r"\s{2,}|\|", name) if x.strip()] if name else [""]
            name = parts[0] if where == "first" else parts[-1]
            words = name.split()
            if len(words) > 4:   # a block holding several rows: keep the words just before this PN G
                name = " ".join(words[-2:])
            end = t.find("PN", m.end())
            r = _row(name, m.group("png"), m.group("ra"), m.group("dec"), t[m.end(): end if end > 0 else None])
            if r is None or not r["name"]:
                continue
            key = (r["name"], round(r["ra_deg"], 4), round(r["dec_deg"], 4))
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return rows


def robots_allows(get, path="/FR/page_np_resultat.php"):
    try:
        txt = get(BASE.split("/FR/")[0] + "/robots.txt")
    except Exception:
        return True, "robots.txt not reachable (treated as allowed)"
    agent_all, blocked = False, []
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k == "user-agent":
            agent_all = v == "*"
        elif k == "disallow" and agent_all and v:
            blocked.append(v)
    hit = [b for b in blocked if path.startswith(b) or b == "/"]
    return (not hit), (f"robots.txt disallows {hit[0]}" if hit else "robots.txt allows these pages")


def fetch_all(delay=1.5, progress=print, getter=None):
    """Returns (rows, notes). getter(url) -> html is for testing; default uses requests with a clear agent."""
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = AGENT
    get = getter or (lambda url: s.get(url, timeout=60).text)
    ok, note = robots_allows(get)
    notes = [note]
    if not ok:
        return [], notes + ["stopped: the site asks automated readers not to read these pages"]
    pages = list_pages(get(SEARCH_PAGE)) + EXTRA_PAGES
    notes.append(f"{len(pages)} list pages found")
    rows, seen, failed, empty = [], set(), 0, []
    for n, url in enumerate(pages, 1):
        time.sleep(delay)
        try:
            html = get(url)
            got = parse_page(html)
        except Exception:
            failed += 1
            continue
        if not got:
            empty.append((url, html))
        for r in got:
            key = r.get("obj_id") or (r["name"], round(r["ra_deg"], 4), round(r["dec_deg"], 4))
            if key not in seen:
                seen.add(key)
                rows.append(dict(r, page=url.split("/")[-1]))
        if n % 10 == 0 or n == len(pages):
            progress(f"  planetarynebulae.net: {n}/{len(pages)} pages read, {len(rows)} objects so far")
    notes.append(f"{len(rows)} distinct objects; {failed} page(s) could not be read; {len(empty)} page(s) gave no rows")
    LAST_EMPTY[:] = empty   # raw pages with no rows, kept for checking
    return rows, notes


def save_csv(rows, path):
    import csv
    from pathlib import Path
    cols = ["name", "pn_g", "ra_deg", "dec_deg", "size", "status", "unpublished", "obj_id", "page"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
