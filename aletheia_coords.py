"""
aletheia_coords: Aletheia coordinate core (Session 3)

parse_position(text) or parse_position(ra_text, dec_text)  -> SkyCoord (ICRS, J2000)
pn_g_name(coord, style="PN G")                              -> e.g. "PN G036.1-57.1"
describe(coord)                                             -> dict with every coordinate form

Rules
- Decimal numbers are DEGREES for both RA and Dec.
- RA written with colons, spaces or h/m/s is HOURS (sexagesimal).
- A comma between two digits with no space is a decimal comma (11,07 -> 11.07).
- PN G names use Galactic l and b TRUNCATED to one decimal (Acker et al. 1992 convention),
  with the sign of b kept even for -00.0.
"""
import math
import re

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, get_constellation

_EPS = 1e-9  # protects truncation from float noise such as 120.39999999999


class PositionError(ValueError):
    """Raised when a position cannot be read or is out of range."""


def _tokens(text, decimal_comma):
    t = re.sub(r"([+-])\s+(?=\d)", r"\1", text.strip())  # "+ 64" -> "+64"
    if decimal_comma:
        t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = re.sub(r"[,;]", " ", t)
    return t.split()


def _ra_angle(ra_text):
    ra_text = ra_text.strip()
    if re.fullmatch(r"[+]?\d+(\.\d+)?", ra_text):  # plain decimal -> degrees
        deg = float(ra_text)
        if not 0 <= deg < 360:
            raise PositionError(f"RA {deg} deg is outside 0 to 360")
        return Angle(deg, unit=u.deg)
    ra = Angle(ra_text.replace(" ", ":") if " " in ra_text else ra_text, unit=u.hourangle)
    if not 0 <= ra.hour < 24:
        raise PositionError(f"RA {ra_text} is outside 0h to 24h")
    return Angle(ra.deg, unit=u.deg)


def _dec_angle(dec_text):
    dec_text = re.sub(r"^([+-])\s+", r"\1", dec_text.strip())  # "+ 64 07" -> "+64 07"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", dec_text):
        dec = Angle(float(dec_text), unit=u.deg)
    else:
        dec = Angle(dec_text.replace(" ", ":") if " " in dec_text else dec_text, unit=u.deg)
    if not -90 <= dec.deg <= 90:
        raise PositionError(f"Dec {dec_text} is outside -90 to +90")
    return dec


def _from_tokens(tok):
    n = len(tok)
    if n == 2:
        ra_t, dec_t = tok
    elif n == 4:   # h m.m  d m
        ra_t, dec_t = " ".join(tok[:2]), " ".join(tok[2:])
    elif n == 6:   # h m s  d m s
        ra_t, dec_t = " ".join(tok[:3]), " ".join(tok[3:])
    else:
        raise PositionError(f"Could not split into RA and Dec ({n} parts found)")
    return SkyCoord(ra=_ra_angle(ra_t), dec=_dec_angle(dec_t), frame="icrs")


def parse_position(ra_text, dec_text=None):
    """Read a sky position written in almost any common format."""
    if dec_text is not None:
        fix = lambda s: re.sub(r"(?<=\d),(?=\d)", ".", str(s).strip())
        return SkyCoord(ra=_ra_angle(fix(ra_text)), dec=_dec_angle(fix(dec_text)), frame="icrs")
    errors = []
    for decimal_comma in (True, False):
        try:
            return _from_tokens(_tokens(str(ra_text), decimal_comma))
        except Exception as exc:  # try the other comma reading
            errors.append(str(exc))
    raise PositionError("; ".join(dict.fromkeys(errors)))


def _trunc1(x):
    """Truncate toward zero to one decimal, safe against float noise."""
    return math.copysign(math.floor(abs(x) * 10 + _EPS) / 10, x)


def truncation_margin_arcsec(coord):
    """How far l and b sit from the nearest 0.1 deg truncation boundary (arcsec)."""
    g = coord.galactic
    out = []
    for v in (g.l.deg % 360.0, abs(g.b.deg)):
        frac = (v * 10) % 1.0
        out.append(min(frac, 1.0 - frac) * 0.1 * 3600)
    return min(out)


def pn_g_name(coord, style="PN G"):
    g = coord.galactic
    l = g.l.deg % 360.0
    b = g.b.deg
    l_t = math.floor(l * 10 + _EPS) / 10
    if l_t >= 360.0:
        l_t = 0.0
    b_t = abs(_trunc1(b))
    sign = "-" if b < 0 else "+"
    return f"{style}{l_t:05.1f}{sign}{b_t:04.1f}"


def describe(coord):
    g = coord.galactic
    return {
        "ra_deg": round(float(coord.ra.deg), 6),
        "dec_deg": round(float(coord.dec.deg), 6),
        "ra_hms": str(coord.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)),
        "dec_dms": str(coord.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)),
        "l_deg": round(float(g.l.deg), 4),
        "b_deg": round(float(g.b.deg), 4),
        "pn_g": pn_g_name(coord),
        "pn_g_hyphen": pn_g_name(coord, style="PN-G "),
        "constellation": str(get_constellation(coord)),
    }
