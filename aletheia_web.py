"""
aletheia_web: the web edition (Session 22). The same Aletheia screen, served as a web page by Voila.

start()   builds a private working folder for each visitor (deleted after a day), loads the shared, public
          starting data (logo, ranking accuracy; never private candidate lists), and shows the screen.
Bridge    an invisible widget that carries clicks on your image, drawn circles and uploaded files from the
          browser to Aletheia, where Colab used its own channel. The screen code is the same in both.
"""
import json
import shutil
import time
import uuid
from pathlib import Path

import anywidget
import traitlets

HERE = Path(__file__).resolve().parent
SEED = HERE / "seed"                         # shared starting data shipped with the web edition
ROOT = Path("/tmp/aletheia_visitors")        # one folder per visitor
KEEP_HOURS = 24
REGISTRY_DAYS = 7


class Bridge(anywidget.AnyWidget):
    """Browser -> Python calls: window.aletheiaBridge(name, args) returns a promise with the answer."""
    _esm = """
    function render({ model, el }) {
      let next = 0;
      const waiting = new Map();
      model.on("msg:custom", (m) => {
        const w = waiting.get(m.id);
        if (!w) return;
        waiting.delete(m.id);
        if (m.ok) w.res(m.result); else w.rej(new Error(m.error));
      });
      window.aletheiaBridge = (name, args) => new Promise((res, rej) => {
        const id = ++next;
        waiting.set(id, { res, rej });
        model.send({ id, name, args });
      });
      el.style.display = "none";
    }
    export default { render };
    """
    ready = traitlets.Bool(True).tag(sync=True)

    def __init__(self, handlers, **kw):
        super().__init__(**kw)
        self.handlers = handlers
        self.on_msg(self._on_msg)

    def _on_msg(self, _widget, content, _buffers):
        mid = content.get("id")
        try:
            fn = self.handlers[content.get("name")]
            out = fn(*(content.get("args") or []))
            out = out if isinstance(out, (str, int, float, bool, type(None))) else str(out)
            self.send({"id": mid, "ok": True, "result": out})
        except Exception as exc:   # the page shows the message; the problem log keeps the details
            self.send({"id": mid, "ok": False, "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"})


def _tidy(root):
    """Remove visitor folders older than KEEP_HOURS."""
    now = time.time()
    for d in root.glob("v_*"):
        try:
            if now - d.stat().st_mtime > KEEP_HOURS * 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def _registry(root):
    """The survey list (CDS MocServer), cached for everyone for a week: the page opens faster."""
    import aletheia_panels as pn
    f = root / "registry.json"
    try:
        if f.exists() and time.time() - f.stat().st_mtime < REGISTRY_DAYS * 86400:
            return json.loads(f.read_text())
    except Exception:
        pass
    reg = pn.resolve_surveys()
    try:
        f.write_text(json.dumps(reg))
    except Exception:
        pass
    return reg


def visitor_folder(root=ROOT, seed=SEED):
    root.mkdir(parents=True, exist_ok=True)
    _tidy(root)
    base = root / f"v_{uuid.uuid4().hex[:12]}"
    for sub in ("config", "data", "uploads", "results"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    if seed.exists():
        for sub in ("config", "data"):
            if (seed / sub).exists():
                shutil.copytree(seed / sub, base / sub, dirs_exist_ok=True)
    return base


WELCOME = ("<div class='al-cards'><div class='al-card wide'><h4>Welcome</h4>"
           "<div class='al-title'>Check a nebula in one step</div>"
           "Type a position (any format) and press <b>Analyse field</b>, or choose <b>Your image</b> to drop in "
           "one of your own pictures. Aletheia checks the catalogues, shows the survey images, looks for a "
           "central star and writes a cautious, candidate-only assessment you can download as a PDF."
           "<div class='al-muted' style='margin-top:8px;font-size:12px'>Early test version. Your images stay in a "
           "private folder on this server and are deleted after a day. Try M57: 18 53 35.08 +33 01 45.0</div>"
           "</div></div>")


def start(root=ROOT, seed=SEED, show=True, **app_kwargs):
    """Called by the web page's only cell. Returns the screen object (for testing)."""
    import aletheia_app as app
    import aletheia_health as hl
    base = visitor_folder(root, seed)
    hl.set_base(base)
    if "registry" not in app_kwargs:
        app_kwargs["registry"] = _registry(root)
    ui = app.AletheiaApp(base, [], **app_kwargs)
    ui.web = True
    ui.summary.value = WELCOME
    ui.mine_refresh.layout.display = "none"            # files arrive through the drop zone only
    ui.mine.children[0].value = ("<div class='al-muted' style='font-size:12px;line-height:1.5'>Drop an image of your "
                                 "own below. With a rough position (an object name or RA Dec) it is plate solved in "
                                 "seconds; then click the nebula, or draw a circle around it.</div>")
    # astrometry.net key: pasted by the visitor, kept in memory for this visit only (never saved, never in the code)
    import ipywidgets as W
    keybox = W.Password(placeholder="astrometry.net key (optional, only for this visit)",
                        layout=W.Layout(width="420px"))
    kids = list(ui.mine.children)
    kids.insert(kids.index(ui.mine_auto), keybox)
    ui.mine.children = tuple(kids)
    ui.keybox = keybox
    saved_key = ui._api_key
    ui._api_key = lambda: (keybox.value or "").strip() or saved_key()
    bridge = Bridge(ui.callbacks)
    ui.ui.children = tuple(ui.ui.children) + (bridge,)
    ui.bridge = bridge
    if show:
        from IPython.display import HTML, display
        display(HTML("<style>body, .jp-Notebook, #rendered_cells { background:#000 !important; }"
                     ".jp-OutputArea-output { background:transparent !important; }</style>"))
        display(ui.ui)
    return ui
