"""
hydro_common.py - helpers shared by build_manifest.py, resize_cache.py and make_splits.py.

Nothing here touches images. It covers:
  - loading hydro_config.yaml (paths inside it are relative to the config file's folder)
  - reading/writing JSON safely (writes go to a temp file first, so a crash never leaves a half-written file)
  - fingerprints (SHA-256 hashes) that prove two runs used identical data
  - taxonomy lookup: what happens to one original class (map / ignore / background)
  - a small report printer that prints and also saves to a text file
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required:  pip install pyyaml")

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "hydro_config.yaml"

ROLES = ("POOL_L", "POOL", "L_EXTRA", "TEST", "CONTROL", "EXCL")
TRAIN_ROLES = {"POOL_L", "POOL", "L_EXTRA"}
EVAL_ROLES = {"TEST", "CONTROL"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def fail(msg):
    """Stop the script with a readable message (no traceback)."""
    raise SystemExit(f"\nERROR: {msg}\n")


# ------------------------------------------------------------------ config
def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        fail(f"config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = Path(str(cfg.get("base", ".")))
    if not base.is_absolute():
        base = (path.resolve().parent / base).resolve()
    if not base.exists():
        fail(f"base folder from config does not exist: {base}")
    cfg["_base"] = base
    cfg["_config_path"] = path.resolve()
    return cfg


def config_snapshot(cfg):
    """Config without the private keys, for storing inside output files."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def resolve(base, rel_posix):
    """Relative path stored in JSON (forward slashes) -> absolute path on this machine."""
    return Path(base) / Path(rel_posix)


def to_posix_rel(path, base):
    return os.path.relpath(path, base).replace(os.sep, "/")


# ------------------------------------------------------------------ json io
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj, path, pretty=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        if pretty:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, fh, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------ fingerprints
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_ids(ids):
    return "sha256:" + sha256_text(",".join(str(i) for i in sorted(ids)))


def fingerprint_obj(obj):
    return "sha256:" + sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ taxonomy
def load_taxonomy(path):
    tax = load_json(path)
    for key in ("version", "unified_classes", "map"):
        if key not in tax:
            fail(f"taxonomy {path} is missing '{key}'")
    tax.setdefault("ignore", {})
    tax.setdefault("background", {})
    tax.setdefault("tiers", {})
    if len(set(tax["unified_classes"])) != len(tax["unified_classes"]):
        fail("taxonomy unified_classes contains duplicates")
    bad = []
    for src, mm in tax["map"].items():
        for name, tgt in mm.items():
            if tgt not in tax["unified_classes"]:
                bad.append(f"{src}.{name} -> '{tgt}'")
    if bad:
        fail("taxonomy maps to classes not listed in unified_classes: " + ", ".join(bad))
    tiered = [c for cs in tax["tiers"].values() for c in cs]
    if tax["tiers"]:
        missing = [c for c in tax["unified_classes"] if c not in tiered]
        stray = [c for c in tiered if c not in tax["unified_classes"]]
        if missing or stray:
            fail(f"taxonomy tiers mismatch - untiered: {missing}, unknown: {stray}")
    return tax


def tier_of(tax, cls):
    for tier, members in tax.get("tiers", {}).items():
        if cls in members:
            return tier
    return None


def resolve_class(tax, source, name):
    """
    What happens to annotations of original class `name` from `source`:
      ("background", None)  annotation dropped; pixels become plain background
      ("ignore", None)      box becomes a don't-care zone on its image (no class)
      ("map", k)            becomes unified class number k (1-based, order of unified_classes)
      (None, None)          not covered by the taxonomy -> caller must stop
    """
    if name in tax["background"].get(source, []):
        return "background", None
    if name in tax["ignore"].get(source, []):
        return "ignore", None
    tgt = tax["map"].get(source, {}).get(name)
    if tgt is not None:
        return "map", tax["unified_classes"].index(tgt) + 1
    return None, None


# ------------------------------------------------------------------ report
class Report:
    """Print a line and keep it, so the whole run can be saved to a text file."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def section(self, title):
        self("\n" + "=" * 86 + f"\n{title}\n" + "=" * 86)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")
