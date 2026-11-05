"""Python front-end of the native core: build/load the reference asset from a Key, decode crops with geostego_core.

    from stegocore import asset_for, decode, decode_multi
    a = asset_for(model.key)            # exports once (tools/export_android_ref.py) and caches under core/build_assets/
    r = decode(a, crop)                 # dict: hex, crc_ok, accept, z_pilot, amp_um, soft, H, timing_ms, ...

Build the extension first (see core/README.md): `core/build.sh`.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geostego_core as _gc  # noqa: E402

ASSET_DIR = ROOT / "core" / "build_assets"
_cache = {}


def asset_for(key_or_path, name=None):
    """Asset for a Key (or a key json path). Exported on first use, then cached on disk (keyed by the key json + exporter)."""
    from dataclasses import asdict
    if isinstance(key_or_path, (str, Path)):
        kj = json.load(open(key_or_path))
    else:
        kj = asdict(key_or_path)
        kj["font"] = str(kj["font"])
    exporter = ROOT / "tools" / "export_android_ref.py"
    h = hashlib.sha1((json.dumps(kj, sort_keys=True) + exporter.read_text()).encode()).hexdigest()[:12]
    nm = name or f"asset_{h}"
    if nm in _cache:
        return _cache[nm]
    js = ASSET_DIR / f"{nm}.json"
    if not js.exists():
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        tmp = ASSET_DIR / f"{nm}.key.json"
        # the font is stored repo-relative when possible so the exporter's Key.load resolves it
        font = Path(kj["font"])
        try:
            kj["font"] = str(font.relative_to(ROOT))
        except ValueError:
            pass
        json.dump(kj, open(tmp, "w"))
        subprocess.run([sys.executable, str(exporter), "--key", str(tmp), "--out", str(ASSET_DIR), "--name", nm], check=True,
                       stdout=subprocess.DEVNULL)
    _cache[nm] = _gc.load_asset(str(js))
    return _cache[nm]


def asset_for_model(model, name=None):
    """Asset for any Model-like object (text Model, marks.MarkModel, ...): exported in-process, cached on disk keyed by key + geometry."""
    from dataclasses import asdict
    import numpy as np
    kj = asdict(model.key)
    kj["font"] = str(kj["font"])
    hh = hashlib.sha1(json.dumps(kj, sort_keys=True).encode())
    hh.update(repr((getattr(model, "edge_w", 0.2), getattr(model, "corner_mask_mm", 0.5))).encode())
    for c in model.raw:
        hh.update(np.round(np.asarray(c), 4).tobytes())
    nm = name or f"model_{hh.hexdigest()[:14]}"
    if nm in _cache:
        return _cache[nm]
    js = ASSET_DIR / f"{nm}.json"
    if not js.exists():
        sys.path.insert(0, str(ROOT / "tools"))
        import export_android_ref as ex
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        ex.export_model(model, kj, ASSET_DIR, nm)
    _cache[nm] = _gc.load_asset(str(js))
    return _cache[nm]


def decode(asset, crop, H=None, **kw):
    """crop: 2-D uint8 or float32 array (a block crop). H: optional 3x3 ref-mm -> crop-px start/decode transform."""
    return _gc.decode(asset, crop, H, **kw)


def decode_multi(asset, crops, **kw):
    return _gc.decode_multi(asset, list(crops), **kw)


def finalize_chips(asset, chip_y, chip_w):
    """Decode from accumulated chip statistics: sum_k a_k*r_k['chip_y'], sum_k a_k*r_k['chip_w'] with a_k = 1/r_k['pilot_s2']."""
    return _gc.finalize_chips(asset, chip_y, chip_w)


def register(asset, crop, H=None, **kw):
    return _gc.register(asset, crop, H, **kw)


def decode_model(image, model, transform=None):
    """geostego.decode.decode(..., backend="cpp"): result dict with the Python decoder's keys."""
    import numpy as np
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    H = None
    if transform is not None:
        if isinstance(transform, np.ndarray) and transform.shape == (3, 3):
            H = np.asarray(transform, float)
        else:
            H = np.eye(3)
            H[:2, :2], H[:2, 2] = np.asarray(transform[0], float), np.asarray(transform[1], float)
    r = decode(asset_for(model.key), img, H)
    r["transform"] = r["H"]
    r["crc_ok"] = bool(r["crc_ok"])
    return r
