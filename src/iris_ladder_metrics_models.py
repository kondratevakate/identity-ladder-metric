"""Learned iris segmentation for the iris metric panel (preprocessing P2, docs/PROTOCOL_iris_metrics.md).

Two published models, run on CPU, weights read from the private model folder and checked by sha256:
  * Notre Dame CVRL (OpenSourceIrisRecognition, HDBIF/Python): NestedSharedAtrousResUNet iris-texture mask (iris
    pixels not occluded by eyelids, lashes or reflections) and a ResNet-18 regressor of the pupil and limbus
    circles. Trained on NIR and visible-light sets (incl. UBIRIS v2). Preprocessing as their `hdbif.py`: red channel
    of colour images (grey otherwise), padded with grey 127 to 4:3, resized to 640 x 480, then 320 x 240 for the
    networks, normalised (x - 0.5) / 0.5. The network class is imported from their `network.py` by explicit path.
  * Worldcoin open-iris (Hugging Face Worldcoin/iris-semantic-segmentation): UNet++ / MobileNetV2 ONNX, per-pixel
    probabilities of eyeball, iris, pupil and eyelashes, NIR only (model card). Input 640 x 480 x 3, ImageNet
    normalisation. Used as an independent check on UTIRIS NIR.
All outputs are mapped back to the original image pixel grid. No download happens here.
"""
import hashlib
import importlib.util
import os

import numpy as np
from PIL import Image

MODELS = r"C:\Projects\data\projects\identity-ladder-metric\iris-models-20261107"
CVRL_NET = os.path.join(MODELS, "cvrl-code", "methods", "HDBIF", "Python", "modules", "network.py")
CVRL_MASK = os.path.join(MODELS, "cvrl-models", "models", "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth")
CVRL_CIRC = os.path.join(MODELS, "cvrl-models", "models", "resnet18-027-0.008222-maskIoU-0.967159.pth")
WORLDCOIN = os.path.join(MODELS, "worldcoin", "iris_semseg_upp_scse_mobilenetv2.onnx")
ISO = (640, 480)
NET = (320, 240)
WC_CLASSES = ("eyeball", "iris", "pupil", "eyelashes")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def red_or_grey(path: str) -> Image.Image:
    """As CVRL hdbif.py: the red channel of colour images; grey images unchanged."""
    with Image.open(path) as im:
        if im.mode in ("RGB", "RGBA"):
            return im.getchannel("R").copy()
        return im.convert("L").copy()


def to_iso(im: Image.Image) -> tuple[Image.Image, float, float, float]:
    """Pad with grey 127 to 4:3 (centred) and resize to 640 x 480, as CVRL fix_image.
    Returns the image and (scale, pad_x, pad_y) so that original = iso / scale - pad."""
    w, h = im.size
    ar = w / h
    if abs(ar - 4 / 3) < 5e-4:
        canvas, px, py = im, 0.0, 0.0
    elif ar < 4 / 3:
        wn = int(h * 4 / 3)
        px, py = (wn - w) / 2, 0.0
        canvas = Image.new("L", (wn, h), 127)
        canvas.paste(im, (int(px), 0))
        px = float(int(px))
    else:
        hn = int(w * 3 / 4)
        px, py = 0.0, (hn - h) / 2
        canvas = Image.new("L", (w, hn), 127)
        canvas.paste(im, (0, int(py)))
        py = float(int(py))
    scale = ISO[0] / canvas.size[0]
    return canvas.resize(ISO, Image.Resampling.BILINEAR), scale, px, py


def from_iso_map(m: np.ndarray, size: tuple[int, int], scale: float, px: float, py: float, order: int = 0) -> np.ndarray:
    """Map an ISO-grid (480 x 640) array back to the original image grid (h x w)."""
    from scipy import ndimage
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return ndimage.map_coordinates(m.astype(np.float64), [(yy + py) * scale, (xx + px) * scale], order=order, mode="nearest")


class CVRL:
    def __init__(self):
        import torch
        from torchvision import models
        spec = importlib.util.spec_from_file_location("cvrl_network", CVRL_NET)
        net = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(net)
        self.torch = torch
        self.mask_model = net.NestedSharedAtrousResUNet(1, 1, width=32, resolution=(240, 320))
        self.mask_model.load_state_dict(torch.load(CVRL_MASK, map_location="cpu", weights_only=True))
        self.mask_model.eval()
        self.circle_model = models.resnet18()
        self.circle_model.avgpool = net.conv(in_channels=512, out_n=6)
        self.circle_model.fc = net.fclayer(out_n=6)
        self.circle_model.load_state_dict(torch.load(CVRL_CIRC, map_location="cpu", weights_only=True))
        self.circle_model.eval()
        self.sha = {"cvrl_mask_sha256": sha256(CVRL_MASK), "cvrl_circle_sha256": sha256(CVRL_CIRC),
                    "cvrl_network_py_sha256": sha256(CVRL_NET)}

    def _tensor(self, iso: Image.Image):
        a = np.asarray(iso.resize(NET, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return self.torch.from_numpy((a - 0.5) / 0.5)[None, None]

    def __call__(self, im: Image.Image) -> dict:
        """Iris-texture mask (bool, original grid) and circles (x, y, r) in original pixels."""
        torch = self.torch
        iso, scale, px, py = to_iso(im)
        x = self._tensor(iso)
        with torch.inference_mode():
            logit = self.mask_model(x)[0, 0].numpy()
            p = self.circle_model(x.repeat(1, 3, 1, 1))[0].numpy()
        prob = 1.0 / (1.0 + np.exp(-logit))                        # 240 x 320
        prob_iso = np.asarray(Image.fromarray(prob.astype(np.float32)).resize(ISO, Image.Resampling.BILINEAR))
        mask = from_iso_map(prob_iso, im.size, scale, px, py, order=1) > 0.5
        w, h = ISO
        diag = np.hypot(w, h)
        pupil = (p[0] * w, p[1] * h, p[2] * 0.5 * 0.8 * diag)   # CVRL circApprox, ISO pixels
        iris = (p[3] * w, p[4] * h, p[5] * 0.5 * diag)
        back = lambda c: (c[0] / scale - px, c[1] / scale - py, c[2] / scale)  # noqa: E731
        return {"mask": mask, "pupil": back(pupil), "iris": back(iris)}


class Worldcoin:
    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(WORLDCOIN, providers=["CPUExecutionProvider"])
        self.sha = {"worldcoin_onnx_sha256": sha256(WORLDCOIN)}

    def iso(self, im: Image.Image) -> tuple[dict, float, float, float]:
        """Per-class probability maps on the 640 x 480 ISO grid and the transform (scale, pad_x, pad_y):
        original = iso / scale - pad. Grid cells that are padding (outside the photograph) are NaN."""
        iso, scale, px, py = to_iso(im)
        a = np.asarray(iso, dtype=np.float32) / 255.0
        a = np.repeat(a[None], 3, 0)
        a = (a - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        out = self.sess.run(None, {"input": a[None].astype(np.float32)})[0][0]
        w, h = im.size
        yy, xx = np.mgrid[0:ISO[1], 0:ISO[0]]
        xo, yo = xx / scale - px, yy / scale - py
        photo = (xo >= 0) & (xo <= w - 1) & (yo >= 0) & (yo <= h - 1)
        return {c: np.where(photo, out[k], np.nan) for k, c in enumerate(WC_CLASSES)}, scale, px, py

    def __call__(self, im: Image.Image) -> dict:
        """Per-class probability maps on the original grid: eyeball, iris, pupil, eyelashes."""
        maps, scale, px, py = self.iso(im)
        return {c: from_iso_map(np.nan_to_num(m), im.size, scale, px, py, order=1) for c, m in maps.items()}


def fit_circle(x: np.ndarray, y: np.ndarray, trims: int = 2) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) least-squares circle, refitted after dropping points whose radial residual exceeds
    3 x the median absolute residual. Returns (cx, cy, r, share of points kept)."""
    keep = np.ones(len(x), bool)
    for _ in range(trims + 1):
        A = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
        b = x[keep] ** 2 + y[keep] ** 2
        (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = a1 / 2, a2 / 2
        r = float(np.sqrt(max(a3 + cx * cx + cy * cy, 0.0)))
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        mad = np.median(res[keep])
        keep = res <= max(3 * mad, 0.5)
    return float(cx), float(cy), r, float(keep.mean())
