"""Model-fill code (pretrained inpainting models on the ladder): synthetic arrays only, no private data, no weights."""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BC = load(os.path.join(ROOT, "gcp", "model_fills", "brain_container.py"), "brain_container")
JOB = {"shape": [197, 233, 189], "mirror_constant": 196, "size": 16}


def scan(seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.random(JOB["shape"]).astype(np.float32) + 1.0
    mask = np.zeros(JOB["shape"], bool)
    mask[19:178, 19:217, 0:162] = True          # brain box as in QTIM: taller than the 155 slices of the BraTS grid
    return np.where(mask, vol, 0).astype(np.float32), mask


@pytest.mark.parametrize("flips", [(0, 1), (), (0, 1, 2), (0,)])
@pytest.mark.parametrize("frame", [0, 1])
@pytest.mark.parametrize("start", [(40, 50, 9), (120, 150, 133), (90, 100, 70)])
def test_brain_grid_round_trip_returns_the_ladder_cube(flips, frame, start):
    """The cube cut from the BraTS array comes back exactly as the cube of the ladder frame (frame 1 = mirrored x),
    without interpolation, and the model input has that cube zeroed and nothing else changed."""
    vol, mask = scan()
    v, hide, cube = BC.case_arrays(JOB, vol, mask, start, frame, flips)
    assert v.shape == BC.BRATS and hide.sum() == 16 ** 3 and float(np.abs(v[cube]).max()) == 0.0
    full = BC.to_brats(vol, *BC.geometry(JOB, mask, start, frame, flips)[:2], flips)
    back = BC.cube_back(full[cube], frame, flips)
    sx = slice(start[0], start[0] + 16)
    truth = vol if frame == 0 else vol[JOB["mirror_constant"] - np.arange(JOB["shape"][0])]
    want = truth[sx, start[1]:start[1] + 16, start[2]:start[2] + 16]
    assert np.array_equal(back, want)
    outside = hide == 0
    assert np.array_equal(v[outside], full[outside].astype(np.float64))      # context untouched


def test_brain_z_window_keeps_the_cube_and_most_of_the_brain():
    vol, mask = scan()
    for z in (9, 60, 133):
        z0, _, cube = BC.geometry(JOB, mask, (60, 60, z), 0, (0, 1))
        assert 0 <= z0 <= 7 and 0 <= cube[2].start and cube[2].stop <= 155


def test_face_backend_hides_the_window_from_the_model():
    """A backend that returns its input shows what the model saw: inside the hidden window only the fill value
    of zeroed pixels (u = 0, z = -ZCLIP), never the target."""
    torch = pytest.importorskip("torch")
    import model_fills_infer as MI

    class Echo(MI.Base):
        def forward(self, x, m, keys):
            return x

    net = Echo("cpu", 128, 128)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(2, 128, 128)).astype(np.float32)
    hide = np.zeros_like(z, bool)
    hide[:, 40:50, 60:70] = True
    out = net(z, hide)
    assert np.allclose(out[:, 40:50, 60:70], -MI.ZCLIP)
    keep = ~hide & (np.abs(z) < MI.ZCLIP)
    assert np.allclose(out[keep], z[keep], atol=1e-5)
    assert torch.__version__


def test_model_input_does_not_depend_on_hidden_pixels():
    """Regression (17 Nov 2026): v1 upsampled the frame before masking, so the bicubic kernel carried hidden border
    values into the visible band next to the mask. Two frames that differ only inside the hidden window must give
    the model exactly the same input, in every input mode."""
    pytest.importorskip("torch")
    import model_fills_infer as MI

    class Echo(MI.Base):
        def forward(self, x, m, keys):
            self.seen = x.clone()
            return x

    rng = np.random.default_rng(2)
    z = rng.normal(size=(1, 128, 128)).astype(np.float32)
    hide = np.zeros_like(z, bool)
    hide[:, 40:50, 60:70] = True
    z2 = z.copy()
    z2[hide] = rng.normal(size=int(hide.sum())).astype(np.float32) * 2
    for mode, size in (("norm", 256), ("norm", 512), ("nearest", 256), ("nearest", 512), ("native", 128)):
        net = Echo("cpu", size, 128)
        net.input_mode = mode
        net(z, hide)
        a = net.seen
        net(z2, hide)
        assert bool((a == net.seen).all()), (mode, size)


def test_model_input_draws_no_outline_around_the_window():
    """Regression (17 Nov 2026): v2 replaced the hidden pixels by the mean of a ring before bicubic upsampling. On an
    image with a gradient that mean differs from the local value, so the visible band next to the window was pulled
    toward it by about one pixel step: a faint outline of the window, which the scene model continued into the hole
    (R 0.88 instead of 0.23 in CelebA). On a smooth ramp the visible input must stay within 0.35 of one pixel
    step of the plain upsampling of the full frame, also right next to the window (measured: 0.16 at 2x, 0.28 at 4x)."""
    torch = pytest.importorskip("torch")
    import model_fills_infer as MI

    class Echo(MI.Base):
        def forward(self, x, m, keys):
            self.seen, self.mask = x.clone(), m.clone()
            return x

    step = 0.02                                               # ramp of 0.02 z per pixel, far inside the clip
    z = (np.arange(128, dtype=np.float32)[None, None, :] - 64) * step + np.zeros((1, 128, 1), np.float32)
    hide = np.zeros_like(z, bool)
    hide[:, 40:50, 60:70] = True
    step_u = step / (2 * MI.ZCLIP)
    for size in (256, 512):
        net = Echo("cpu", size, 128)
        net(z, hide)
        plain = torch.nn.functional.interpolate(torch.from_numpy(MI.z_to_u(z))[:, None], size=size, mode="bicubic",
                                                align_corners=False)
        vis = net.mask[0, 0] == 0
        dev = float((net.seen[0, 0] - plain[0, 0]).abs()[vis].max())
        assert dev <= 0.35 * step_u, (size, dev / step_u)
    # the discarded v2 construction fails this test by a wide margin (about one pixel step)
    x = torch.from_numpy(MI.z_to_u(z))[:, None]
    m0 = torch.from_numpy(hide.astype(np.float32))[:, None]
    ring = (torch.nn.functional.max_pool2d(m0, 7, stride=1, padding=3) - m0).clamp(0, 1)
    v2 = torch.nn.functional.interpolate(x * (1 - m0) + (x * ring).sum() / ring.sum() * m0, size=256, mode="bicubic",
                                         align_corners=False)
    plain = torch.nn.functional.interpolate(x, size=256, mode="bicubic", align_corners=False)
    m = torch.nn.functional.interpolate(m0, size=256, mode="nearest")
    assert float((v2 - plain).abs()[m == 0].max()) > 0.8 * step_u


def test_grey_mapping_is_inverse_inside_the_clip():
    import model_fills_infer as MI
    z = np.linspace(-MI.ZCLIP, MI.ZCLIP, 101)
    assert np.allclose(MI.u_to_z(MI.z_to_u(z)), z)
    assert MI.z_to_u(np.array([-99.0, 99.0])).tolist() == [0.0, 1.0]


def test_paired_difference_on_a_toy_ladder():
    import model_fills_summary as MS
    rows = []
    for t in range(40):
        for p in range(3):
            rows.append({"target": t, "patch": p, "fill": "a", "mse": 1.0})
            rows.append({"target": t, "patch": p, "fill": "b", "mse": 3.0})
            rows += [{"target": t, "patch": p, "fill": f"str{j:02d}", "mse": 4.0} for j in range(16)]
    r = pd.DataFrame(rows)
    v, lo, hi, n, nu = MS.paired(r, ["target", "patch"], "target", "mse", "a", "b", np.random.default_rng(0))
    assert (v, lo, hi, n, nu) == (-0.5, -0.5, -0.5, 120, 40)


def test_results_json_marks_gates_and_coverage():
    path = os.path.join(ROOT, "results", "model_fills_v1", "model_fills_f8.json")
    if not os.path.exists(path):
        pytest.skip("model_fills_f8.json not built")
    d = json.load(open(path, encoding="utf-8"))
    for name, ds in d["datasets"].items():
        assert ds["gate"]["passed"] and ds["gate"]["max_abs_diff_to_common_f8_v1"] <= 1e-9
        for model, e in ds["models"].items():
            if "status" in e:
                continue
            assert e["n_targets"] <= e["n_targets_dataset"]
            assert ("subset" in e["coverage"]) == (e["n_targets"] < e["n_targets_dataset"])
            for m, v in e["R_self"].items():
                assert v[1] <= v[0] <= v[2], (name, model, m)


def test_checkpoint_is_refused_unless_its_hash_is_pinned(tmp_path):
    """A checkpoint file is loaded only if its SHA-256 equals a pinned value (commit security review, 17 Nov 2026)."""
    import model_fills_infer as MI
    f = tmp_path / "weights.pt"
    f.write_bytes(b"not a model")
    with pytest.raises(SystemExit):                       # no hash given
        MI.verify_ckpt(str(f), None)
    with pytest.raises(SystemExit):                       # a hash that is not one of the pins
        MI.verify_ckpt(str(f), MI.sha256(str(f)))
    with pytest.raises(SystemExit):                       # a pinned hash that the file does not have
        MI.verify_ckpt(str(f), MI.PINNED_SHA256["big-lama.pt"])
