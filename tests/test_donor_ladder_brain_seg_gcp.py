"""Input clip and label checks of the GCP FastSurfer step (Amendment 15, "Input clip")."""
import os
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_seg_gcp as G  # noqa: E402


def test_clip_level_ignores_zero_background_and_outliers():
    rng = np.random.default_rng(0)
    v = np.zeros((40, 40, 40), np.float32)
    v[5:35, 5:35, 5:35] = rng.uniform(1, 100, (30, 30, 30))
    v[10, 10, 10] = 3.4e6   # the kind of outlier that emptied FastSurfer's conform on t009
    c = G.clip_level(v)
    assert 99 < c <= 100
    out = G.clip(v, c)
    assert out.dtype == np.float32 and out.max() == pytest.approx(c) and out.min() == 0


def test_clip_level_refuses_empty_oracle():
    with pytest.raises(ValueError):
        G.clip_level(np.zeros((4, 4, 4), np.float32))


def test_fill_above_counts_only_scored_voxels():
    box = np.array([[[1.0, 5.0], [9.0, 20.0]]])
    valid = np.array([[[True, True], [False, True]]])
    r = G.fill_above(box, valid, 8.0)
    assert r == {"n_scored": 3, "n_above": 1, "share": pytest.approx(1 / 3)}


def test_collect_refuses_empty_label(tmp_path, monkeypatch):
    priv = tmp_path / "priv"
    (priv / "t001" / "in").mkdir(parents=True)
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), aff), str(priv / "t001" / "in" / "oracle.nii.gz"))
    dl = tmp_path / "dl"
    for name, val in (("oracle", 17), ("noise", 0)):
        d = dl / f"t001_{name}" / "mri"
        d.mkdir(parents=True)
        nib.save(nib.MGHImage(np.full((6, 6, 6), val, np.int32), aff), str(d / "aparc.DKTatlas+aseg.deep.mgz"))
    monkeypatch.setattr(G.S, "PRIVATE", str(priv))
    monkeypatch.setattr(G, "DL_DIR", str(dl))
    st = G.collect_target("t001", ["oracle", "noise", "repeat"])
    assert st == {"oracle": "done", "noise": "empty", "repeat": "missing"}
    assert (priv / "t001" / "label_fastsurfer_oracle.npy").exists()
    assert not (priv / "t001" / "label_fastsurfer_noise.npy").exists()


RUNNER = os.path.join(os.path.dirname(__file__), "..", "gcp", "fastsurfer_run.sh")


def test_runner_docker_does_not_read_the_list_from_stdin():
    # 6 Nov 2026: `docker run` and then `gcloud storage rsync` inside `while read` consumed the rest of the list;
    # only the first volume ran. The list is read on fd 3 so no command in the loop can take it.
    src = open(RUNNER, encoding="utf-8").read()
    assert "while read -r item <&3" in src and 'done 3< "$LIST"' in src


def test_runner_does_not_trust_exit_code_zero():
    # 6 Nov 2026: FastSurfer exited 0 after create_mask raised AssertionError on an empty conform.
    src = open(RUNNER, encoding="utf-8").read()
    assert "mask.mgz" in src and "Traceback" in src and "AssertionError" in src


def test_runner_honours_stop_file_before_each_volume():
    src = open(RUNNER, encoding="utf-8").read()
    loop = src[src.index("while read -r item <&3"):]
    assert loop.index('"$BASE/STOP"') < loop.index("sudo docker run")


def test_runner_stops_when_another_job_holds_the_gpu():
    src = open(RUNNER, encoding="utf-8").read()
    loop = src[src.index("while read -r item <&3"):]
    assert loop.index("nvidia-smi --query-gpu=memory.used") < loop.index("sudo docker run")


LOCAL_RUNNER = os.path.join(os.path.dirname(__file__), "..", "local", "fastsurfer_native_run.sh")


def test_local_runner_reads_the_list_on_fd3_and_closes_stdin():
    src = open(LOCAL_RUNNER, encoding="utf-8").read()
    assert "while read -r item <&3" in src and 'done 3< "$LIST"' in src
    assert "< /dev/null" in src


def test_local_runner_does_not_trust_exit_code_zero():
    src = open(LOCAL_RUNNER, encoding="utf-8").read()
    assert "rc=97" in src and "mask.mgz" in src and "rc=98" in src and "AssertionError" in src and "rc=99" in src


def test_local_runner_stop_file_and_skip_done_come_before_fastsurfer():
    src = open(LOCAL_RUNNER, encoding="utf-8").read()
    loop = src[src.index("while read -r item <&3"):]
    run = loop.index('"$FS_HOME/run_fastsurfer.sh"')
    assert loop.index('"$BASE/STOP"') < run and loop.index("DONE") < run


def test_local_runner_requires_explicit_threads_and_private_folder():
    # no silent defaults: unset FS_THREADS or PRIVATE must stop the script
    src = open(LOCAL_RUNNER, encoding="utf-8").read()
    assert "${FS_THREADS:?" in src and "${PRIVATE:?" in src


def test_run_summary_counts_retried_volume_once_and_keeps_no_names(tmp_path):
    st = tmp_path / "status.tsv"
    st.write_text("t1_oracle\tDONE\t900\tattempt1\tT\nt1_noise\tFAILED\trc=97\tlog\tT\nt1_noise\tDONE\t950\tattempt1\tT\n"
                  "t1_mirror\tFAILED\trc=99\tlog\tT\n")
    us = tmp_path / "usage.log"
    us.write_text("a\tb\tseg_cpu\tt1_oracle\tattempt1\trc=0\t900s\t[a]\n"
                  "a\tb\tseg_cpu\tt1_mirror\tattempt1\trc=99\t50s\t[a]\n"
                  "a\tb\tseg_cpu\tt1_noise\tattempt1\trc=0\t1000s\t[b]\n")
    r = G.run_summary(str(st), str(us), n_total=5)
    assert (r["n_done"], r["n_failed"], r["n_left"]) == (2, 1, 2)
    assert r["seconds_per_volume"]["n"] == 2 and r["seconds_per_volume"]["median"] == 950
    assert "t1_" not in str(r)


CPU_VM = os.path.join(os.path.dirname(__file__), "..", "gcp", "fastsurfer_cpu_vm.sh")


def test_cpu_vm_keeps_the_laptop_device_class_and_runner():
    # 9 Nov 2026 remainder: same device class (CPU), version and runner as the laptop run; no GPU flag anywhere
    src = open(CPU_VM, encoding="utf-8").read()
    assert "cuda" not in src.lower().replace("torch.cuda.is_available()", "")
    assert 'FS_VERSION=2.5.3+926af8d' in src and 'TORCH=2.7.1+cpu' in src and '"$TORCH False"' in src
    assert 'fastsurfer_native_run.sh' in src and "${BUCKET:?" in src and "${4:?threads}" in src


def test_cpu_vm_checks_input_hashes_before_any_run():
    src = open(CPU_VM, encoding="utf-8").read()
    pull = src[src.index("  pull)"):src.index("  run)")]
    assert "sha256sum -c" in pull and "exit 3" in pull
    assert src.index("  run)") < src.index("check\n    rm -f")   # run starts with the version and device check
