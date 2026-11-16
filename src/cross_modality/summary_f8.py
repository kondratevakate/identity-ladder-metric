"""Panel-B values of the summary figure with error and similarity metrics at a common window of 1/8 of the object.

Runs summary_figure.py unchanged on a temporary copy of results/cross_modality in which common_all.json is replaced
by common_f8_v1.json (Amendment 10). LPIPS, topology, FID and FWD still come from values_generative.json at their
original windows, which the output records in `window_note`.
Usage: python -I src/cross_modality/summary_f8.py   (run from the repository root)
Writes results/cross_modality/fig_metric_summary_f8_R.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

CM = os.path.join("results", "cross_modality")
OUT = os.path.join(CM, "fig_metric_summary_f8_R.json")
NOTE = ("Error and similarity metrics (MSE, SSIM, TM-, GDT-, lDDT-like, RMSE) from common_f8_v1.json: hidden window "
        "1/8 of the object in all nine image sets (Amendment 10). LPIPS, topology, FID and FWD from "
        "values_generative.json at their original windows (brain 16 mm = 1/8; faces, iris and retina 16 px, "
        "about 1/5 of the object).")

with tempfile.TemporaryDirectory() as tmp:
    for name in os.listdir(CM):
        if name.endswith(".json"):
            shutil.copy(os.path.join(CM, name), tmp)
    shutil.copy(os.path.join(CM, "common_f8_v1.json"), os.path.join(tmp, "common_all.json"))
    subprocess.run([sys.executable, "-I", os.path.join("src", "cross_modality", "summary_figure.py"), tmp,
                    os.path.join(tmp, "fig_f8")], check=True)
    values = json.load(open(os.path.join(tmp, "fig_f8_R.json"), encoding="utf-8"))
values["window_note"] = NOTE
json.dump(values, open(OUT, "w", encoding="utf-8"), indent=1)
print("saved", OUT)
