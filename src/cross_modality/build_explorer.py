"""Assemble the metrics-explorer page from the template and the aggregate files (no IDs, aggregates only).

Usage: python -I src/cross_modality/build_explorer.py results/cross_modality <out.html>
Placeholders in explorer_template.html: __DATA__, __COMMON__ (JSON) and __TILES__, __TILESDS__, __CAL__, __AGREE__, __GEN__, __CARD__
(PNG figures, embedded as base64). A placeholder left in the output is an error, not a silent gap.
"""
import base64
import os
import sys

SCR, OUT = sys.argv[1], sys.argv[2]
page = open(os.path.join(os.path.dirname(__file__), "explorer_template.html"), encoding="utf-8").read()
for key, name in (("__DATA__", "data.json"), ("__COMMON__", "common_all.json")):
    page = page.replace(key, open(os.path.join(SCR, name), encoding="utf-8").read())
for key, name in (("__TILESDS__", "rank_tiles_by_dataset.png"), ("__TILES__", "rank_tiles.png"), ("__CAL__", "cal_tiles.png"),
                  ("__AGREE__", "agreement.png"), ("__GEN__", "generative_tiles.png"), ("__CARD__", "dataset_card.png")):
    page = page.replace(key, base64.b64encode(open(os.path.join(SCR, name), "rb").read()).decode())
left = [k for k in ("__DATA__", "__COMMON__", "__TILES__", "__TILESDS__", "__CAL__", "__AGREE__", "__GEN__", "__CARD__") if k in page]
if left:
    raise SystemExit(f"placeholders not filled: {left}")
open(OUT, "w", encoding="utf-8").write(page)
print(f"written {OUT} ({len(page) / 1e6:.1f} MB)")
