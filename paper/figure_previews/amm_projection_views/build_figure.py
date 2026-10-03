"""Editable concept preview: modality vectors to learned projection views.

The figure is schematic, not a computation graph. See figure_contract.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from html import escape
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
STEM = "01_amm_projection_views_preview"
W, H = 1280, 720
WIDTH_MM = 86
INK = "#344354"
MUTED = "#6a7c8f"
GRID = "#ccd6df"
PALETTE = {
    "S": {"top": "#b5d8e8", "front": "#83b6d0", "right": "#629dbb", "ink": "#347fa3"},
    "M": {"top": "#d5cbe6", "front": "#b4a1cf", "right": "#9580b4", "ink": "#7b629f"},
    "D": {"top": "#f1d7b7", "front": "#deb680", "right": "#c19258", "ink": "#a7753a"},
}
parts: list[str] = []


def text(x, y, label, size=37, color=INK, anchor="start", weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
                 f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(label)}</text>')


def poly(points, fill="none", stroke=GRID, width=1.4, extra=""):
    coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    parts.append(f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}" stroke-linejoin="round" {extra}/>')


def line(a, b, stroke=GRID, width=1.5, dash=False):
    if not dash:
        parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                     f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>')
        return
    # Explicit dash segments are required by the PyMuPDF SVG importer.
    length = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    at = 0.0
    while at < length:
        end = min(at + 6, length)
        line((a[0] + at*ux, a[1] + at*uy), (a[0] + end*ux, a[1] + end*uy), stroke, width)
        at += 13


def arrow(a, b, stroke=INK, width=2.2, head=11):
    line(a, b, stroke, width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    base = (b[0] - head*math.cos(angle), b[1] - head*math.sin(angle))
    offset = (-head*.44*math.sin(angle), head*.44*math.cos(angle))
    poly([b, (base[0]+offset[0], base[1]+offset[1]),
          (base[0]-offset[0], base[1]-offset[1])], stroke, stroke, .5)


def p(d, s, m):
    """Project the cuboid from its inner back-bottom-left vertex."""
    return 707 + 49*d - 30.3*s, 433 + 17*s - 60*m


def tensor_wireframe():
    origin = p(0, 0, 0)
    edges = []
    for s in (0, 6):
        for m in (0, 3):
            edges.append((p(0, s, m), p(7, s, m)))
    for d in (0, 7):
        for m in (0, 3):
            edges.append((p(d, 0, m), p(d, 6, m)))
    for d in (0, 7):
        for s in (0, 6):
            edges.append((p(d, s, 0), p(d, s, 3)))
    # Transparent container: no grey face fills conceal the internal bar faces.
    for d in range(1, 7):
        line(p(d, 0, 3), p(d, 6, 3))
        line(p(d, 6, 0), p(d, 6, 3))
    for s in range(1, 6):
        line(p(0, s, 3), p(7, s, 3))
        line(p(7, s, 0), p(7, s, 3))
    for m in range(1, 3):
        line(p(0, 6, m), p(7, 6, m))
        line(p(7, 0, m), p(7, 6, m))
    parts.append('<g id="cuboid-edges" data-edge-count="12">')
    for a, b in edges:
        hidden = a == origin or b == origin
        line(a, b, "#a3b0bd" if hidden else "#8395a6", 1.6 if hidden else 2.0, hidden)
    parts.append('</g>')


def solid_fibers():
    selected = {
        "S": {(3, s, 2) for s in range(6)},
        "M": {(6, 1, m) for m in range(3)},
        "D": {(d, 5, 0) for d in range(7)},
    }
    faces = []
    for axis, cells in selected.items():
        for d, s, m in cells:
            candidates = [
                ("top", (d, s, m+1), [(d,s,m+1),(d+1,s,m+1),(d+1,s+1,m+1),(d,s+1,m+1)]),
                ("front", (d, s+1, m), [(d,s+1,m),(d+1,s+1,m),(d+1,s+1,m+1),(d,s+1,m+1)]),
                ("right", (d+1, s, m), [(d+1,s,m),(d+1,s+1,m),(d+1,s+1,m+1),(d+1,s,m+1)]),
            ]
            for name, neighbor, vertices in candidates:
                if neighbor in cells:
                    continue
                center = [sum(v[k] for v in vertices)/4 for k in range(3)]
                depth = (30.3/49)*center[0] + center[1] + (17/60)*center[2]
                faces.append((depth, axis, name, (d,s,m), vertices))
    # Opaque exposed faces occlude grid lines. Internal shared faces are omitted.
    parts.append('<g id="solid-axis-fibers">')
    for _, axis, name, cell, vertices in sorted(faces):
        poly([p(*v) for v in vertices], PALETTE[axis][name], PALETTE[axis]["ink"], 1.15,
             f'data-axis="{axis}" data-face="{name}" data-cell="{cell[0]},{cell[1]},{cell[2]}"')
    parts.append('</g>')
    return selected, len(faces)


def build_svg():
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" '
                 f'height="{WIDTH_MM*H/W}mm" viewBox="0 0 {W} {H}">')
    parts.append('<title>AMM: from aligned modality vectors to learned projection views</title>')
    parts.append('<desc>Three aligned modality rows undergo a shared learned projection '
                 'from D to S times D followed by reshape. The resulting M by S by D '
                 'tensor contains learned views, not temporal steps. Solid colored fibers '
                 'show view, modality and feature mixing directions, not parallel branches.</desc>')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    text(170, 115, "Aligned modality", anchor="middle", size=37)
    text(170, 157, "vectors", anchor="middle", size=37)
    text(898, 115, "Learned projection views", anchor="middle", size=38)

    # Top-to-bottom labels T/A/V match the cuboid's M-axis orientation.
    x0, y0, cw, rh = 84, 297, 180/7, 54
    for row, label in enumerate(("T", "A", "V")):
        for col in range(7):
            x, y = x0+col*cw, y0+row*rh
            poly([(x,y),(x+cw,y),(x+cw,y+rh),(x,y+rh)],
                 ("#e2e9ee", "#eef2f5")[col % 2], "#bdcbd6", 1.3)
        text(x0-18, y0+(row+.5)*rh+12, label, anchor="end", size=37)
    text(174, 511, "M × D", anchor="middle")

    text(397, 268, "Shared", anchor="middle", size=37)
    text(397, 309, "projection", anchor="middle", size=37)
    arrow((292, 378), (502, 378), width=2.6, head=14)
    text(397, 448, "D → S·D", anchor="middle", size=37)
    text(397, 491, "+ reshape", anchor="middle", size=37, color=MUTED)

    tensor_wireframe()
    selected, face_count = solid_fibers()

    # Three axes extend the hidden edges from the same inner origin O.
    arrow(p(0,0,3), (707, 203), width=1.9, head=10)
    arrow(p(7,0,0), (1157, 433), width=1.9, head=10)
    arrow(p(0,6,0), p(0,7.2,0), width=1.9, head=10)
    ox, oy = p(0,0,0)
    parts.append(f'<circle cx="{ox}" cy="{oy}" r="3.3" fill="{MUTED}"/>')
    text(724, 214, "M = 3", size=37, color=PALETTE["M"]["ink"])
    text(1168, 445, "D", size=37, color=PALETTE["D"]["ink"])
    text(471, 611, "S = 6", size=37, color=PALETTE["S"]["ink"])

    # Callouts identify axis fibers. No arrows imply three parallel computation branches.
    text(1002, 190, "View mixing", size=37, anchor="middle", color=PALETTE["S"]["ink"])
    line(p(3.5,1.4,3), (876, 214), PALETTE["S"]["ink"], 1.6)
    line((876,214), (933,214), PALETTE["S"]["ink"], 1.6)
    text(1140, 303, "Modality", size=37, anchor="middle", color=PALETTE["M"]["ink"])
    text(1140, 345, "mixing", size=37, anchor="middle", color=PALETTE["M"]["ink"])
    line(p(7,1.5,1.5), (1058,364), PALETTE["M"]["ink"], 1.6)
    line((1058,364), (1138,364), PALETTE["M"]["ink"], 1.6)
    text(971, 620, "Feature mixing", size=37, anchor="middle", color=PALETTE["D"]["ink"])
    line(p(5.5,6,0.5), (939,575), PALETTE["D"]["ink"], 1.6)
    line((939,575), (992,575), PALETTE["D"]["ink"], 1.6)
    text(800, 686, "M × S × D", anchor="middle", size=37)
    parts.append('</svg>')
    return "\n".join(parts), selected, face_count


def main():
    svg, selected, face_count = build_svg()
    svg_path, pdf_path = HERE/f"{STEM}.svg", HERE/f"{STEM}.pdf"
    svg_path.write_text(svg, encoding="utf-8")
    with fitz.open(svg_path) as src:
        pdf_bytes = src.convert_to_pdf()
    with fitz.open("pdf", pdf_bytes) as doc:
        doc.set_metadata({"title": "AMM: from modality vectors to learned projection views",
                          "subject": "Concept preview; learned projection and axis-wise operations",
                          "creator": "Editable SVG exported with PyMuPDF"})
        doc.save(pdf_path, garbage=4, deflate=True)
        doc[0].get_pixmap(dpi=450).save(HERE/f"{STEM}.png")
    contract = {
        "status": "concept_preview_not_integrated_in_Main",
        "claim": "A shared learned projection constructs S views per aligned modality branch before axis-wise mixing.",
        "size_mm": [WIDTH_MM, WIDTH_MM*H/W],
        "input_shape_without_batch": [3,256], "output_shape_without_batch": [3,6,256],
        "display_grid_shape_M_S_D": [3,6,7], "D_grid_schematic": True,
        "input": "MCA-aligned branch vectors; not raw unimodal observations",
        "expansion": "shared Linear D->S*D followed by reshape; no feature copying or dimensional partition",
        "weights_shared_across": "modality branches",
        "axis_fiber_indices_d_s_m": {k: sorted(v) for k,v in selected.items()},
        "colored_exposed_face_count": face_count,
        "operators": {"S": "MLP 6->12->6", "M": "bias-free static Linear 3->3", "D": "MLP 256->1536->256"},
        "callout_semantics": "axis directions, not parallel computation branches",
        "omitted": ["batch dimension", "block orders S-M-D / M-S-D", "LayerNorm", "residual additions", "pooling"],
        "minimum_main_font_pt": 37*WIDTH_MM/25.4*72/W,
        "source_code": ["vendor/meld/model.py:102-160", "vendor/iemocap/factorized_mixer/model.py:321-396"],
        "svg_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    (HERE/"figure_contract.json").write_text(json.dumps(contract, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"svg":str(svg_path), "pdf":str(pdf_path), "size_mm":contract["size_mm"]}))


if __name__ == "__main__":
    main()
