"""Create an editable SVG of AMM and export vector PDF plus PNG previews.

Geometry is illustrative; dimensions and operators follow the released model.
No experiment data or synthetic performance values are used.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
W, H = 1000, 940
WIDTH_MM = 86
HEIGHT_MM = WIDTH_MM * H / W
INK = "#243447"
GRID = "#a2b0bc"
COLORS = {"S": "#337fae", "M": "#8370b4", "D": "#b47730"}
parts: list[str] = []


def add(value: str) -> None:
    parts.append(value)


def text(x, y, value, size=27, color=INK, anchor="start", weight="normal"):
    add(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')


def line(a, b, color=GRID, width=1.8, dash=None, opacity=1):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
        f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>')


def polygon(points, fill="none", stroke=GRID, width=1.8, opacity=1):
    values = " ".join(f"{x},{y}" for x, y in points)
    add(f'<polygon points="{values}" fill="{fill}" fill-opacity="{opacity}" '
        f'stroke="{stroke}" stroke-width="{width}"/>')


def rect(x, y, w, h, fill="white", stroke=GRID, radius=0, width=1.8):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def circle(x, y, r=5, fill=INK, stroke="white", width=1.2):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>')


def arrow(a, b, color=INK, width=2, head=10):
    import math
    line(a, b, color, width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    back = (b[0] - head * math.cos(angle), b[1] - head * math.sin(angle))
    offset = (-head * .42 * math.sin(angle), head * .42 * math.cos(angle))
    polygon([b, (back[0]+offset[0], back[1]+offset[1]),
             (back[0]-offset[0], back[1]-offset[1])], color, color, .5)


def p(d, s, m):
    return (300 + 55*d - 23*s, 335 + 17*s - 61*m)


def fiber(axis, values):
    color = COLORS[axis]
    line(values[0], values[-1], "white", 11)
    line(values[0], values[-1], color, 6)
    for x, y in values:
        circle(x, y, 6, color, "white", 1.8)


def vector(x, y, axis):
    color = COLORS[axis]
    if axis == "S":
        for i in range(6):
            rect(x + 28*i, y-14, 23, 28, "white", color, 2, 2)
    elif axis == "M":
        for i, name in enumerate(["V", "A", "T"]):
            rect(x + 8 + 52*i, y-18, 44, 36, "white", color, 2, 2)
            text(x + 30 + 52*i, y+9, name, 25, color, "middle")
    else:
        for i in [0, 1, 2, 5]:
            rect(x+28*i, y-14, 23, 28, "white", color, 2, 2)
        text(x+114, y+6, "···", 25, color, "middle")


def build_svg():
    parts.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" '
        f'height="{HEIGHT_MM}mm" viewBox="0 0 {W} {H}">')
    add('<title>Axis-wise Multimodal Mixer</title>')
    add('<desc>Three aligned modality branches are expanded into a tensor with '
        'three modality rows, six learned projection views, and 256 hidden features. '
        'Colored vectors illustrate view MLP, modality linear, and feature MLP mappings. '
        'The mappings are shared over other positions and applied sequentially.</desc>')
    rect(0, 0, W, H, "white", "none")
    text(500, 38, "AMM: axis-wise interaction", 30, anchor="middle", weight="bold")
    text(500, 83, "Shared expansion per branch: D → S × D", 27, anchor="middle")

    # The back-bottom-left vertex is the common origin of the three axes.
    # Modality index increases upwards: V, A, T, matching tensor order.
    polygon([p(0,0,0),p(8,0,0),p(8,0,3),p(0,0,3)], "#f2f5f8", GRID, 1.7, .32)
    polygon([p(0,0,0),p(0,6,0),p(0,6,3),p(0,0,3)], "#e9eff5", GRID, 1.7, .38)
    polygon([p(0,0,3),p(8,0,3),p(8,6,3),p(0,6,3)], "#eef3f7", GRID, 1.7, .6)
    for m in range(3):
        shade = ["#f3f6f9", "#e8eef4", "#f3f6f9"][m]
        polygon([p(0,6,m),p(8,6,m),p(8,6,m+1),p(0,6,m+1)], shade, GRID, 1.7, .48)
    for d in range(9):
        line(p(d,0,3),p(d,6,3))
        line(p(d,6,0),p(d,6,3))
    for s in range(7):
        line(p(0,s,3),p(8,s,3))
        line(p(0,s,0),p(0,s,3))
    for m in range(4):
        line(p(0,0,m),p(0,6,m))
        line(p(0,6,m),p(8,6,m))
    # Hidden edges make the inside vertex and its geometric role visible.
    line(p(0,0,0),p(8,0,0),INK,1.5,"6 6",.45)
    line(p(0,0,0),p(0,0,3),INK,1.5,"6 6",.45)
    line(p(0,0,0),p(0,6,0),INK,1.5,"6 6",.45)
    arrow(p(0,0,0),(845,335),INK,2.1,12)
    arrow(p(0,0,0),(300,111),INK,2.1,12)
    arrow(p(0,0,0),(121,467),INK,2.1,12)
    circle(*p(0,0,0),5.5,INK)
    text(309,360,"O",25)
    text(332,123,"M = 3",27,COLORS["M"])
    text(858,329,"D = 256",27,COLORS["D"])
    text(70,494,"S = 6",27,COLORS["S"])
    for m, name in enumerate(["V", "A", "T"]):
        x,y=p(0,6,m+.5)
        text(x-27,y+9,name,28,INK,"middle", "bold")
    fiber("S",[p(5.5,s+.5,3) for s in range(6)])
    fiber("M",[p(1.5,5,m+.5) for m in range(3)])
    fiber("D",[p(d+.5,6,.5) for d in range(8)])
    text(622,181,"S",27,COLORS["S"],weight="bold")
    text(244,250,"M",27,COLORS["M"],weight="bold")
    text(621,415,"D",27,COLORS["D"],weight="bold")
    text(552,487,"D channels shown schematically",25,"#586779","middle")

    line((22,520),(978,520),"#d9e1e7",1.5)
    text(22,555,"One vector from each axis",27,weight="bold")
    rows=[("S",607,"View axis", "Z[m, :, d]", "View MLP", "6 → 12 → 6"),
          ("M",697,"Modality axis", "Z[:, s, d]", "Linear", "3 → 3"),
          ("D",787,"Feature axis", "Z[m, s, :]", "Feature MLP", "256 → 1536 → 256")]
    for axis,y,label,index,op,shape in rows:
        color=COLORS[axis]
        text(22,y-5,label,26,color,weight="bold")
        text(22,y+27,index,26,color)
        vector(247,y,axis)
        arrow((423,y),(470,y),color,2.5,10)
        rect(480,y-35,278,70,"#fafbfc",color,8,2)
        text(619,y-6,op,27,color,"middle", "bold")
        text(619,y+23,shape,25,color,"middle")
        arrow((767,y),(794,y),color,2.5,10)
        vector(810,y,axis)
    text(500,854,"Mappings are shared across the other positions.",25,"#586779","middle")
    text(295,892,"Block 1: S → M → D",25,INK,"middle")
    arrow((470,883),(530,883),INK,1.8,9)
    text(720,892,"Block 2: M → S → D",25,INK,"middle")
    text(500,929,"Mean over S → 3 × D",26,INK,"middle")
    add('</svg>')
    return "\n".join(parts)


def main(export_only=False):
    svg_path=HERE/'01_amm_axis_mixing.svg'
    pdf_path=PAPER/'01_amm_axis_mixing.pdf'
    if not export_only:
        svg_path.write_text(build_svg(),encoding='utf-8')
    with fitz.open(svg_path) as source:
        pdf_bytes=source.convert_to_pdf()
    with fitz.open('pdf',pdf_bytes) as doc:
        doc.set_metadata({'title':'AMM: axis-wise interaction',
                          'subject':'Implementation-grounded SVG schematic; M=3, S=6, D=256',
                          'creator':'SVG source exported with PyMuPDF'})
        doc.save(pdf_path,garbage=4,deflate=True)
        doc[0].get_pixmap(dpi=300).save(HERE/'01_amm_axis_mixing.png')
    contract={
        'claim':'AMM applies distinct shared mappings along learned-view, modality, and hidden-feature axes.',
        'source_format':'editable SVG',
        'exporter':'PyMuPDF SVG to vector PDF',
        'native_size_mm':[WIDTH_MM,HEIGHT_MM],
        'tensor_shape':[3,6,256],
        'modality_axis_order_from_origin':['V','A','T'],
        'operators':{'S':{'type':'MLP','widths':[6,12,6]},
                     'M':{'type':'bias-free Linear','widths':[3,3]},
                     'D':{'type':'MLP','widths':[256,1536,256]}},
        'sharing':'Each axis mapping is shared across other positions within a block; block weights are distinct.',
        'block_orders':[['S','M','D'],['M','S','D']],
        'output':'Mean over S, giving 3 × D',
        'omissions':['LayerNorm over D before each operation','Residual addition after each operation'],
        'illustration_not_data':True,
        'svg_sha256':hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    (HERE/'figure_contract.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'svg':str(svg_path),'pdf':str(pdf_path),'size_mm':[WIDTH_MM,HEIGHT_MM]}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only',action='store_true',
                        help='Export the existing editable SVG without regenerating it.')
    main(export_only=parser.parse_args().export_only)
