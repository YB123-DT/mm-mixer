"""Draw the complete AMM tensor as editable SVG and export vector PDF/PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
W, H = 1000, 580
WIDTH_MM = 86
HEIGHT_MM = WIDTH_MM * H / W
INK = "#243447"
GRID = "#93a4b3"
COLORS = {"S": "#75b4d8", "M": "#b5a2d6", "D": "#e7b475"}
LABELS = {"S": "#28668f", "M": "#71549e", "D": "#a96b24"}
parts: list[str] = []


def text(x, y, value, size=29, color=INK, anchor="start", weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" '
                 f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(value)}</text>')


def line(a, b, color=GRID, width=1.7, dash=None, opacity=1):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                 f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{extra}/>')


def polygon(points, fill="none", stroke=GRID, width=1.7, opacity=1, extra=""):
    points_text = " ".join(f"{x},{y}" for x, y in points)
    parts.append(f'<polygon points="{points_text}" fill="{fill}" fill-opacity="{opacity}" '
                 f'stroke="{stroke}" stroke-width="{width}" {extra}/>')


def arrow(a, b, color=INK, width=2.4, head=12):
    line(a, b, color, width)
    angle = math.atan2(b[1]-a[1], b[0]-a[0])
    back = (b[0]-head*math.cos(angle), b[1]-head*math.sin(angle))
    off = (-head*.42*math.sin(angle), head*.42*math.cos(angle))
    polygon([b, (back[0]+off[0], back[1]+off[1]),
             (back[0]-off[0], back[1]-off[1])], color, color, .5)


def p(d, s, m):
    """Common origin: inner back-bottom-left vertex; V/A/T increase up."""
    return (310 + 57.5*d - 27*s, 360 + 20*s - 70*m)


def face(vertices, fill, opacity=.36):
    polygon([p(*v) for v in vertices], fill, "none", 0, opacity)


def colored_cell(axis, vertices, index, cap=False):
    role = "cap" if cap else "cell"
    polygon([p(*v) for v in vertices], COLORS[axis], LABELS[axis], 1.7,
            extra=f'data-axis="{axis}" data-role="{role}" data-index="{index}"')


def build_svg():
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" '
                 f'height="{HEIGHT_MM}mm" viewBox="0 0 {W} {H}">')
    parts.append('<title>AMM tensor with three separate axis vectors</title>')
    parts.append('<desc>A complete transparent cuboid has three modality rows, '
                 'six learned projection views and a schematic feature dimension. '
                 'Blue, purple and orange full-cell strips are spatially separated '
                 'and show vectors along S, M and D, respectively.</desc>')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    text(500, 40, "AMM: three representation axes", 32, anchor="middle", weight="bold")

    # All six faces are present. Transparency exposes the inner common origin.
    face([(0,0,0),(8,0,0),(8,0,3),(0,0,3)], "#edf2f6", .18)
    face([(0,0,0),(8,0,0),(8,6,0),(0,6,0)], "#edf2f6", .12)
    face([(8,0,0),(8,6,0),(8,6,3),(8,0,3)], "#edf2f6", .45)
    face([(0,0,0),(0,6,0),(0,6,3),(0,0,3)], "#edf2f6", .26)
    face([(0,0,3),(8,0,3),(8,6,3),(0,6,3)], "#eef3f7", .65)
    face([(0,6,0),(8,6,0),(8,6,3),(0,6,3)], "#f3f6f9", .35)

    # M: fixed d=0, s=1. Three full cells, one per modality.
    for m in range(3):
        colored_cell("M",[(0,1,m),(0,2,m),(0,2,m+1),(0,1,m+1)],m)
    colored_cell("M",[(0,1,3),(1,1,3),(1,2,3),(0,2,3)],"top",cap=True)
    # S: fixed m=2, d=6. All six projection cells on the top.
    for s in range(6):
        colored_cell("S",[(6,s,3),(7,s,3),(7,s+1,3),(6,s+1,3)],s)
    colored_cell("S",[(6,6,2),(7,6,2),(7,6,3),(6,6,3)],"front",cap=True)
    # D: fixed m=0, s=5. Eight schematic cells represent the 256 channels.
    for d in range(8):
        colored_cell("D",[(d,6,0),(d+1,6,0),(d+1,6,1),(d,6,1)],d)
    colored_cell("D",[(8,5,0),(8,6,0),(8,6,1),(8,5,1)],"right",cap=True)

    for d in range(9):
        line(p(d,0,3),p(d,6,3))
        line(p(d,6,0),p(d,6,3))
    for s in range(7):
        line(p(0,s,3),p(8,s,3))
        line(p(8,s,0),p(8,s,3))
    for m in range(4):
        line(p(0,6,m),p(8,6,m))
        line(p(8,0,m),p(8,6,m))
        line(p(0,0,m),p(0,6,m),GRID,1.2,"5 5",.40)
    # The far-left face grid is faint, distinguishing it from the front face.
    for s in range(1,6):
        line(p(0,s,0),p(0,s,3),GRID,1.2,"5 5",.38)

    edges=[]
    for s in [0,6]:
        for m in [0,3]: edges.append((p(0,s,m),p(8,s,m)))
    for d in [0,8]:
        for m in [0,3]: edges.append((p(d,0,m),p(d,6,m)))
    for d in [0,8]:
        for s in [0,6]: edges.append((p(d,s,0),p(d,s,3)))
    parts.append('<g id="cuboid-edges">')
    for a,b in edges: line(a,b,"#8295a6",2.4)
    parts.append('</g>')

    text(710,177,"S",31,LABELS["S"],weight="bold")
    text(211,263,"M",31,LABELS["M"],weight="bold")
    text(630,451,"D",31,LABELS["D"],weight="bold")
    origin=p(0,0,0)
    arrow(origin,(865,360),INK)
    arrow(origin,(310,95),INK)
    arrow(origin,(103,514),INK)
    parts.append(f'<circle cx="{origin[0]}" cy="{origin[1]}" r="5.5" '
                 f'fill="{INK}" stroke="white" stroke-width="1.4"/>')
    text(322,387,"O",27)
    text(333,105,"M = 3",30,LABELS["M"])
    text(873,349,"D = 256",29,LABELS["D"])
    text(61,550,"S = 6",30,LABELS["S"])
    for m,name in enumerate(["V","A","T"]):
        x,y=p(0,6,m+.5)
        text(x-31,y+10,name,32,INK,"middle","bold")
    parts.append('</svg>')
    return '\n'.join(parts)


def main(export_only=False):
    svg_path=HERE/'01_amm_axis_mixing.svg'
    pdf_path=PAPER/'01_amm_axis_mixing.pdf'
    if not export_only:
        svg_path.write_text(build_svg(),encoding='utf-8')
    with fitz.open(svg_path) as source:
        pdf_bytes=source.convert_to_pdf()
    with fitz.open('pdf',pdf_bytes) as doc:
        doc.set_metadata({'title':'AMM: three representation axes',
                          'subject':'Complete cuboid and disjoint full-cell S/M/D strips',
                          'creator':'Editable SVG exported with PyMuPDF'})
        doc.save(pdf_path,garbage=4,deflate=True)
        doc[0].get_pixmap(dpi=300).save(HERE/'01_amm_axis_mixing.png')
    contract={
        'claim':'Three disjoint cell strips illustrate axis-wise vectors in the AMM tensor.',
        'source_format':'editable SVG','exporter':'PyMuPDF SVG to vector PDF',
        'native_size_mm':[WIDTH_MM,HEIGHT_MM],'tensor_shape':[3,6,256],
        'display_grid_shape':[3,6,8],'modality_axis_order_from_origin':['V','A','T'],
        'bounding_edge_count':12,
        'selected_indices':{'S':{'m':2,'d':6,'varies':'s'},
                            'M':{'s':1,'d':0,'varies':'m'},
                            'D':{'m':0,'s':5,'varies':'d'}},
        'primary_colored_cell_counts':{'S':6,'M':3,'D':8},
        'operators':{'S':{'type':'MLP','widths':[6,12,6]},
                     'M':{'type':'bias-free Linear','widths':[3,3]},
                     'D':{'type':'MLP','widths':[256,1536,256]}},
        'operators_shown':False,'lower_operation_panels_shown':False,
        'omissions':['Projection expansion','Axis mappings','LayerNorm over D',
                     'Residual additions','Block order','Pooling'],
        'illustration_not_data':True,
        'svg_sha256':hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    (HERE/'figure_contract.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'svg':str(svg_path),'pdf':str(pdf_path),'size_mm':[WIDTH_MM,HEIGHT_MM]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only',action='store_true',
                        help='Export the existing SVG without regenerating it.')
    main(export_only=parser.parse_args().export_only)
