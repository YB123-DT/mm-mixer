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
W, H = 1000, 540
WIDTH_MM = 86
HEIGHT_MM = WIDTH_MM * H / W
INK = "#354152"
GRID = "#c3cdd7"
COLORS = {"S": "#8ebdd3", "M": "#b5a8d1", "D": "#e7bd8d"}
LABELS = {"S": "#347b9c", "M": "#77609d", "D": "#a66f32"}
parts: list[str] = []


def text(x, y, value, size=29, color=INK, anchor="start", weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
                 f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(value)}</text>')


def line(a, b, color=GRID, width=1.3, dash=None, opacity=1):
    if dash:
        # MuPDF's SVG importer ignores stroke-dasharray. Explicit subpaths keep
        # hidden edges dashed in both the editable SVG and the exported PDF.
        on, off = map(float, dash.split())
        length = math.dist(a, b)
        ux, uy = (b[0]-a[0])/length, (b[1]-a[1])/length
        segments = []
        start = 0.0
        while start < length:
            end = min(start+on, length)
            segments.append(f'M {a[0]+ux*start:.3f} {a[1]+uy*start:.3f} '
                            f'L {a[0]+ux*end:.3f} {a[1]+uy*end:.3f}')
            start += on+off
        parts.append(f'<path d="{" ".join(segments)}" fill="none" '
                     f'stroke="{color}" stroke-width="{width}" opacity="{opacity}" '
                     f'data-edge-visibility="hidden"/>')
        return
    parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                 f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')


def polygon(points, fill="none", stroke=GRID, width=1.3, opacity=1, extra=""):
    points_text = " ".join(f"{x},{y}" for x, y in points)
    parts.append(f'<polygon points="{points_text}" fill="{fill}" fill-opacity="{opacity}" '
                 f'stroke="{stroke}" stroke-width="{width}" {extra}/>')


def arrow(a, b, color=INK, width=1.8, head=10):
    line(a, b, color, width)
    angle = math.atan2(b[1]-a[1], b[0]-a[0])
    back = (b[0]-head*math.cos(angle), b[1]-head*math.sin(angle))
    off = (-head*.42*math.sin(angle), head*.42*math.cos(angle))
    polygon([b, (back[0]+off[0], back[1]+off[1]),
             (back[0]-off[0], back[1]-off[1])], color, color, .5)


def p(d, s, m):
    """Common origin: inner back-bottom-left vertex; V/A/T increase up."""
    return (295 + 58*d - (203/6)*s, 338 + 20*s - 70*m)


def face(vertices, fill, opacity=1):
    polygon([p(*v) for v in vertices], fill, "none", 0, opacity)


def colored_cell(axis, vertices, index):
    polygon([p(*v) for v in vertices], COLORS[axis], LABELS[axis], 1.1,
            extra=f'data-axis="{axis}" data-role="cell" data-index="{index}"')


def build_svg():
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" '
                 f'height="{HEIGHT_MM}mm" viewBox="0 0 {W} {H}">')
    parts.append('<title>AMM tensor with three separate axis vectors</title>')
    parts.append('<desc>A complete cuboid has three modality rows, '
                 'six learned projection views and a schematic feature dimension. '
                 'Blue, purple and orange full-cell strips are spatially separated '
                 'and show vectors along S, M and D, respectively.</desc>')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # Only visible faces carry grids. Hidden edges below retain the full cuboid.
    face([(8,0,0),(8,6,0),(8,6,3),(8,0,3)], "#edf2f6")
    face([(0,0,3),(8,0,3),(8,6,3),(0,6,3)], "#f4f7fa")
    face([(0,6,0),(8,6,0),(8,6,3),(0,6,3)], "#fafbfc")

    # M: fixed d=7, s=1. Three visible cells on the right face.
    for m in range(3):
        colored_cell("M",[(8,1,m),(8,2,m),(8,2,m+1),(8,1,m+1)],m)
    # S: fixed m=2, d=4. Six projection cells on the top face.
    for s in range(6):
        colored_cell("S",[(4,s,3),(5,s,3),(5,s+1,3),(4,s+1,3)],s)
    # D: fixed m=0, s=5. Eight schematic cells represent the 256 channels.
    for d in range(8):
        colored_cell("D",[(d,6,0),(d+1,6,0),(d+1,6,1),(d,6,1)],d)

    for d in range(1,8):
        line(p(d,0,3),p(d,6,3))
        line(p(d,6,0),p(d,6,3))
    for s in range(1,6):
        line(p(0,s,3),p(8,s,3))
        line(p(8,s,0),p(8,s,3))
    for m in range(1,3):
        line(p(0,6,m),p(8,6,m))
        line(p(8,0,m),p(8,6,m))

    edges=[]
    for s in [0,6]:
        for m in [0,3]: edges.append((p(0,s,m),p(8,s,m)))
    for d in [0,8]:
        for m in [0,3]: edges.append((p(d,0,m),p(d,6,m)))
    for d in [0,8]:
        for s in [0,6]: edges.append((p(d,s,0),p(d,s,3)))
    parts.append('<g id="cuboid-edges">')
    origin=p(0,0,0)
    for a,b in edges:
        hidden = a==origin or b==origin
        line(a,b,"#8998a8" if hidden else "#718498",
             1.25 if hidden else 1.9,"5 6" if hidden else None,.65 if hidden else 1)
    parts.append('</g>')

    # Axes share the three hidden cuboid edges and extend only outside the body.
    arrow(p(8,0,0),(824,338))
    arrow(p(0,0,3),(295,73))
    arrow(p(0,6,0),p(0,7.2,0))
    parts.append(f'<circle cx="{origin[0]}" cy="{origin[1]}" r="3.6" '
                 f'fill="{INK}" stroke="white" stroke-width="1.2"/>')
    text(305,360,"O",23,color="#6f7f90")
    text(312,83,"M = 3",29,LABELS["M"])
    text(838,348,"D = 256",29,LABELS["D"])
    text(28,523,"S = 6",29,LABELS["S"])
    for m,name in enumerate(["V","A","T"]):
        x,y=p(0,6,m+.5)
        text(x-28,y+10,name,29,INK,"middle")
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
        'selected_indices':{'S':{'m':2,'d':4,'varies':'s'},
                            'M':{'s':1,'d':7,'varies':'m'},
                            'D':{'m':0,'s':5,'varies':'d'}},
        'primary_colored_cell_counts':{'S':6,'M':3,'D':8},
        'visible_face_strips':{'S':'top','M':'right','D':'front'},
        'visible_edges':9,'hidden_edges':3,
        'axis_style':'Hidden cuboid edges from O, then short external arrow segments',
        'internal_grid_shown':False,'colored_caps_shown':False,'title_shown':False,
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
