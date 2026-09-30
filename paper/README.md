# Manuscript sources and PDFs

[`Main.tex`](Main.tex) is the primary editable and compilable source.
It includes [`Supplement.tex`](Supplement.tex) as a body-only fragment
with one shared bibliography. The scene illustration uses one column;
the AMM diagram and the three supplementary t-SNE plots are included in
the same complete document.

The current complete PDF is [`Main.pdf`](Main.pdf). Final `.tex` and
`.pdf` files are kept directly in this directory. Editable SVG/PNG figure
assets and provenance are under `figures/motivation/`; temporary build
files are under the ignored `build/` directory.

Build from this directory:

```bash
mkdir -p build/main
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build/main -jobname=Main Main.tex
cp build/main/Main.pdf Main.pdf
```

Completed revisions are compiled, recorded in
[`TEX_CHANGELOG.md`](TEX_CHANGELOG.md), and committed and pushed to
GitHub. Numbered PDFs and Git tags are listed in
[`VERSIONS.md`](VERSIONS.md); file hashes are in `versions.json` and
`Main-build-manifest.json`. Edit Main for subsequent revisions.

## Preserved author original

`MM-mixer.tex` is an exact, byte-for-byte copy of the author-supplied
`/data2/yb/MM-mixer/MM-mixer.tex` (SHA256
`7e9d5470c30331704a1baadfa9031906cbda5c0f322bb4fbefe3f6646f447064`).
No scores, standard deviations, seeds, loss formulas, or ablation values
were edited for this upload.

The supplied `MM-Mixer.pdf` is an architecture figure, so keep the Main
job name when compiling. To compile the original separately, use
`latexmk -pdf -outdir=build/original -jobname=paper-original MM-mixer.tex`.
The AAAI style, bibliography, bibliography style, and architecture
figure are included here. The original standalone Supplement source
can be recovered from the `paper-v003` tag.
