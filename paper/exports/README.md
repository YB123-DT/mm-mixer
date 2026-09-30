# PDF exports

The full 9-page v003 preview is [`../MM-Mixer-v003-with-figures.pdf`](../MM-Mixer-v003-with-figures.pdf), built from the existing revised manuscript with the scene and AMM figures. The scene figure appears on page 2 and the AMM figure on page 5.

Source: `../MM-mixer-with-figures.tex`. Source and output hashes are recorded in `source_manifest.json`. Final TeX and PDF files live in the `paper/` root; this directory contains export metadata only. Version history is in [`../VERSIONS.md`](../VERSIONS.md).

From the paper directory, rebuild with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build/with-figures -jobname=MM-Mixer-full-with-figures MM-mixer-with-figures.tex
```

Standalone vector PDFs are [`../01_same_words_scenes.pdf`](../01_same_words_scenes.pdf) and [`../02_amm_projection_axes.pdf`](../02_amm_projection_axes.pdf).
