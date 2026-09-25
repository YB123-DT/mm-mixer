# Manuscript base

`MM-mixer.tex` is an exact, byte-for-byte copy of the author-supplied
`/data2/yb/MM-mixer/MM-mixer.tex` (SHA256
`7e9d5470c30331704a1baadfa9031906cbda5c0f322bb4fbefe3f6646f447064`).
No scores, standard deviations, seeds, loss formulas, or ablation values
were edited for this upload.

Build from this directory with
`latexmk -pdf -jobname=paper-preview MM-mixer.tex`. The distinct job
name avoids a collision between the TeX output `MM-mixer.pdf` and the
referenced figure `MM-Mixer.pdf` on some TeX setups. The AAAI style,
bibliography, bibliography style, and referenced architecture figure
are included here.
