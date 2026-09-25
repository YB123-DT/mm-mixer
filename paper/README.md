# Manuscript base

`MM-mixer.tex` is an exact, byte-for-byte copy of the author-supplied
`/data2/yb/MM-mixer/MM-mixer.tex` (SHA256
`7e9d5470c30331704a1baadfa9031906cbda5c0f322bb4fbefe3f6646f447064`).
No scores, standard deviations, seeds, loss formulas, or ablation values
were edited for this upload.

The manuscript is a **base draft**, not a result-aligned description of
the corrected Full code and peak-test bundles in this repository. The
manuscript's MELD main row and Full ablation row do not match those bundles.
Its own table values should therefore be read as the author's supplied
draft values, not silently attributed to the corrected Full release.

Build from this directory with
`latexmk -pdf -jobname=paper-preview MM-mixer.tex`. The distinct job
name avoids a collision between the TeX output `MM-mixer.pdf` and the
referenced figure `MM-Mixer.pdf` on some TeX setups. The AAAI style,
bibliography, bibliography style, and referenced architecture figure
are included here.
