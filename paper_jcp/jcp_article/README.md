# JCP article

This directory contains the working manuscript for submission to the
Journal of Computational Physics.

The template was downloaded from the official Elsevier LaTeX instructions on
2026-08-10:

https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions

The working files use elsarticle 3.4 (2024-04-04) with numbered references. The
untouched downloaded bundle and its documentation are stored in template/.

Regenerate the result tables, scaling curves, advantage map, and copied
qualitative panels from the final experiment namespaces:

    cd /path/to/pinn-small-data-jcp
    python3 -B paper_jcp/jcp_article/build_article_assets.py

The builder audits 720 result rows and writes the exact merged result set used
by the paper to generated/final_results_used.csv. Test 2 PINN rows come from
corrected_test2_final; its classical rows are replaced by the stronger
corrected_test2_classical_strong time-CV results. Tests 3 and 4 already contain
the strong time-CV baselines.

Compile the manuscript from this directory:

    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

The manuscript body is split into sections/. Generated tables are under
generated/, publication figures are under figures/, and the compiled article
is main.pdf.
