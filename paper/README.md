# IECON 2026 final-submission source

`main_final_submission.tex` is the only manuscript source in this package.
It contains the final scientific-polish revisions: the central-baseline scope
is explicit, the packet-loss and drag-interaction claims have formal tests,
and a state-replicating hot-standby comparator is evaluated across three
pre-specified failover timeouts.

## Compile before uploading

Compile with PDFLaTeX, BibTeX, then PDFLaTeX twice:

```text
pdflatex main_final_submission
bibtex main_final_submission
pdflatex main_final_submission
pdflatex main_final_submission
```

Do not upload a previous PDF: compile this source, inspect the result, and
then run that exact PDF through the IECON/IEEE-required PDF validation process.
The resulting paper must remain no more than six pages and below the portal's
file-size limit.

## Included files

- `main_final_submission.tex`: final manuscript source.
- `references_IECON.bib`: bibliography used by the manuscript.
- `pso_convergence_rates.pdf`, `pso_swarm_trajectories.pdf`: original result figures.
- `comm_fragility_sweep.pdf`: verified 500-trial communication-fragility figure.
- `FINAL_SUBMISSION_CHECKLIST.md`: final metadata and upload checks.
- `REVIEWER_RESPONSE.md`: mapping from review comments to manuscript changes.
- `VERIFICATION_RECORD.md`: completed static/test audits and the remaining PDF checks.
