# Experiment logging

Use `experiments/templates/run_card.md` for every non-trivial run.

Recommended convention:

- one run card per training / sweep / evaluation job
- record exact config paths and checkpoint paths
- separate **claim-facing** runs from **debug / sanity** runs
- explicitly mark whether a result belongs to
  - shared decoding probe
  - direct baseline
  - robustness / stronger-adapter control
