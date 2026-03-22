# Changelog

## v0.1.5 (2026-03-21) [unreleased]

## v0.1.5 (2026-03-21)
- docs: replace pip install with uv add in README
- make torch optional: move to [neural] extra with lazy imports
- Add blog post link and community CTA to README
- Fix benchmark: remove sklearn import to avoid numpy/scipy version conflicts
- Add benchmark: R2VF factor clustering vs manual quintile banding for territory rating
- QA audit fixes: P0 and P1 issues resolved, bump to 0.1.4
- benchmarks: Databricks R2VF vs manual quintile banding, honest results
- Fix P0/P1 bugs: δ₁ penalty, embedding offset, territory predict, BIC k_eff
- Pin statsmodels>=0.14.5 to fix scipy _lazywhere removal
- Add shields.io badge row to README
- Add Quick Start section to README
- docs: add Databricks notebook link
- Add Related Libraries section to README
- Reconcile insurance-glm-cluster unique functions into cluster subpackage

