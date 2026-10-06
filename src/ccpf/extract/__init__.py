"""Stage 2 — structured extraction.

Realizes the project's core insight: a judgment's usefulness for answering
"what have similar consumers been awarded" comes from structured facts, not
raw text. See schema.py for the Judgment record, extractor.py for the
tiered Haiku-4.5/Sonnet-5 extraction strategy, and run.py for the resumable
orchestrator.

Extraction accuracy is measured against a hand-labeled validation set —
see validation_run.py (re-run extraction on the labeled docs) and
scoring.py (field-level accuracy), driven by scripts/score_validation.py.
"""
