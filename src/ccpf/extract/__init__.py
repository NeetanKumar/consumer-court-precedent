"""Stage 2 — structured extraction.

Realizes the project's core insight: a judgment's usefulness for answering
"what have similar consumers been awarded" comes from structured facts, not
raw text. See schema.py for the Judgment record, extractor.py for the
tiered Haiku-4.5/Sonnet-5 extraction strategy, and run.py for the resumable
orchestrator.

Hand-labeled validation set (~30-50 docs) for measuring extraction accuracy
is not yet built — do that before trusting extraction output at scale.
"""
