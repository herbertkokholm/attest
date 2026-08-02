"""Vendor adapters for the LLM ensemble used in screening.

`base` defines the `Rater` protocol, `DeterministicRater`, and
`run_ensemble`, all importable with no optional dependency installed.
Live provider adapters live under `vendors.providers` and are gated behind
per-vendor extras; `registry.build_raters` wires a `Config` up to them.
"""
