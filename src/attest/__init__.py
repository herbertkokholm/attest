"""attest: a dependency-light kernel for LLM-ensemble evidence screening.

This package implements the screening-and-self-validation kernel for an
LLM-based evidence-screening method used in systematic-review title and
abstract screening. It defines the input and validation-record contracts,
the deterministic prefilter framework, and (in later commits) the ensemble,
statistically separated planes, provenance tracking, and statistics used to
self-validate screening quality.

Boundary rule: this kernel never imports or references collectors,
schedulers, HTTP, object storage, databases, multi-tenancy, web frameworks,
or UI. Those concerns belong to the product shell that imports ``attest``,
not to ``attest`` itself.
"""

__version__ = "0.1.0"
