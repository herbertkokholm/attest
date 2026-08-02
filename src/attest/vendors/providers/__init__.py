"""Live LLM vendor adapters implementing the `Rater` protocol.

Each adapter module lazily imports its vendor SDK inside `rate`/`_client`,
never at module import time, so `import attest` -- and even
`import attest.vendors.providers.<vendor>` -- succeeds without that vendor's
optional extra installed. Only calling `rate()` on an adapter requires the
corresponding SDK to be present.
"""
