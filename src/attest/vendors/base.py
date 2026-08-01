"""Vendor adapter base interface for calling ensemble members from different LLM vendors."""


def get_vendor_adapter(name: str) -> None:
    """Look up a vendor adapter by name (not yet implemented)."""
    raise NotImplementedError
