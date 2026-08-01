"""Error correlation: correlated false negatives across ensemble members on the relevant class."""


def pairwise_fn_correlation(errors: dict[str, list[bool]]) -> dict[str, float]:
    """Compute pairwise false-negative correlation between ensemble members (not implemented)."""
    raise NotImplementedError
