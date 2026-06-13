import torch


def resolve_device(preferred: str = None) -> str:
    """cuda > mps (Apple GPU) > cpu. Pass --device cpu if an MPS op is unsupported."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
