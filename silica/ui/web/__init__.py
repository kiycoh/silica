"""Optional localhost GUI (`silica --gui`). Requires the `[gui]` extra."""
from .server import serve

__all__ = ["serve"]
