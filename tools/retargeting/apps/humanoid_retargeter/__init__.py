"""Shared PyRoki retargeting from articulated human motion to Humanoid Light v2."""

__all__ = ["RetargetResult", "retarget_gemx_hpe", "retarget_motion_clip"]


def __getattr__(name: str):
    """Load the solver only when a caller requests retargeting or export types."""
    if name == "RetargetResult":
        from .shared import RetargetResult

        return RetargetResult
    if name in {"retarget_gemx_hpe", "retarget_motion_clip"}:
        from . import pyroki_retargeter

        return getattr(pyroki_retargeter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
