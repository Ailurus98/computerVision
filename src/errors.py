"""Custom exceptions for the stereo pipeline."""


class CalibrationError(Exception):
    """Raised when camera calibration cannot be performed (e.g. too few images)."""
