"""Pre-inference image quality gate for industrial camera frames.

Motivated by Phase 4 robustness analysis: extreme under-exposure
(``brightness_down``) or severe sensor noise shifts PatchCore feature distances
upward and triggers false-alarm overkill. Checking mean brightness, contrast,
and Laplacian focus variance upfront lets the system flag lighting/camera faults
explicitly in the inspection response and traceability log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityReport:
    """Quality metrics and warnings for a single input image."""

    passed: bool
    mean_brightness: float
    contrast_std: float
    blur_laplacian_var: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_image_quality(
    img_bgr: np.ndarray,
    *,
    min_brightness: float = 20.0,
    max_brightness: float = 240.0,
    min_contrast_std: float = 8.0,
    min_laplacian_var: float = 15.0,
) -> ImageQualityReport:
    """Evaluate brightness, contrast, and focus sharpness of a BGR uint8 image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    mean_b = float(np.mean(gray))
    std_c = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings: list[str] = []
    if mean_b < min_brightness:
        warnings.append(f"too_dark (mean={mean_b:.1f} < {min_brightness})")
    elif mean_b > max_brightness:
        warnings.append(f"too_bright (mean={mean_b:.1f} > {max_brightness})")
    if std_c < min_contrast_std:
        warnings.append(f"low_contrast (std={std_c:.1f} < {min_contrast_std})")
    if lap_var < min_laplacian_var:
        warnings.append(f"blurry (laplacian_var={lap_var:.1f} < {min_laplacian_var})")

    return ImageQualityReport(
        passed=len(warnings) == 0,
        mean_brightness=round(mean_b, 2),
        contrast_std=round(std_c, 2),
        blur_laplacian_var=round(lap_var, 2),
        warnings=warnings,
    )
