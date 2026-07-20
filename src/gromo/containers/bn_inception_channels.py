"""Frozen BN-Inception channel tables for Net2Net §3.2 (√0.3 teacher).

Named reference
---------------
- Full-width student: GoogLeNet / Inception module widths from Szegedy et al.
  2014 Table 1 (the BN-Inception topology of Ioffe & Szegedy 2015 uses these
  Inception-module channel counts; stem / pool / classifier unchanged).
- **Not** torchvision ``inception_v3``.
- Net2Net (Chen et al. 2015) §3.2: reduce convolution channels *inside all
  Inception modules* by ``sqrt(0.3)``; do not modify other components.

Note: Net2Net PDF "Appendix B" is related work, not a channel table. The
freeze below is the operational App-B-style reference derived from the
§3.2 recipe + Szegedy Table 1.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# Width scale applied inside Inception modules only (Net2Net §3.2).
INCEPTION_WIDTH_SCALE = math.sqrt(0.3)  # ≈ 0.5477225575

# Allowed band for teacher / student *Inception-module* parameter ratio.
# Paper: √0.3 width → ~30% params when both in/out channels scale.
PARAM_RATIO_TARGET = 0.30
PARAM_RATIO_BAND = (0.22, 0.38)


@dataclass(frozen=True)
class InceptionModuleChannels:
    """Per-module branch widths (Szegedy Table 1 / BN-Inception)."""

    name: str
    n1x1: int
    n3x3_reduce: int
    n3x3: int
    n5x5_reduce: int
    n5x5: int
    pool_proj: int

    @property
    def out_channels(self) -> int:
        """Total concat width of the four Inception branches."""
        return self.n1x1 + self.n3x3 + self.n5x5 + self.pool_proj


# Full-width student Inception modules (stem / pool / classifier excluded).
FULL_INCEPTION_MODULES: tuple[InceptionModuleChannels, ...] = (
    InceptionModuleChannels("3a", 64, 96, 128, 16, 32, 32),
    InceptionModuleChannels("3b", 128, 128, 192, 32, 96, 64),
    InceptionModuleChannels("4a", 192, 96, 208, 16, 48, 64),
    InceptionModuleChannels("4b", 160, 112, 224, 24, 64, 64),
    InceptionModuleChannels("4c", 128, 128, 256, 24, 64, 64),
    InceptionModuleChannels("4d", 112, 144, 288, 32, 64, 64),
    InceptionModuleChannels("4e", 256, 160, 320, 32, 128, 128),
    InceptionModuleChannels("5a", 256, 160, 320, 32, 128, 128),
    InceptionModuleChannels("5b", 384, 192, 384, 48, 128, 128),
)

# Stem / classifier (unchanged under √0.3; Net2Net §3.2).
STEM_CONV1_OUT = 64
STEM_CONV2_OUT = 64
STEM_CONV3_OUT = 192
CLASSIFIER_IN = 1024  # inception 5b full out


def round_inception_channels(width: int, scale: float = INCEPTION_WIDTH_SCALE) -> int:
    """Round-to-nearest with floor at 1 (bit-reproducible across builders).

    Rule: ``max(1, int(round(width * scale)))``.
    Ties follow Python 3 ``round`` (banker's rounding on .5).
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    return max(1, round(width * scale))


def scale_module(
    module: InceptionModuleChannels, scale: float = INCEPTION_WIDTH_SCALE
) -> InceptionModuleChannels:
    """Apply √0.3 (or other) scale to all branch widths of one module."""
    return InceptionModuleChannels(
        name=module.name,
        n1x1=round_inception_channels(module.n1x1, scale),
        n3x3_reduce=round_inception_channels(module.n3x3_reduce, scale),
        n3x3=round_inception_channels(module.n3x3, scale),
        n5x5_reduce=round_inception_channels(module.n5x5_reduce, scale),
        n5x5=round_inception_channels(module.n5x5, scale),
        pool_proj=round_inception_channels(module.pool_proj, scale),
    )


def teacher_inception_modules(
    scale: float = INCEPTION_WIDTH_SCALE,
) -> tuple[InceptionModuleChannels, ...]:
    """Teacher channel table: √0.3 inside Inception modules only."""
    return tuple(scale_module(m, scale) for m in FULL_INCEPTION_MODULES)


def channel_tables_as_dicts(
    scale: float = INCEPTION_WIDTH_SCALE,
) -> dict[str, list[dict]]:
    """Publish explicit teacher / full tables for docs and Hydra."""
    return {
        "full_student": [asdict(m) for m in FULL_INCEPTION_MODULES],
        "teacher_sqrt_0_3": [asdict(m) for m in teacher_inception_modules(scale)],
        "rounding_rule": "max(1, int(round(width * sqrt(0.3))))",
        "width_scale": scale,
        "param_ratio_band": list(PARAM_RATIO_BAND),
        "stem_unchanged": {
            "conv1_out": STEM_CONV1_OUT,
            "conv2_out": STEM_CONV2_OUT,
            "conv3_out": STEM_CONV3_OUT,
        },
    }


def assert_teacher_param_ratio(
    teacher_inception_params: int,
    student_inception_params: int,
    *,
    band: tuple[float, float] = PARAM_RATIO_BAND,
) -> float:
    """Fail construction if teacher/student Inception-param ratio outside band."""
    if student_inception_params <= 0:
        raise ValueError("student_inception_params must be positive.")
    ratio = teacher_inception_params / student_inception_params
    lo, hi = band
    if not (lo <= ratio <= hi):
        raise AssertionError(
            f"Teacher/student Inception param ratio {ratio:.4f} outside "
            f"documented band [{lo}, {hi}] (target ~{PARAM_RATIO_TARGET}). "
            f"teacher={teacher_inception_params}, student={student_inception_params}."
        )
    return ratio
