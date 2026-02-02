"""
Grading Scales

Defines the 4 answer granularities:
- Binary: {0, 1}
- Ternary: {0, 0.5, 1}
- Quaternary: {0, 0.33, 0.66, 1}
- Continuous: [0, 1]
"""

from typing import Any, Union, List, Tuple
from dataclasses import dataclass


@dataclass
class DiscreteScale:
    """A discrete scale with fixed valid values."""
    name: str
    values: List[float]
    labels: List[str]  # Semantic labels explaining what each value means

    def validate(self, value: Any) -> float:
        """Validate and snap to nearest valid value."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert {value} to float")

        # Find nearest valid value
        closest = min(self.values, key=lambda x: abs(x - v))
        return closest

    def format_instruction(self) -> str:
        """Format scale instruction with semantic labels."""
        options = [f"{v} = {label}" for v, label in zip(self.values, self.labels)]
        instruction = "SCORING SCALE - Score based on how DEEPLY and RELEVANTLY the feature appears:\n"
        instruction += "\n".join(f"  - {opt}" for opt in options)
        if len(self.values) > 2:
            instruction += f"\n\nUse intermediate values ({', '.join(str(v) for v in self.values[1:-1])}) when the feature is:\n"
            instruction += "  - Present but peripheral/tangential to the main point\n"
            instruction += "  - Implicit or suggested rather than explicit\n"
            instruction += "  - Weakly expressed or only partially applicable"
        return instruction


@dataclass
class ContinuousScale:
    """A continuous scale with a range."""
    name: str
    min_val: float
    max_val: float

    def validate(self, value: Any) -> float:
        """Validate and clamp to range."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert {value} to float")

        # Clamp to range
        return max(self.min_val, min(self.max_val, v))

    def format_instruction(self) -> str:
        """Format scale instruction for continuous 0-1 scoring."""
        return (
            f"SCORING SCALE - Rate how STRONGLY this feature appears on a continuous 0-1 scale:\n"
            f"  - 0.0 = completely absent, no trace of this feature\n"
            f"  - 0.1-0.3 = barely present, very minor or tangential\n"
            f"  - 0.4-0.6 = moderately present, somewhat relevant but not central\n"
            f"  - 0.7-0.9 = strongly present, clearly relevant and significant\n"
            f"  - 1.0 = deeply and centrally present, core to the statement\n\n"
            f"You may use ANY decimal value between 0.0 and 1.0 (e.g., 0.15, 0.42, 0.87).\n"
            f"This is a CONTINUOUS scale - use the full range, not just 0 or 1."
        )


Scale = Union[DiscreteScale, ContinuousScale]

# Define the 4 scales with semantic labels
SCALES: dict[str, Scale] = {
    "binary": DiscreteScale(
        "binary",
        [0, 1],
        ["Absent - not present in the statement",
         "Present - appears in the statement"]
    ),
    "ternary": DiscreteScale(
        "ternary",
        [0, 0.5, 1],
        ["Absent - not present at all",
         "Peripheral - present but minor, tangential, or implicit",
         "Central - clearly and explicitly present"]
    ),
    "quaternary": DiscreteScale(
        "quaternary",
        [0, 0.33, 0.66, 1],
        ["Absent - not present",
         "Weak - barely present, very minor or implicit",
         "Moderate - present but not emphasized",
         "Strong - clearly present and significant"]
    ),
    "continuous": ContinuousScale("continuous", 0.0, 1.0),
}

# Order for experiments (progression from coarse to fine)
SCALE_ORDER = ["binary", "ternary", "quaternary", "continuous"]


def get_scale(name: str) -> Scale:
    """Get a scale by name."""
    if name not in SCALES:
        raise ValueError(f"Unknown scale: {name}. Valid scales: {list(SCALES.keys())}")
    return SCALES[name]


def validate_response(value: Any, scale_name: str) -> float:
    """
    Validate a response value for a given scale.

    Args:
        value: The value to validate
        scale_name: Name of the scale

    Returns:
        Validated (and possibly snapped/clamped) float value
    """
    scale = get_scale(scale_name)
    return scale.validate(value)


def format_scale_instruction(scale_name: str) -> str:
    """
    Get the prompt instruction for a scale.

    Args:
        scale_name: Name of the scale

    Returns:
        Instruction string for the prompt
    """
    scale = get_scale(scale_name)
    return scale.format_instruction()


def is_discrete(scale_name: str) -> bool:
    """Check if a scale is discrete (vs continuous)."""
    return isinstance(get_scale(scale_name), DiscreteScale)


def get_scale_values(scale_name: str) -> List[float]:
    """Get the valid values for a discrete scale."""
    scale = get_scale(scale_name)
    if isinstance(scale, DiscreteScale):
        return scale.values
    else:
        # For continuous, return None or raise
        raise ValueError(f"Scale {scale_name} is continuous, not discrete")
