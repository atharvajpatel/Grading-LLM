"""Tests for scales module."""

import pytest
from grading_llm.scales import (
    SCALES,
    SCALE_ORDER,
    get_scale,
    validate_response,
    format_scale_instruction,
    is_discrete,
    get_scale_values,
)


class TestScales:
    def test_all_scales_exist(self):
        """All expected scales should be defined."""
        assert "binary" in SCALES
        assert "ternary" in SCALES
        assert "quaternary" in SCALES
        assert "continuous" in SCALES

    def test_scale_order(self):
        """Scale order should be correct."""
        assert SCALE_ORDER == ["binary", "ternary", "quaternary", "continuous"]

    def test_get_scale_valid(self):
        """get_scale should return scale for valid names."""
        scale = get_scale("binary")
        assert scale is not None
        assert scale.name == "binary"

    def test_get_scale_invalid(self):
        """get_scale should raise for invalid names."""
        with pytest.raises(ValueError):
            get_scale("invalid_scale")


class TestValidateResponse:
    def test_binary_valid(self):
        """Binary scale should accept 0 and 1."""
        assert validate_response(0, "binary") == 0
        assert validate_response(1, "binary") == 1

    def test_binary_snapping(self):
        """Binary scale should snap to nearest value."""
        assert validate_response(0.2, "binary") == 0
        assert validate_response(0.8, "binary") == 1
        assert validate_response(0.5, "binary") in [0, 1]  # Could go either way

    def test_ternary_valid(self):
        """Ternary scale should accept 0, 0.5, 1."""
        assert validate_response(0, "ternary") == 0
        assert validate_response(0.5, "ternary") == 0.5
        assert validate_response(1, "ternary") == 1

    def test_quaternary_snapping(self):
        """Quaternary scale should snap correctly."""
        assert validate_response(0, "quaternary") == 0
        assert validate_response(0.33, "quaternary") == 0.33
        assert validate_response(0.5, "quaternary") in [0.33, 0.66]
        assert validate_response(1, "quaternary") == 1

    def test_continuous_clamping(self):
        """Continuous scale should clamp to [0, 1]."""
        assert validate_response(0.5, "continuous") == 0.5
        assert validate_response(-0.5, "continuous") == 0.0
        assert validate_response(1.5, "continuous") == 1.0


class TestScaleInstructions:
    def test_binary_instruction(self):
        """Binary instruction should mention 0 and 1."""
        instr = format_scale_instruction("binary")
        assert "0" in instr
        assert "1" in instr

    def test_continuous_instruction(self):
        """Continuous instruction should mention range."""
        instr = format_scale_instruction("continuous")
        assert "0" in instr
        assert "1" in instr
        assert "decimal" in instr.lower() or "between" in instr.lower()


class TestScaleProperties:
    def test_discrete_scales(self):
        """Binary, ternary, quaternary should be discrete."""
        assert is_discrete("binary")
        assert is_discrete("ternary")
        assert is_discrete("quaternary")

    def test_continuous_not_discrete(self):
        """Continuous should not be discrete."""
        assert not is_discrete("continuous")

    def test_get_scale_values_binary(self):
        """Binary should have values [0, 1]."""
        values = get_scale_values("binary")
        assert values == [0, 1]

    def test_get_scale_values_ternary(self):
        """Ternary should have values [0, 0.5, 1]."""
        values = get_scale_values("ternary")
        assert values == [0, 0.5, 1]

    def test_get_scale_values_continuous_error(self):
        """Continuous should raise when getting values."""
        with pytest.raises(ValueError):
            get_scale_values("continuous")
