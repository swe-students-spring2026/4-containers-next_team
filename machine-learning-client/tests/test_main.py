"""Test cases for the ML client."""

from main import analyze_posture


def test_analyze_posture():
    """Test that analyze_posture returns True."""
    assert analyze_posture() is True
