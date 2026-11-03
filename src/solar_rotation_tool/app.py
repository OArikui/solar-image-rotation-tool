"""Application entry point."""

from solar_rotation_tool.qt_gui import run_rotation_tool


def main() -> int:
    """Start the GUI application."""
    return run_rotation_tool()
