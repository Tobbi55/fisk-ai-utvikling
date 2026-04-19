"""Application-wide UI styling.

Keep this minimal and theme-friendly: prefer native palettes over hard-coded colors.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication


def apply_app_style(app: QApplication) -> None:
    """Apply a modern, cross-platform Qt style and small UI polish."""

    # Fusion tends to look more consistent/modern across Linux environments.
    app.setStyle("Fusion")

    # Minimal stylesheet: spacing and typography only (no hard-coded colors).
    app.setStyleSheet(
        "\n".join(
            [
                "QWidget { font-size: 11pt; }",
                "QPushButton { padding: 6px 12px; min-height: 30px; }",
                "QComboBox, QSpinBox, QLineEdit { min-height: 28px; }",
            ]
        )
    )
