"""Sphinx configuration for the PyTorch-Chipyard documentation."""

project = "PyTorch-Chipyard"
author = "PyTorch-Chipyard contributors"
copyright = "2026, PyTorch-Chipyard contributors"

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "PyTorch-Chipyard Documentation"

myst_heading_anchors = 3
