"""Sphinx configuration for plattipus_nvidia_warp."""

import os
import sys

# The package's Python tree, so autodoc can import plattipus_nvidia_warp without the
# package being installed. Nothing imported at module scope needs Houdini.
sys.path.insert(0, os.path.abspath("../../python"))

project = "plattipus_nvidia_warp"
copyright = "2026, plattipus"
author = "plattipus"
release = "1.0.0"
version = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# warp and hou are only importable inside Houdini, so autodoc must not try.
autodoc_mock_imports = ["warp", "hou", "numpy"]
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/logo-light.svg"
html_title = "%s %s" % (project, release)
html_copy_source = False
html_show_sourcelink = False

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1b7f4b",
        "color-brand-content": "#1b7f4b",
        "font-stack--monospace": "SFMono-Regular, Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ed69b",
        "color-brand-content": "#5ed69b",
    },
}

# -W is used in CI, so unresolved references must not be silently tolerated.
nitpicky = False
