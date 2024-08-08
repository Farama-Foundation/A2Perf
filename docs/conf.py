# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath("../a2perf"))
sys.path.insert(0, os.path.abspath("../a2perf/domains/circuit_training"))
sys.path.insert(0, os.path.abspath("../a2perf/domains/quadruped_locomotion"))
sys.path.insert(0, os.path.abspath("../a2perf/domains/web_navigation"))


# -- Project information -----------------------------------------------------
from typing import Any, Dict

# TODO: Replace A2Perf, remove comment and remove this line
# import A2Perf


project = "A2Perf"
copyright = "2022 Farama Foundation"
author = "Farama Foundation"

# The full version, including alpha/beta/rc tags
# TODO: Replace A2Perf, remove comment and remove this line
# release = A2Perf.__version__
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "myst_parser",
    "nbsphinx",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "a2perf/a2perf_benchmark_submission",
    "a2perf/analysis",
    "a2perf/launch",
]
autodoc_mock_imports = [
    "a2perf.analysis.evaluation",
    "a2perf.submission.main_submission",
]

# Napoleon settings
# napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "A2Perf Documentation"
html_baseurl = "https://a2perf.farama.org"
html_copy_source = False
html_favicon = "_static/img/logo/favicon/A2Perf-favicon.png"
html_theme_options = {
    "light_logo": "img/logo/vectors/A2Perf.svg",
    "dark_logo": "img/logo/vectors/A2Perf-white.svg",
    "gtag": "G-6H9C8TWXZ8",
    "description": "A2Perf is a benchmarking suite for evaluating autonomous agents on real-world problems.",
    "image": "_static/img/logo/vectors/A2Perf-github.svg",
    "versioning": True,
    "source_repository": "https://github.com/Farama-Foundation/A2Perf/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
html_css_files = []

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
