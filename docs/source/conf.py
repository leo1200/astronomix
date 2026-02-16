# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# sys.path.insert(0, os.path.abspath('../../astronomix'))  # Source code dir relative to this file

# import astronomix

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'astronomix'
copyright = '2025'
author = 'Leonard Storcks at <a href="https://astroai-lab.de/index.html">AstroAi Lab</a>.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [

    # automatic API documentation
    'autodoc2',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinxcontrib.apidoc',

    # enable google style docstrings
    'sphinx.ext.napoleon',

    # myst parser with notebook support
    'myst_nb',
    'sphinx_copybutton',

    # further extensions
    'sphinx_design',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
]


apidoc_module_dir = '../../astronomix'
apidoc_output_dir = 'source'
apidoc_separate_modules = True

autodoc2_packages = [
    "../../astronomix",
]

autodoc2_hidden_objects = ["inherited", "private"]
autodoc2_module_summary = True

myst_amsmath_enable = True
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",

]

autodoc_default_flags = ['members']
autosummary_generate = True

napolean_use_rtype = False

# templates_path = ['_templates']
exclude_patterns = []

# notebook compilation
nb_execution_mode = "off"
# nb_execution_excludepatterns = ['wind_parameter_optimization']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']

html_logo = "astronomix_logo.png"

html_favicon = 'icon.svg'

html_title = "astronomix"

html_theme_options = {
    "repository_url": "https://github.com/leo1200/astronomix",
    "use_repository_button": True,
}