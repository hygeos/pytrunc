# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the
# documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from pytrunc.constant import VERSION

# -- Project information -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'Pytrunc'
copyright = '2026, Pytrunc team'
author = 'Pytrunc team'
release = VERSION

# -- General configuration ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

extensions = ['sphinx.ext.todo',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'myst_parser',
              'nbsphinx',
              'sphinx.ext.graphviz',
              'numpydoc']
              #"sphinx.ext.napoleon"]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*constant.rst',
                    'modules.rst']

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'special-members': '__call__'
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
# -- Options for HTML output -------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

html_theme = 'pydata_sphinx_theme'  # or 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo': {
        'image_light': '_static/pytrunc-logo-horizontal-light-bg.png',
        'image_dark': '_static/pytrunc-logo-horizontal-dark-bg.png',
    },
}

html_context = {"default_mode": "light"}
