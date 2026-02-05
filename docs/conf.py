# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from pytrunc.constant import VERSION


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pytrunc'
copyright = '2026, Pytrunc team'
author = 'Pytrunc team'
release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'myst_parser',
              'nbsphinx',
              'sphinx.ext.graphviz',
              'numpydoc']
              #"sphinx.ext.napoleon"]

templates_path = ['_templates']

# autodoc_default_flags = ['members', 'show-inheritance', 'special-members']

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
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'#'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {"default_mode": "light"}