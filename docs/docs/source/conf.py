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
from qdpmc import __version__ as v
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../../qdpmc/src/'))
from numpydoc.xref import DEFAULT_LINKS

# -- Project information -----------------------------------------------------

project = 'QdpMC'
copyright = '2021, Yield Chain Technology Co., Ltd.'
author = 'YC'

# The full version, including alpha/beta/rc tags
release = v


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.extlinks',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
pygments_style = 'sphinx'
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = dict(
    collapse_navigation=False,
    navigation_depth=4
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# IPython configuration options
ipython_execlines = [
    'import qdpmc as qm',
    'import numpy as np',
    'import datetime'
]
autoclass_content = 'both'

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# Napoleon options
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_include_init_with_doc = False  # move init doc to 'class' doc
napoleon_type_aliases = DEFAULT_LINKS.copy()
napoleon_type_aliases.update(
    {'Payoff': ':class:`qdpmc.tools.payoffs.Payoff`',
     'Engine': ':class:`qdpmc.engine.monte_carlo.MonteCarlo`',
     'Heston': ':class:`qdpmc.model.market_process.Heston`',
     'BlackScholes': ':class:`qdpmc.model.market_process.BlackScholes`',
     'Calendar': ':class:`qdpmc.dateutil.date.Calendar`'
     }
)


intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'python': ('https://docs.python.org/3/', None),
}
