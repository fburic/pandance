import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

import pandance


# -- Project information -----------------------------------------------------
project = 'Pandance'
copyright = f'CC BY-SA 4.0 2022-{datetime.now().year}, Filip Buric'
author = 'Filip Buric'
language = 'en'
version = pandance.__version__


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              "sphinx.ext.autosummary",
              'sphinx_autodoc_typehints',
              "sphinx.ext.mathjax",
              'sphinx_togglebutton',
              "sphinx.ext.todo",
              'sphinx_toggleprompt']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/fburic/pandance',
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'secondary_sidebar_items': ['page-toc'],
    'show_nav_level': 2,
    'show_toc_level': 2,

    "pygment_light_style": 'tango',
    "pygment_dark_style": "monokai"
}

html_logo = 'img/pandance_logo.svg'
html_favicon = 'img/pandance_logo.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css'
]
