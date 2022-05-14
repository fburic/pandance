import os
import sys
from datetime import datetime

import pandance

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project = 'Pandance'
copyright = f'2021-{datetime.now().year}, Filip Buric'
author = 'Filip Buric'
language = 'en'
version = pandance.__version__


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              "sphinx.ext.autosummary",
              'sphinx_autodoc_typehints',
              "sphinx.ext.todo",
              'sphinx_toggleprompt']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/fburic/panda-grove',
    'navbar_end': ['navbar-icon-links'],
}
# html_logo = '../img/pandance_logo.svg'
# html_favicon = '../img/pandance_logo.svg'
# html_sidebars = {'**': ['localtoc.html', 'searchbox.html'] }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
