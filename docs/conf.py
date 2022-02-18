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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'porter'
copyright = '2022, Cadent Data Science'
author = 'Cadent Data Science'

# The full version, including alpha/beta/rc tags
import porter
release = porter.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    #'autodocsumm',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

# show todos (for now)
todo_include_todos = True
todo_emit_warnings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# Don't try to import pandas or numpy
autodoc_mock_imports = ['pandas', 'numpy']


# -- Options for HTML output -------------------------------------------------


# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
import os
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'                           
                                                                                 
if not on_rtd:  # only import and set the theme if we're building docs locally   
    import sphinx_rtd_theme                                                      
    html_theme = 'sphinx_rtd_theme'                                              
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]                   
    # Override default css to get a larger width for local build                 
    def setup(app):                                                              
        #app.add_javascript("custom.js")                                         
        app.add_stylesheet('theme_overrides.css')                                
else:                                                                            
    # Override default css
    html_context = {
        'css_files': [
            'https://media.readthedocs.org/css/sphinx_rtd_theme.css',
            'https://media.readthedocs.org/css/readthedocs-doc-embed.css',
            '_static/theme_overrides.css',
        ],                                                                       
    }



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
