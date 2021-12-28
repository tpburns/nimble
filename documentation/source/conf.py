# -*- coding: utf-8 -*-
#
# Nimble documentation build configuration file, created by
# sphinx-quickstart on Wed Nov 11 14:47:51 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

from __future__ import absolute_import
import sys
import os
import inspect
import re
import json
from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
confFilePath = os.path.abspath(inspect.getfile(inspect.currentframe()))
NimbleParentDirPath = os.path.dirname(os.path.dirname(os.path.dirname(confFilePath)))
sys.path.insert(0, NimbleParentDirPath)

os.environ['PYTHONPATH'] = NimbleParentDirPath

# -- General configuration ------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.3'

# Sphinx extension functions
def process_docstring(app, what, name, obj, options, lines):
    """
    This is a workaround to allow sphinx to show functions from their __init__
    module location, not their exact source. For example, nimble.calculate.mean
    instead of nimble.calculate.statistic.mean.
    """
    if what == 'module' and options.get('noindex', False):
        del lines[:]
    if what in ['function', 'method', 'attribute', 'property']:
        # remove keywords header in documentation in favor of normal text
        # all changes to lines must be in-place
        for i, line in enumerate(lines):
            if line == '.. rubric:: Keywords':
                lines[i] = 'Keywords: '
                if not lines[i + 1]:
                    del lines[i + 1]
                break


def process_signature(app, what, name, obj, options, signature,
                      return_annotation):
    """
    Prevents Sphinx from showing the signature for classes that are not
    designed to be constructed by the user.
    """
    keepSignatures = ['nimble.Tune', 'nimble.Init', 'nimble.Tuning']
    if what == 'class' and name not in keepSignatures:
        signature = ''
    # all nan values display as "nan", so we need to be more specific.
    if what == 'function' and name == 'nimble.data':
        signature = re.sub('nan, nan', 'float("nan"), numpy.nan', signature)
        signature = re.sub('=nan', '=numpy.nan', signature)

    return signature, return_annotation

def capitalizer(string):
    if string in ['and']:
        return string
    return string.capitalize()

def setHyperlinks(app):
    """
    Use the generated stubs to determine hyperlinks for examples.
    """
    exampleLinks = {}
    for file in os.listdir(os.path.join('source', 'examples')):
        if file.endswith('.py'):
            example = file[:-3]
            key = ' '.join(map(capitalizer, example.split('_'))) + ' example'
            exampleLinks[key] = "{}.html".format(example)

    app.nimble_examples = exampleLinks

    hyperlinks = {}
    markdownOnly = {}
    for _, _, files in os.walk(os.path.join('source', 'docs', 'generated')):
        for rst in files:
            path = rst[:-4]
            name = None
            # define hyperlinks to custom files
            if path == 'nimble':
                hyperlinks[path] = '../docs/index.html'
                continue
            if path == 'nimble.calculate':
                hyperlinks[path] = '../docs/nimble.calculate.html'
                continue
            # any remaining paths are in docs/generated
            pathSplit = path.split('.')
            link = '../docs/generated/{}.html'.format(path)
            # classes
            if re.match(r'[A-Z]', pathSplit[-1][0]):
                className = pathSplit[-1]
                if className in ['Tune', 'Init', 'CustomLearner', 'Tuning']:
                    # leave 'nimble.' prefix, class name for markdown only
                    name = path
                    markdownOnly[className] = link
                else:
                    name = className
            # class methods
            elif re.search(r'[A-Z][a-z]+\..*', path):
                isClass = list(map(lambda s: bool(re.match(r'[A-Z]', s[0])),
                               pathSplit))
                clsIdx = isClass.index(True)
                className = pathSplit[clsIdx]
                if className == 'SessionConfiguration':
                    name = 'nimble.settings.' + pathSplit[-1]
                else:
                    if className in ['Points', 'Features']:
                        pathSplit[clsIdx] = className.lower()
                    else:
                        clsIdx += 1
                    name = '.' + '.'.join(pathSplit[clsIdx:])
            # functions
            else:
                name = path
                # add object without prefix for markdown
                markdownOnly[name.split('.')[-1]] = link

            # sanity check
            if name is None:
                raise ValueError('name was not set for ' + path)

            # conflicting name; will require custom mapping
            if '.' + name in hyperlinks:
                hyperlinks['.' + name] = None
            elif name.startswith('.') and name[1:] in hyperlinks:
                hyperlinks[name[1:]] = None
            if name in hyperlinks:
                hyperlinks[name] = None
            else:
                hyperlinks[name] = link

    app.nimble_hyperlinks = hyperlinks
    app.nimble_markdown = markdownOnly

def addStringReplacements(original, replacements):
    """
    Create a new string from original, replacing the span at index 0 of
    replacements with the string at index 1 of replacements.
    """
    start = 0
    newString = ''
    for replacement in replacements:
        end = replacement[0][0]
        newString += original[start:end]
        newString += replacement[1]
        start = replacement[0][1]
    newString += original[start:]

    return newString

def getHyperlink(app, value):
    if app.nimble_hyperlinks[value] is None:
        return app.nimble_mapping[value]
    return app.nimble_hyperlinks[value]

def addExampleLinks(app, string):
    exampleReplacements = []
    anchor = '<a href="{}">{}</a>'.format
    for example, href in app.nimble_examples.items():
        string = re.sub(example, anchor(href, example), string)

    return string

def addMarkdownLinks(app, string):
    markdownReplacements = []
    anchor = '<a class="nimble-hyperlink" href="{}">{}</a>'.format
    for match in re.finditer(r'<span class="pre">(.*?)</span>', string):
        variable = match.group(1)
        if variable in app.nimble_nolink:
            continue
        href = None
        if variable in app.nimble_hyperlinks:
            href = getHyperlink(app, variable)
        elif '.' + variable in app.nimble_hyperlinks:
            href = getHyperlink(app, '.' + variable)
        # nimble_markdown is last resort
        elif variable in app.nimble_markdown:
            href = app.nimble_markdown[variable]
        if href is not None:
            markdownReplacements.append((match.span(), anchor(href,
                                                      match.group())))

    return addStringReplacements(string, markdownReplacements)

def addCodeLinks(app, string):
    # each element of code block is wrapped in a span
    dotSpan = '<span class="o">\.</span>'
    nameSpan = '<span class="nn?">{}</span>'.format
    # need most complex links first so regex prioritizes them
    sortedLinks = sorted(app.nimble_hyperlinks.keys(), reverse=True)
    htmlLinks = []
    for link in sortedLinks:
        if link.startswith('.'):
            name = link[1:]
            # may need to look at calling object so also find that if available
            variable = '<span class="n">[_A-Za-z][_A-Za-z0-9]*</span>'
            # account for possible indexing  before method call
            optional = '(<span class="p">\[.*</span>)?'
            html = variable + optional + dotSpan
        else:
            htmlLinks.append(nameSpan(link))
            name = link
            html = ''
        spannedName = list(map(nameSpan, name.split('.')))
        html += dotSpan.join(spannedName)
        htmlLinks.append(html)
    linkPattern = '|'.join(htmlLinks)
    anchor = '<a class="nimble-hyperlink" href="{}">{}</a>'.format
    # we will replace the code (in the <pre> tags) with code that wraps
    # nimble calls with anchor tags to their api documentation
    codeReplacements = []
    for code in re.finditer(r'<pre>.*?</pre>', string, flags=re.DOTALL):
        lines = []
        splitCode = code.group().split('\n')
        for line in splitCode:
            linkedLines = []
            for match in re.finditer(linkPattern, line):
                htmlName = match.group()
                name = re.sub(r'<.*?>', '', htmlName)
                # Object type cannot be determined so all methods are
                # assumed to be nimble objects unless explicitly named in
                # app.nimble_nolink which is defined in setHyperlinks
                if name in app.nimble_nolink:
                    continue
                # object methods
                if name not in app.nimble_hyperlinks or name.startswith('.'):
                    # ignore everything before method call
                    name = '.' + name.split('.', 1)[1]
                    # keep features and points as part of hyperlink
                    if '.features.' in name or '.points.' in name:
                        rsplit = 4
                    else:
                        rsplit = 2
                    # first component is everything we don't want to link
                    components = htmlName.rsplit('</span>', rsplit)
                    htmlName = '</span>'.join(components[1:])
                    # first component is missing </span> so add 7
                    span = (match.start() + len(components[0]) + 7,
                            match.end())
                # nimble functions
                else:
                    span = match.span()
                href = getHyperlink(app, name)
                linkedLines.append((span, anchor(href, htmlName)))

            lines.append(addStringReplacements(line, linkedLines))
        codeReplacements.append((code.span(), '\n'.join(lines)))

    return addStringReplacements(string, codeReplacements)

def exampleHyperlinks(app, pagename, templatename, context, doctree):
    """
    Wrap the nimble calls in an anchor tag linking to API Documentation.
    """
    path = '../docs/generated/{}.html'.format
    # It is not possible to tell object types when applying hyperlinks to
    # the html code. So, we assume that any method names in the example
    # code are referring to the Nimble objects. If the example uses another
    # object type with a shared method name, it must be explicitly ignored.
    app.nimble_nolink = []
    # define default mapping for names that currently have a conflict
    app.nimble_mapping = {
        'train': path('nimble.train'),
        '.train': path('nimble.core.interfaces.TrainedLearner.train'),
        '.apply': path('nimble.core.interfaces.TrainedLearner.apply'),
        '.copy': path('nimble.core.data.Base.copy')}
    if pagename.startswith('examples/'):
        if 'additional_functionality' in pagename:
            app.nimble_nolink = ['tempDir.name', 'learnerType']
            app.nimble_mapping['train'] = path('nimble.CustomLearner.train')
            app.nimble_mapping['.train'] = path('nimble.CustomLearner.train')
            app.nimble_mapping['.apply'] = path('nimble.CustomLearner.apply')

        context['body'] = addExampleLinks(app, context['body'])
        context['body'] = addMarkdownLinks(app, context['body'])
        context['body'] = addCodeLinks(app, context['body'])
    elif pagename.startswith('docs/index'):
        app.nimble_nolink = ['nimble'] # no need to link to same page
        context['body'] = addMarkdownLinks(app, context['body'])
    elif pagename.startswith('cheatsheet'):
        # remove header that sphinx adds
        context['body'] = re.sub('<h1>cheatsheet', '', context['body'])
        context['body'] = re.sub('</h1>', '', context['body'])

def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    app.connect('autodoc-process-signature', process_signature)
    app.connect('builder-inited', setHyperlinks)
    app.connect('html-page-context', exampleHyperlinks)

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    #    'sphinx.ext.coverage',
    #    'sphinx.ext.ifconfig',
]

intersphinx_mapping = {'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
                       'random': ('https://docs.python.org/3/', None)}

autodoc_default_options = {
    'undoc-members': True,
}

autosummary_generate = True
autosummary_imported_members = True

# prevents autoclass from adding an autosummary table which leads to a warning
numpydoc_show_class_members = False

# napoleon settings
napoleon_google_docstring = False # only allow numpy style docstrings
napoleon_use_rtype = False # Returns section formatted to match numpy style
napoleon_custom_sections = ['Keywords'] # add a Keywords section for docstrings

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Nimble'
copyright = u'{}, Spark Wave'.format(datetime.now().year)
author = u'Spark Wave'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 'Alpha'
# The full version, including alpha/beta/rc tags.
release = 'Alpha'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    'docs/generated/nimble.rst',
    'docs/generated/nimble.calculate.rst',
    ]

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'fixed_sidebar': True,
    'page_width': '95%',
    'body_max_width': None,
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Nimbledoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #'preamble': '',

    # Latex figure (float) alignment
    #'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'nimble.tex', u'Nimble Documentation',
     u'Spark Wave', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'nimble', u'Nimble Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'nimble', u'Nimble Documentation',
     author, 'nimble', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

# -- Options for nbsphinx -------------------------------------------------

# suffix for source files, by default is .txt
html_sourcelink_suffix = ''

# recommended when matplotlib is used
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]

# removes input and output prompts
nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

# -- IPython Notebook conversion ------------------------------------------

class PYtoIPYNB:

    def __init__(self, file, metadata=None, nbformat=4, nbformat_minor=4):
        self.file = file
        self.name = os.path.splitext(file)[0]
        self.notebook = {}
        self.notebook['cells'] = []
        if metadata is None:
            self.notebook['metadata'] = {}
        else:
            self.notebook['metadata'] = metadata
        self.notebook['nbformat'] = nbformat
        self.notebook['nbformat_minor'] = nbformat_minor
        self.celltype = None
        self.multiline = False

    def convertToNotebook(self):
        with open(self.file, 'r') as f:
            setTerminalSize = [
                "import os\n",
                "import shutil\n",
                "size = os.terminal_size((132, 30))\n",
                "shutil.get_terminal_size = lambda *args, **kwargs: size\n"
            ]
            terminalSize = dict(source=setTerminalSize, cell_type='code',
                                execution_count=None,
                                metadata={"nbsphinx": "hidden"}, outputs=[])
            self.notebook['cells'].append(terminalSize)
            for line in f.readlines():
                if re.match(r'["\']{3}\n?', line):
                    self.multiline = not self.multiline
                    self.celltype = None
                    continue
                self._createCells(line)
            self._trimPreviousCellNewlines()

        ipynb = self.name + '.ipynb'
        with open(ipynb, 'w+') as notebookFile:
            json.dump(self.notebook, notebookFile, indent=1)

    def _createCells(self, line):
        pattern = r'(##\s)(.*?)(#*)(\n)'
        markdownLine = re.match(pattern, line)
        isMarkdown = (self.multiline or markdownLine)
        if markdownLine and not self.multiline:
            groups = markdownLine.groups()
            line = groups[1] + groups[3]
            if groups[2]:
                line = groups[2] + ' ' + line
        if line == '##\n':
            line = '\n'
        isNewline = line == '\n'
        # new markdown cell if previous was code or this line is a heading
        wasCode = self.celltype != 'markdown'
        newHeading = line.startswith('## ') or line.startswith('**Reference')
        if (not isNewline and isMarkdown and (wasCode or newHeading)):
            self.celltype = 'markdown'
            cellInfo = dict(cell_type=self.celltype, metadata={}, source=[])
            self._addNewCell(cellInfo)
        elif not (isNewline or isMarkdown) and self.celltype != 'code':
            self.celltype = 'code'
            cellInfo = dict(cell_type=self.celltype, execution_count=None,
                            metadata={}, outputs=[], source=[])
            self._addNewCell(cellInfo)

        self.notebook['cells'][-1]['source'].append(line)

    def _addNewCell(self, cellInfo):
        self._trimPreviousCellNewlines()
        self.notebook['cells'].append(cellInfo)

    def _trimPreviousCellNewlines(self):
        if self.notebook['cells'] and self.notebook['cells'][-1]['source']:
            if self.notebook['cells'][-1]['source'][-1] == '\n':
                _ = self.notebook['cells'][-1]['source'].pop()
            if self.notebook['cells'][-1]['source'][-1].endswith('\n'):
                trimmed = self.notebook['cells'][-1]['source'][-1][:-1]
                self.notebook['cells'][-1]['source'][-1] = trimmed

examplesDir = os.path.join(os.path.dirname(confFilePath), 'examples')
exampleFiles = [os.path.join(examplesDir, f) for f in os.listdir(examplesDir)
                if f.endswith('.py')]

for file in exampleFiles:
    PYtoIPYNB(file).convertToNotebook()
