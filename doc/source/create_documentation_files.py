#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Documentation creation utils."""

import re
import sys

import pandoc

from queens.utils.path_utils import relative_path_from_queens


def relative_to_doc_source(relative_path):
    """Relative path from documentation source.

    Args:
        relative_path (str): path relative to `doc/source/`

    Returns:
        pathlib.Path: Path relative from documentation
    """
    return relative_path_from_queens("doc/source/" + relative_path)


def create_tutorial_from_readme():
    """Create tutorial from readme."""
    # Extract the example from the readme
    sys.path.insert(1, str(relative_path_from_queens("test_utils").resolve()))
    from get_queens_example_from_readme import (  # pylint: disable=import-outside-toplevel, import-error
        get_queens_example_from_readme,
    )

    example = get_queens_example_from_readme(".")
    tutorial = relative_to_doc_source("tutorials.rst")
    current_tutorial_text = """.. _tutorials:

Tutorials and examples
==================================

Work in progress!

But here a simple example from our `README.md <https://github.com/queens-py/queens/blob/main/README.md>`_:

.. code-block:: python

%s

Resulting the histogram

.. image:: images/monte_carlo_uq.png
  :width: 500
  :align: center
  :alt: Monte Carlo Histogram.""" % example.replace(
        "\n", "\n   "
    )
    # Create file.
    tutorial.write_text(current_tutorial_text)


def remove_markdown_emojis(md_text):
    """Remove emojis from markdown text.

    Args:
        md_text (str): Markdown text

    Returns:
        str: Cleaned text
    """
    # Define a regex pattern for matching markdown-style emojis
    emoji_pattern = re.compile(r":\w+:")

    # Remove markdown emojis from the text
    return emoji_pattern.sub("", md_text)


def prepend_relative_links(md_text, base_url):
    """Prepend url for relative links.

    Args:
        md_text (str): Text to check
        base_url (str): Base URL to add

    Returns:
        str: Prepended markdown text
    """
    md_link_regex = "\\[([^]]+)]\\(\\s*(.*)\\s*\\)"
    for match in re.findall(md_link_regex, md_text):
        _, link = match
        if link.strip().startswith("http"):
            continue
        md_text = md_text.replace(f"({link})", f"({base_url}/{link.strip()})")
    return md_text


def load_markdown(md_path):
    """Load markdown and escape relative links and emojis."""
    md_text = relative_path_from_queens(md_path).read_text()
    md_text = remove_markdown_emojis(md_text)
    md_text = prepend_relative_links(md_text, "https://www.github.com/queens-py/queens/blob/main")
    return md_text


def create_testing():
    """Create pages related to testing."""
    contributing_path = relative_path_from_queens("tests/README.md")
    md_text = load_markdown(md_path=contributing_path)
    doc = pandoc.read(md_text, format="markdown")

    rst_path = "testing.rst"
    pandoc.write(doc, file=relative_to_doc_source(rst_path), format="rst")
    return rst_path


def create_contributing():
    """Create pages related to contributing."""
    contributing_path = relative_path_from_queens("CONTRIBUTING.md")
    md_text = load_markdown(md_path=contributing_path)
    doc = pandoc.read(md_text)

    rst_path = "contributing.rst"
    pandoc.write(doc, file=relative_to_doc_source(rst_path), format="rst")
    return rst_path


def create_development():
    """Create pages related to development."""
    development_text = "Development"
    development_text += "\n==================\n\n"
    development_text += "\n\n"
    development_text += "\n.. toctree::\n"
    development_text += "   :maxdepth: 1\n"
    development_text += "   :caption: Thanks for your interest in developing QUEENS!\n"

    rst_path = create_contributing()
    development_text += "\n   " + rst_path

    rst_path = create_testing()
    development_text += "\n   " + rst_path
    relative_to_doc_source("development.rst").write_text(development_text)


def main():
    """Create all the rst files."""
    create_tutorial_from_readme()
    create_development()
