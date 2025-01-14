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
import shutil
import sys

from queens.utils.path_utils import relative_path_from_queens


def create_tutorial_from_readme():
    sys.path.insert(1, str(relative_path_from_queens("test_utils").resolve()))
    from get_queens_example_from_readme import get_queens_example_from_readme

    example = get_queens_example_from_readme(".")
    tutorial = relative_path_from_queens("doc/source/tutorials.rst")
    current_tutorial_text = """.. _tutorials:

Tutorials and examples
==================================

Work in progress, but here a simple example from our `README.md <https://github.com/queens-py/queens/blob/main/README.md>`_:

.. code-block:: python""" + example.replace(
        "\n", "\n   "
    )
    tutorial.write_text(current_tutorial_text)
