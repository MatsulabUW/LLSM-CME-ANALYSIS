#!/usr/bin/env python

# Remove generated output from Jupyter files to get clear and meaningful diffs
# This runs automatically when doing a "git commit" to ensure code quality

import sys
from nbformat import read, write, NO_CONVERT

def strip_output(nb):
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    return nb

if __name__ == "__main__":
    nb = read(sys.stdin, as_version=NO_CONVERT)
    nb = strip_output(nb)
    write(nb, sys.stdout)
    sys.stdout.write("\n")
