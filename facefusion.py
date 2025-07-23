#!/usr/bin/env python3

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

from facefusion import core

# Print the reference face path argument if present
for i, arg in enumerate(sys.argv):
	if arg == '--reference-face-path' and i + 1 < len(sys.argv):
		print(f'REFERENCE FACE PATH ARGUMENT: {sys.argv[i+1]}')

if __name__ == '__main__':
	core.cli()
