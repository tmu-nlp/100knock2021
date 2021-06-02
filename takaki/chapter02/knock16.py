import subprocess
import sys

args = ['split', '-n', sys.argv[1], '-d', 'popular-names.txt', 'tmp/knock16.py.']
res = subprocess.check_call(args)
