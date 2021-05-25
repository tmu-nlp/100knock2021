import sys, os
a = sys.argv[1]
os.system(f'/usr/local/bin/gsplit -n {a} ./dat/popular-names.txt ./out_py/popular-names-')
