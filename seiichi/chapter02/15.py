import sys
cnt = int(sys.argv[1])
tar = open('./dat/popular-names.txt', 'r').readlines()
with open('./out_py/15.txt', 'w') as f:
    for i in range(len(tar) - min(cnt, len(tar)), len(tar)):
        f.write(tar[i])
    
