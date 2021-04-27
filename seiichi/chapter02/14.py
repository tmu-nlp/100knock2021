import sys
cnt = int(sys.argv[1])
tar = open('./dat/popular-names.txt', 'r').readlines()
with open('./out_py/14.txt', 'w') as f:
    for i in range(min(cnt, len(tar))):
        f.write(tar[i])
    
