from knock20 import load_england

for line in load_england().split('\n'):
    if '[[Category:' in line:
        print(line)
