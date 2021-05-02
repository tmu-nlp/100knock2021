N = input('Enter number: ')
with open('c:\Git\popular-names.txt') as f:
    print("".join(f.readlines()[:int(N)]))