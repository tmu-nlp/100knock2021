import random
print(' '.join([s[0]+''.join(random.sample(s[1:-1], len(s[1:-1])))+s[-1] if len(s) > 4 else s for s in input().split()]))
