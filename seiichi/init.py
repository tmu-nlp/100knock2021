import os, sys
ag = sys.argv[1]
tar = 'chapter{}'.format(format(ag, '0>2'))
if not os.path.exists(tar): os.mkdir(tar)
ex = 'sh' if ag == '2' else 'py'
for i in range(10): os.system(f'touch {tar}/{i+(int(ag)-1)*10}.{ex}')
