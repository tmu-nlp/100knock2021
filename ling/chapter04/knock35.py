from knock30 import sentences
from collections import defaultdict

ans=defaultdict(int)
'''
defaultdict(***)=>key不存在的时候的默认返回值，写个0不用初始化爽歪歪
list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
'''

for sentence in sentences:
    for mor in sentence:
        if mor['pos']!='記号':
            ans[mor['base']]+=1

ans=sorted(ans.items(),key=lambda x:x[1],reverse=True)

for k in ans[:10]:
    print(k)

