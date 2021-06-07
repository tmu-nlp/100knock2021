import re
from knock50_pd import train,test,valid
import pandas as pd

def preprocess(text):
    table=
    text=re.sub('[0-9]+','0',text)
    text.lower()
'''
pd.concat(obj,axis,ignore_index)
@para
obj: a sequence or mapping of Series or DataFrame objects
axis: {0/’index’, 1/’columns’}, default 0
ignore_index: If True, do not use the index values along the concatenation axis
'''



df=pd.concat([train,valid,test],axis=0,ignore_index=True)
print(df)