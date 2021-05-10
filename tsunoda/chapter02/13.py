import pandas as pd

df1 = pd.read_csv('./result_py/col1.txt', delimiter='\t', header=None)
df2 = pd.read_csv('./result_py/col2.txt', delimiter='\t', header=None)
df = pd.concat([df1, df2],axis=1)
df.to_csv('./result_py/13.txt', sep='\t' ,header=False, index=False)
