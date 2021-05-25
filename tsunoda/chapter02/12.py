import pandas as pd

df = pd.read_csv('./data/popular-names.txt', delimiter='\t', header=None)
df.iloc[:,0].to_csv('./result_py/col1.txt', sep=' ',header=False, index=False)
df.iloc[:,1].to_csv('./result_py/col2.txt', sep=' ',header=False, index=False)
