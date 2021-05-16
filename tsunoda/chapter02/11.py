import pandas as pd

df = pd.read_csv('./data/popular-names.txt', delimiter='\t', header=None)
df.to_csv('./result_py/ans11.txt', sep=' ', index=False, header=None)