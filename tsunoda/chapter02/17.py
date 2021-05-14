import pandas as pd

df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
ans = df[0].unique()
ans.sort()
print(ans)