import pandas as pd

df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
ans = df.sort_values(by=2, ascending=False)
print(ans)
