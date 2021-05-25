import pandas as pd
n = int(input('出力したい行数を入力'))
df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
print(df.head(n))

