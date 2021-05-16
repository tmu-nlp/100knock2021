import pandas as pd
n = int(input('末尾から出力したい行数を入力'))
df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
print(df.tail(n))
