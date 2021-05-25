N = 3
import pandas as pd
#read file
df = pd.read_csv('popular-names.txt', delimiter='\t', header=None)
#各ファイルの行数
step = - (-len(df) // N)

for n in range(N):
  #stepごとに行を取り出す
  df_split = df.iloc[n*step:(n+1)*step]
  df_split.to_csv('popular-names'+str(n)+'.txt', sep='\t',header=False, index=False)