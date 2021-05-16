import pandas as pd
import os
import glob

n = int(input('分割する数'))
df = pd.read_csv('./data/popular-names.txt',delimiter='\t',header=None)
count = -(-len(df)//n)

files = glob.glob("popular-names-*txt")
if len(files) > 0:
   for f in files:
       os.remove(f)
   print('remove splited files')

for i in range(n):
   df_split = df.iloc[n*count:(n+1)*count]
   df_split.to_csv('popular-names-{}.txt'.format(i+1), sep='\t', header=False, index=False)
