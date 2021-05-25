import pandas as pd
df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
print (len(df))