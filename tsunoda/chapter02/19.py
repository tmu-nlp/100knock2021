import pandas as pd

df = pd.read_csv('./data/popular-names.txt',delimiter='\t',  header=None)
vc = df[0].value_counts()
vc = pd.DataFrame(vc)
vc = vc.reset_index()
vc.columns = ['name','count']
vc = vc.sort_values(['count','name'],ascending=[False,False])
print (vc)
