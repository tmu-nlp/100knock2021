import pandas as pd
import csv
import re

###INPUT_FILE_PATH###
path1 = './datas/2pageSessions.csv'
path2 = './datas/newsCorpora.csv'

###DATA###
#path1
#FORMAT: STORY \t HOSTNAME \t CATEGORY \t URL
#path2
#FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
#CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)

###OUTPUT_FILE_PATH###
train_path = './train.txt'
valid_path = './valid.txt'
test_path = './test.txt'

with open(path1) as p1 , open(path2) as p2:
	df1 = pd.DataFrame(csv.reader(p1, delimiter='\t'))
	df2 = pd.DataFrame(csv.reader(p2, delimiter='\t'))

Huff_df = df2[df2[3].str.contains('Huffington Post')]
Busi_df = df2[df2[3].str.contains('Businessweek')]
Cont_df = df2[df2[3].str.contains('Contactmusic.com')]
Dail_df = df2[df2[3].str.contains('Daily Mail')]

df_all_shuff = pd.concat([Huff_df, Busi_df, Cont_df, Dail_df]).sample(frac=1)

###[10700 rows x 8 columns]###
train_data = df_all_shuff[:8560][[4,1]]
valid_data = df_all_shuff[8560:8560+535][[4,1]]
test_data = df_all_shuff[-535:][[4,1]]

###OUT_PUT###
train_data.to_csv(train_path, index = False, header=False, sep='\t')
valid_data.to_csv(valid_path, index=False, header=False, sep='\t')
test_data.to_csv(test_path, index=False, header=False, sep='\t')

print('train_data shape = ',train_data.shape)
print('valid_data shape=',valid_data.shape)
print('test_data shape=',test_data.shape)

