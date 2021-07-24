chapter57

features = X_train.columns.values
index = [i for i in range(1, 11)]
for c, coef in zip(lg.classes_, lg.coef_):
  print(f'【カテゴリ】{c}')
  best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
  worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
  display(pd.concat([best10, worst10], axis=0))
  print('\n')

出力

【カテゴリ】b
          1      2      3    4       5     6          7       8       9   \
重要度上位   bank    fed  china  ecb  stocks  euro  obamacare     oil  yellen   
重要度下位  video  ebola    the  her     and   she      apple  google    star   

              10  
重要度上位     dollar  
重要度下位  microsoft  


【カテゴリ】e
               1       2       3      4      5     6     7         8   \
重要度上位  kardashian   chris     her  movie   star  film  paul        he   
重要度下位          us  update  google  study  china    gm   ceo  facebook   

            9     10  
重要度上位  wedding   she  
重要度下位    apple  says  


【カテゴリ】m
             1      2       3      4     5     6       7      8        9   \
重要度上位     ebola  study  cancer   drug  mers   fda   cases    cdc    could   
重要度下位  facebook     gm     ceo  apple  bank  deal  google  sales  climate   

               10  
重要度上位  cigarettes  
重要度下位     twitter  


【カテゴリ】t
           1         2      3          4        5         6       7        8   \
重要度上位  google  facebook  apple  microsoft  climate        gm    nasa    tesla   
重要度下位  stocks       fed    her    percent     drug  american  cancer  ukraine   

            9           10  
重要度上位  comcast  heartbleed  
重要度下位    still      shares  
