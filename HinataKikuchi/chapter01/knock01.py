string = 'パタトクカシーー'
print([i for idx,i in enumerate(string) if idx%2==1 ])

###ANS###
#文字列で返そう。
print(string[0::2])