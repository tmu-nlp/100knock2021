##Obtain the string “schooled” by concatenating the letters in “shoe” and “cold” one after the other from head to tail.
word1 = 'shoe'
word2 = 'cold'
wordnew = ''.join([x+y for x,y in zip(word1,word2)])
print(wordnew)


#schooled
#zip
#for example
#a = [1,2,3]
#b = [4,5,6]
#c = [4,5,6,7,8]
#zipped = zip(a,b)
#[(1, 4), (2, 5), (3, 6)]#
#zipped = zip(a,c)
#[(1, 4), (2, 5), (3, 6)]#
#zip(*zipped) 
#[(1, 2, 3), (4, 5, 6)]#
