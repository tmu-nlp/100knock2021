##Split the sentence “Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics”. into words, and create a list whose element presents the number of alphabetical letters in the corresponding word.
sentence = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
s_apart = sentence.split()
print(s_apart)
for word in s_apart:
    print(word + ':' +str(len(word)),end = '\n')
    
    
    
    
#['Now', 'I', 'need', 'a', 'drink,', 'alcoholic', 'of', 'course,', 'after', 'the', 'heavy', 'lectures', 'involving', 'quantum', 'mechanics.']
#Now:3
#I:1
#need:4
#a:1
#drink,:6
#alcoholic:9
#of:2
#course,:7
#after:5
#the:3
#heavy:5
#lectures:8
#involving:9
#quantum:7
#mechanics.:10
