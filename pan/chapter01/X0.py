##Obtain the string that arranges letters of the string “stressed” in reverse order (tail to head).
#method1
x = 'stressed'
x = x[::-1]
print(x)

#method2
from functools import reduce
word = 'stressed'
print(reduce(lambda x,y: y+x, word))



#desserts
