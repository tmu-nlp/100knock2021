##Implement a function that receives arguments, x, y, and z and returns a string “{y} is {z} at {x}”, where “{x}”, “{y}”, and “{z}” denote the values of x, y, and z, respectively. In addition, confirm the return string by giving the arguments x=12, y="temperature", and z=22.4.
def assemble(x,y,z):
    sentence = f'{y} is {z} at {x}'
    return sentence
print(assemble(12,'temperature',22.4))

#temperature is 22.4 at 12
