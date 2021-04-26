#method1:just plus them
x = [0,1]
y = [2,3]
z = [4,5]
h = x + y + z
print(h)

#method2:take out all the elements and put them in a new list
x = [0,1]
y = [2,3]
z = [4,5]
new = []
for item in x:
  new.append(item)
for item in y:
  new.append(item)
for item in z:
  new.append(item)
print(new)

#method3:plus elements of the next list
x = [0,1]
y = [2,3]
z = [4,5]
x.extend(y)
x.extend(z)
print(x)


#【0，1，2，3，4，5】
#list together
