I1 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
enum = enumerate(I1)
d = dict((i,j) for i,j in enum)
print(d)