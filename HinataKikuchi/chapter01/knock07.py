def when_is_that(x,y,z):
	return str(x)+'時の'+str(y)+'は'+str(z)

print(when_is_that(12,"気温",22.4))

###ANS###
def ans(x,y,z):
	return '{}時の{}は{}'.format(x,y,z)

print(ans(12,"気温", 22.4))