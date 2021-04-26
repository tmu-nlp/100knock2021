##Split the sentence “Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can”. into words, and extract the first letter from the 1st, 5th, 6th, 7th, 8th, 9th, 15th, 16th, 19th words and the first two letters from the other words. Create an associative array (dictionary object or mapping object) that maps from the extracted string to the position (offset in the sentence) of the corresponding word.

s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. ' \
    'New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
indeces = [1, 5, 6, 7, 8, 9, 15, 16, 19]
indeces = [index - 1 for index in indeces]
words = s.split(' ')

res = {}
for (i, word) in enumerate(words):
    if i in indeces:
        res[word[:1]] = i + 1
        indeces.pop(0)
    else:
        res[word[:2]] = i + 1
print(res)
