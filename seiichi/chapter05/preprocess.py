import CaboCha

with open('data/neko.txt', encoding='utf-8') as novel:
    cabocha = CaboCha.Parser()
    
    for line in novel:
        CaboChaed = cabocha.parse(line)
        with open('data/neko.txt.cabocha', 'a', encoding='utf-8') as new_file:
            new_file.write(CaboChaed.toString(CaboCha.FORMAT_LATTICE))
