result = []
with open('neko.txt.mecab') as file, open('morpheme.txt', 'w') as output_file:
    for line in file:
        if line != 'EOS'+'\n':
            line = line.split()
            if line[0] != '' and len(line) == 2:
                morpheme = line[1].split(',')
                result.append({'surface': line[0], 'base': morpheme[6], 'pos': morpheme[0], 'pos1': morpheme[1]})
        elif len(result) != 0:
            output_file.write(f'{result}\n')
            result = []