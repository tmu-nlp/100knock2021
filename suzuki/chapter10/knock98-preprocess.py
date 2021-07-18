def preprocess(name):
    en_list = []
    ja_list = []
    file_name = 'jesc_data/' + name
    with open(file_name, 'r') as src, open('jesc_pre/' + name + '.en', 'w') as en_file, open('jesc_pre/' + name + '.ja', 'w') as ja_file:
        for line in src:
            line = line.strip()
            en, ja = line.split('\t')
            en_file.write(en + '\n')
            ja_file.write(ja + '\n')

preprocess('dev')
preprocess('train')
preprocess('test')