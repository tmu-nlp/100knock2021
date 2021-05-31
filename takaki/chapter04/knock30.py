from pprint import pprint


def parse_mecab(lines):
    def parse_line(line: str):
        res = line.split('\t')
        if res[0] == '' or len(res) != 2:
            return None
        attr = res[1].split(',')
        return {
            'surface': res[0],
            'base'   : attr[6],
            'pos'    : attr[0],
            'pos1'   : attr[1]
        }
    return [parsed for line in lines if (parsed := parse_line(line.strip())) != None]


def parse2_mecab(lines):
    def parse_line(line: str):
        res = line.split('\t')
        if res[0] == 'EOS':
            return 'EOS'
        if res[0] == '' or len(res) != 2:
            return None
        attr = res[1].split(',')
        return {
            'surface': res[0],
            'base'   : attr[6],
            'pos'    : attr[0],
            'pos1'   : attr[1]
        }
    res, tmp = [], []
    for line in lines:
        parsed = parse_line(line.strip())
        if parsed == 'EOS':
            res.append(tmp)
            tmp = []
        elif parsed != None:
            tmp.append(parsed)
    return res


if __name__ == '__main__':
    with open('neko.txt.mecab') as f:
        pprint(parse_mecab(f.readlines()))
