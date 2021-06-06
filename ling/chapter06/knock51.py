#contrast [title] [category]
def contrast_feature(file_path,feature_file_name):
    with open(file_path,'r',encoding='utf-8') as f,open(feature_file_name,'w',encoding='utf-8') as a:
        for line in f:
            l=line.strip().split('\t')
            a.write(l[1]+'\t'+l[4]+'\n')


if __name__=='__main__':
    file_path="./train.txt"
    feature_file_name="train.feature.txt"
    contrast_feature(file_path,feature_file_name)
    file_path="./valid.txt"
    feature_file_name="valid.feature.txt"
    contrast_feature(file_path,feature_file_name)
    file_path="./test.txt"
    feature_file_name="test.feature.txt"
    contrast_feature(file_path,feature_file_name)