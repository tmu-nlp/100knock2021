''"""
24. ファイル参照の抽出
記事から参照されているメディアファイルをすべて抜き出せ．

[ref]
‐　ウィキペディアの画像
　　‐　書式：[[ファイル:ファイル名|オプション]]/[[File:ファイル名|オプション]]
　　‐　例：[[ファイル:UK topo en.jpg|thumb|200px|イギリスの地形図]]
‐　ウィキペディでアップロード可能なファイル形式
　　‐　https://ja.wikipedia.org/wiki/Help:画像などのファイルのアップロードと利用
　　　　‐　ファイルをアップロードし、ページ中に挿入できる
　　　　‐　許可される形式は、拡張子がpng,gif,jpg,jpeg,xcf,pdf,mid,ogg,svg,djvuのいずれか
　　　　　　基礎情報　国
　　　　　　‐｜国旗画像 = Flag of the United Kingdom.svg
          <gallery>
          - stonehenge2007 07 30.jpg|[[ストーンヘンジ]]
          <ref>
          - <ref>[http://warp.da.ndl.go.jp/.../country.pdf
"""

import re
from knock20 import read_gzip

if __name__ == '__main__':
    filepath = './data/jawiki-country.json.gz'
    text = read_gzip(filepath, 'イギリス')

    pattern = r'ファイル:(.+?\..*?)(?:\||\])'
    mediafile = re.findall(pattern,text)    # list
    #for i in range(1, len(mediafile)+1):
        #print(i,'\t',mediafile[i-1])

    print('The number of extracted mediafiles is:', len(mediafile), ',and listed as follow:\n',mediafile)
