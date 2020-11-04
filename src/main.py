# ファイルの中身を確認してみる
file_name = './data/reuter_dataset.tsv'

with open(file_name) as text_file:
    text = text_file.readlines()
    # print("0：", text[0])  # URL情報
    # print("1：", text[1])  # タイムスタンプ
    # print("2：", text[2])  # タイトル
    # print("3：", text[3])  # 本文
    print(len(text))