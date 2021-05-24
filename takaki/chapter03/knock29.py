from knock25 import load_england_info
file = load_england_info()['国旗画像'].replace(' ', '_')
print(f"https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/{file}/1200px-{file}.png")
