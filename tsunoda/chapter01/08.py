def cipher(str):
  rep = [chr(219 - ord(x)) if x.islower() else x for x in str]

  return ''.join(rep)

message = 'the quick brown fox jumps over the lazy dog'
message = cipher(message)
print('暗号化:', message)
message = cipher(message)
print('復号化:', message)