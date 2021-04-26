import random

def shuffle(words):
  result = []
  for word in words.split():
    if len(word) > 4:  # 長さが4超であればシャッフル
      word = word[:1] + ''.join(random.sample(word[1:-1], len(word) - 2)) + word[-1:]
    result.append(word)

  return ' '.join(result)

words = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
ans = shuffle(words)

print(ans)