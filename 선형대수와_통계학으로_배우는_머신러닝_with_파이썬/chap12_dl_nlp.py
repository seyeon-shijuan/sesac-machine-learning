from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Embedding


paper = ['많은 것을 보고 싶다면 많은 것을 받아들여라.']
tknz = Tokenizer()
tknz.fit_on_texts(paper)
print(tknz.word_index)

# 단어를 벡터로 변환
idx_paper = tknz.texts_to_sequences(paper)
print(idx_paper)
n = len(tknz.word_index)+1
print(n)
idx_onehot = to_categorical(idx_paper, num_classes=n)
print(idx_onehot)

model = Sequential()
model.add(Embedding(input_dim=n, output_dim=3))
model.compile(optimizer='rmsprop', loss='mse')
embedding = model.predict(idx_paper)
print(embedding)

# 2번째 토큰과 5번째 토큰이 같아서 단어 벡터는 동일하게 나옴

