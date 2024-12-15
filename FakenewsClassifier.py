import pandas as pd
df = pd.read_csv('Projects/data/fake-news-data/train.csv')
df = df.dropna()
x = df.drop('label', axis = 1)
y = df['label']

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot


voc_size = 5000

message = x.copy()
message.reset_index(inplace = True)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

def clearing(message):
    review = re.sub('[^a-zA-z]', ' ', message)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

corpus = [clearing(message['title'][i]) for i in range(len(message))]

onehot_repr = [one_hot(words, voc_size) for words in corpus]
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = 20)

# Creating models:
embedding_vector_features = 40
max_len = 20
model = Sequential()
model.add(Embedding(input_dim = voc_size, output_dim = embedding_vector_features, input_length = max_len))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))

#import numpy as np
#x_dummy = np.random.randint(0, voc_size, size=(1, max_len))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#model.predict(x_dummy)

import numpy as np
x_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size= 0.3, random_state=42)
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 64)

y_pred = model.predict(x_test)

y_pred_binary = (y_pred > 0.5).astype(int)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_binary))