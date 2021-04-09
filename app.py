## Introduction to basic sentiment analysis 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
## IMDB reviews dataset
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)
print(x_train[0])
print(y_train[0])
class_names = ['Negative', 'Positive']
word_index = imdb.get_word_index()
print(word_index['hello'])

## Decoding the Reviews
reverse_word_index = dict((value, key) for key, value in word_index.items())
def decode(review):
      text = ''
      for i in review:
         text += reverse_word_index[i]\n"
         text += ' '
      return text"
decode(x_train[0])
def show_lengths():
        print('Length of 1st training example: ', len(x_train[0]))
        print('Length of 2nd training example: ',  len(x_train[1]))
        print('Length of 1st test example: ', len(x_test[0]))
        print('Length of 2nd test example: ',  len(x_test[1]))
        
    show_lengths()
## Padding the Examples
word_index['the']
from tensorflow.keras.preprocessing.sequence import pad_sequences

  x_train = pad_sequences(x_train, value = word_index['the'], padding = 'post', maxlen = 256)
  x_test = pad_sequences(x_test, value = word_index['the'], padding = 'post', maxlen = 256)
show_lengths()
decode(x_train[0])

## Word Embeddings : define feature vectors

## Creating and Training the Model
from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
    
    model = Sequential(
        Embedding(10000, 16),
        GlobalAveragePooling1D(),
        Dense(16, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
  

    "model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['acc']
    model.summary()
      
from tensorflow.keras.callbacks import LambdaCallback
simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))
  h = model.fit(
        x_train, y_train,
        validation_split = 0.2,
        epochs = E,\n",
        callbacks = [simple_logging],
        verbose = False
 ##  Predictions and Evaluation
    
    %matplotlib inline
    
 import matplotlib.pyplot as plt
    
 plt.plot(range(E), h.history['acc'], label = 'Training')
 plt.plot(range(E), h.history['val_acc'], label = 'Validation') 
 plt.legend()
 plt.show()
    
 loss, acc = model.evaluate(x_test, y_test)
 print('Test set accuracy: ', acc * 100)   
 
 prediction = model.predict(np.expand_dims(x_test[0], axis = 0))
 class_names = ['Negative', 'Positive']
 print(class_names[int(np.squeeze(prediction[0]) > 0.5)])
 
 print(decode(x_test[0]))
      
  

    
