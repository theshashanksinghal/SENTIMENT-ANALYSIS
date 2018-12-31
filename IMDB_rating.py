									"""imdb rating"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
words=5000
max_len=500
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=words)
x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)
model=Sequential()
model.add(Embedding(words,32,input_length=500))
model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
file='imdb_rating.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint=ModelCheckpoint(file,monitor='loss',verbose=0,save_best_only=True,mode='min')
model.fit(x_train,y_train,epochs=3,batch_size=64)
score=model.evaluate(x_train,y_train,verbose=1)
print("accuracy:%f",%(score[1]*100))
