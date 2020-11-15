# -*- Encoding: UTF-8 -*- #
import fasttext.util
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import datetime

class SpamFilter():
    def __init__(self, fname, mode):
        print("\nPROCESING MESSAGE DATA...")
        #configuration
        self.model_directory = 'models/'

        # load data
        self.data = pd.read_csv(fname, encoding='cp949')
        # 1. label processing
        if mode == 0 or mode==1:
            self.labels = self.data['label'].to_numpy(dtype='int32')
            self.y = np.concatenate((self.labels, self.labels), axis = 0)
        else: pass
        # 2. message processing
        self.messages = self.data['message'].str.encode('utf8').str.decode('utf8')
        self.messages = self.messages.tolist()

        # load vector model
        self.ft = fasttext.load_model('wordvec_model/cc.en.300.bin')
        # self.ft = fasttext.load_model('wordvec_model/yelp_review_polarity.bin')
        self.vectordim = 300

        # clean and vectorize string
        self.messages = list(map(self.cleanTxt, self.messages))
        # changed sentences into vector, 한글이 없어진 후 np.array 적용
        for i in range(len(self.messages)):
            # message 내 token 기준으로 vectorize
            self.messages[i] = [self.ft.get_word_vector(token) for token in self.messages[i]]
        # append padding
        if mode == 0 or mode == 1:
            self.messages = pad_sequences(self.messages, dtype = 'float32', maxlen=191, padding = 'pre')
            self.messages_aug = pad_sequences(self.messages, dtype = 'float32', maxlen=191, padding = 'post')
            self.messages = np.concatenate((self.messages, self.messages_aug), axis = 0)
        else:
            self.messages = pad_sequences(self.messages, dtype='float32', maxlen=191, padding='pre')
        self.tokennum = len(self.messages[0])

        self.X = np.array(self.messages)
        self.X= np.reshape(self.X, (len(self.messages), self.tokennum, self.vectordim, 1))

        print("Data processing finished")

    def train(self, X_train, y_train, X_val, y_val):
        input = Input(shape=(self.tokennum, self.vectordim, 1), dtype='float32', name='main_input')

        gram2 = Conv2D(1, kernel_size=(2, self.vectordim), strides=(1, self.vectordim),
                       input_shape=(self.tokennum, self.vectordim), activation='relu', padding='SAME')(input)
        gram2 = MaxPooling2D(pool_size=(gram2.shape[1].value, 1), strides=None, padding="valid")(gram2)
        gram3 = Conv2D(1, kernel_size=(3, self.vectordim), strides=(1, self.vectordim),
                       input_shape=(self.tokennum, self.vectordim), activation='relu', padding='SAME')(input)
        gram3 = MaxPooling2D(pool_size=(gram3.shape[1].value, 1), strides=None, padding="valid")(gram3)
        gram4 = Conv2D(1, kernel_size=(4, self.vectordim), strides=(1, self.vectordim),
                       input_shape=(self.tokennum, self.vectordim), activation='relu', padding='SAME')(input)
        gram4 = MaxPooling2D(pool_size=(gram4.shape[1].value, 1), strides=None, padding="valid")(gram4)
        gram5 = Conv2D(1, kernel_size=(5, self.vectordim), strides=(1, self.vectordim),
                       input_shape=(self.tokennum, self.vectordim), activation='relu', padding='SAME')(input)
        gram5 = MaxPooling2D(pool_size=(gram5.shape[1].value, 1), strides=None, padding="valid")(gram5)
        gram6 = Conv2D(1, kernel_size=(5, self.vectordim), strides=(1, self.vectordim),
                       input_shape=(self.tokennum, self.vectordim), activation='relu', padding='SAME')(input)
        gram6 = MaxPooling2D(pool_size=(gram5.shape[1].value, 1), strides=None, padding="valid")(gram6)

        feature = Concatenate(axis=1)([gram2, gram3, gram4, gram5, gram6])
        feature = Flatten()(feature)
        FC = Dense(5, activation = 'relu')(feature)
        output = Dense(1, activation='sigmoid')(FC)

        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer='Adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        if not os.path.exists(self.model_directory): os.makedirs(self.model_directory)
        timestamp = datetime.datetime.now().strftime('%m%d%H%M')
        model_path = self.model_directory + timestamp + "_" + '{epoch:03d}_{val_loss:.4f}.hdf5'

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=30)

        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=200, batch_size=128, verbose=1,
                            callbacks=[cb_checkpoint, cb_early_stopping])
        print('\nAccuracy: {:.4f}'.format(model.evaluate(X_val, y_val)[1]))

        # get train / val graph
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(history.history['loss'], 'y', label='train loss')
        loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
        acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()
    def test(self, X_test, y_test, model_name):
        model = keras.models.load_model(self.model_directory + model_name)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print("loss: ", loss)
        print("accuracy : ", accuracy)
    def infer(self, data_X, model_name):
        self.result_directory = 'prediction/'
        if not os.path.exists(self.result_directory): os.makedirs(self.result_directory)
        model = keras.models.load_model(self.model_directory + model_name)
        predictions = model.predict(data_X)
        predictions = [1 if p>0.5 else 0 for p in predictions]
        df = pd.DataFrame(predictions)
        df.to_csv(self.result_directory+'prediction_'+ model_name.rstrip(".hdf5") + ".csv", index = True)
    def cleanTxt(self, string):
        # get rid of non-alphabet etc. and lowercase.
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.lower()
        string = string.split(" ")
        return string


if __name__ == "__main__":
    mode = int(input("SELECT MODE No.(0 : TRAIN, 1: TEST, 2: INFER) : "))
    if mode==1 or mode==2: model_name = input("TEST MODEL NAME(.hdf5): ")
    if mode==0 or mode==1:
        filter = SpamFilter('train.csv', mode)
        X_train, X_test, y_train, y_test = train_test_split(filter.X, filter.y, test_size=0.1, random_state=10, stratify = filter.y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=10, stratify = y_train)
        if mode==0: filter.train(X_train, y_train, X_val, y_val)
        if mode==1: filter.test(X_test, y_test, model_name)
    if mode==2:
        filter = SpamFilter('leaderboard_test_file.csv', mode)
        filter.infer(filter.X, model_name)
    else:
        pass

# References
# @inproceedings{mikolov2018advances,
#   title={Advances in Pre-Training Distributed Word Representations},
#   author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
#   booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
#   year={2018}
# }

