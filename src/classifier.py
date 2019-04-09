from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout
import pandas as pd
import os
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
import pickle
from keras.models import load_model

nlp = en_core_web_sm.load()

#function to extract adjective, verb from reviews(words that express emotion)
def filter_av(text):
    tok = nlp(text)
    av = ""
    av_lst = [item.lemma_+ " " for item in tok if (not item.is_stop and not item.is_punct and (item.pos_ == "VERB" or item.pos_ == "ADJ"))]
    return av.join(av_lst)

def preprocess(df):
    df.columns = ['polarity','category','aspect','offsets','review']
    df["review_av"] = df["review"].apply(filter_av)
    return df


class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train = pd.read_csv(trainfile,sep="\t",header=None)
        train = preprocess(train)
        # prepare embedding matrix
        # set dim, max_words
        embedding_dim = 100
        max_words = 10000
        maxlen = 100
        tokenizer = Tokenizer(num_words=max_words) 
        tokenizer.fit_on_texts(train.review_av)
        with open('tokenizer.pickle', 'wb') as handle:
        	pickle.dump(tokenizer, handle)
        
        sequences = tokenizer.texts_to_sequences(train.review_av)
        

        x_train = pad_sequences(sequences, maxlen=maxlen)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train["polarity"])
        y_train = to_categorical(y_train)
        with open('encoder.pickle', 'wb') as handle:
        	pickle.dump(encoder, handle)

        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) 
        model.add(LSTM(32,dropout_U = 0.2, dropout_W = 0.2))
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='sigmoid')) 
        model.summary()
        model.compile(optimizer='rmsprop',               
              loss='categorical_crossentropy',               
              metrics=['acc'])
        model.fit(x_train, y_train,                     
                epochs=8,                     
                batch_size=32)
        model.save('model.h5')         
        


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test = pd.read_csv(datafile,sep="\t",header=None)
        test = preprocess(test)

        with open('tokenizer.pickle', 'rb') as handle:
        	tokenizer = pickle.load(handle)
        with open('encoder.pickle', 'rb') as handle:
                encoder = pickle.load(handle)
        

        x_test = tokenizer.texts_to_sequences(test['review_av'])
        x_test = pad_sequences(x_test, maxlen=100)
        y_test = encoder.transform(test["polarity"])
        y_test = to_categorical(y_test)

        model = load_model('model.h5')
        pred = model.predict_classes(x_test)

        return list(encoder.inverse_transform(pred))
        

        
        

        


        





