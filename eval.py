from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from text_cnn_att import Label_Classification as LC


(X_train, y_train), (X_test, y_test) = "file path"

vocab_size = 10000
max_len = max([len(i) for i in X_train])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

class_num = (to_categorical(y_train)[0].shape)[0]

att = input('Model Name : (EnCAML, SWAM_CAML, CAML)')

if att == 'EnCAML':
    embedding_size = 100
    filter_num = 300
    kernel_size = [3,5,7,9]
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    epochs = 100
    EnCAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size, 
                embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
                dropout=dropout, learning_rates=learning_rates, beta_1=beta_1, beta_2=beta_2, epochs=epochs,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    EnCAML.train()

elif att == 'SWAM_CAML':
    embedding_size = 100
    filter_num = 500
    kernel_size = 4
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.001
    SWAM_CAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size, 
                embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
                dropout=dropout, learning_rates=learning_rates,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    SWAM_CAML.train()
    
elif att == "CAML":
    embedding_size = 100
    filter_num = 50
    kernel_size = 10
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.0001
    CAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size, 
                embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
                dropout=dropout, learning_rates=learning_rates,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    CAML.train()
    
else:
    print("model does not exist.")