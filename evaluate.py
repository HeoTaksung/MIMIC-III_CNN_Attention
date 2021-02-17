import util
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
from text_cnn_att import Label_Classification as LC
from tensorflow.keras.preprocessing.sequence import pad_sequences

file = open('./train.csv', 'r', encoding='utf-8-sig')

rdr = csv.reader(file)

X, y = util.load_data(rdr)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(y_train, dtype='int64')
y_test = np.array(y_test, dtype='int64')

max_len = max([len(i) for i in X_train])

vocab_size = len(word_index)+1

class_num = to_categorical(y_train)[0].shape[0]

X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

embed_model_dir = 'pre-trained embedding model directory'  # Leave blank if there is no embedding model. (etc: '')

embedding_size = 100

pretrain_vec, exist_emb = util.embedding_load(embed_model_dir, word_index, embedding_size)

att = input('Model Name : [EnCAML, SWAM_CAML, CAML]')

if att == 'EnCAML':
    filter_num = 300
    kernel_size = [3, 5, 7, 9]
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    epochs = 100
    EnCAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size,
                embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
                dropout=dropout, learning_rates=learning_rates, beta_1=beta_1, beta_2=beta_2, epochs=epochs,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pretrain_vec=pretrain_vec,
                exist_emb=exist_emb)
    EnCAML.train()

elif att == 'SWAM_CAML':
    filter_num = 500
    kernel_size = 4
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.001
    SWAM_CAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size,
                   embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
                   dropout=dropout, learning_rates=learning_rates,
                   X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pretrain_vec=pretrain_vec,
                   exist_emb=exist_emb)
    SWAM_CAML.train()

elif att == "CAML":
    filter_num = 50
    kernel_size = 3
    padding = 'same'
    dropout = 0.2
    learning_rates = 0.0001
    CAML = LC(model_name=att, max_len=max_len, class_num=class_num, vocab_size=vocab_size,
              embedding_size=embedding_size, filter_num=filter_num, kernel_size=kernel_size,
              dropout=dropout, learning_rates=learning_rates,
              X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pretrain_vec=pretrain_vec,
              exist_emb=exist_emb)
    CAML.train()

else:
    print("model does not exist.")
