from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import initializers


class CNN_Attention(tf.keras.layers.Layer):

    def __init__(self, class_num):
        super(CNN_Attention, self).__init__(class_num)
        self.class_num = class_num
        self.filter_num = None
        self.Wa = None

    def build(self, input_shape):
        self.filter_num = input_shape[2]

        # self.Wa = (the number of classes, the number of filters)
        self.Wa = self.add_weight(shape=(self.class_num, self.filter_num),
                                  initializer=initializers.get('glorot_uniform'), trainable=True)

        super(CNN_Attention, self).build(input_shape)

    def call(self, inputs):
        
        # inputs_trans = (batch_size, the number of filters, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])
        
        # at = (batch_size, the number of classes, sentence_length)
        at = tf.matmul(self.Wa, inputs_trans)
        
        # Softmax
        at = K.exp(at - K.max(at, axis=-1, keepdims=True))
        at = at / K.sum(at, axis=-1, keepdims=True)
        
        # weighted sum
        # v = (batch_size, the number of classes, the number of filters)
        v = K.batch_dot(at, inputs)

        return v

    
def Multi_CAML(self, summary=True):
    
    inputs = Input((self.max_len,))

    if self.exist_emb:
        embedding = Embedding(self.vocab_size, self.embedding_size, weights=[self.pretrain_vec], trainable=False)(inputs)
    else:
        embedding = Embedding(self.vocab_size, self.embedding_size)(inputs)

    cnn1 = Conv1D(self.filter_num, self.kernel_size[0], padding=self.padding, activation=self.activation)(embedding)
    att1 = CNN_Attention(self.class_num)(cnn1)
    att1 = Dropout(self.dropout)(att1)

    cnn2 = Conv1D(self.filter_num, self.kernel_size[1], padding=self.padding, activation=self.activation)(embedding)
    att2 = CNN_Attention(self.class_num)(cnn2)
    att2 = Dropout(self.dropout)(att2)
    
    cnn3 = Conv1D(self.filter_num, self.kernel_size[2], padding=self.padding, activation=self.activation)(embedding)
    att3 = CNN_Attention(self.class_num)(cnn3)
    att3 = Dropout(self.dropout)(att3)
    
    cnn4 = Conv1D(self.filter_num, self.kernel_size[3], padding=self.padding, activation=self.activation)(embedding)
    att4 = CNN_Attention(class_num=self.class_num)(cnn4)
    att4 = Dropout(self.dropout)(att4)

    concat = Concatenate(axis=-1)([att1, att2, att3, att4])

    label_li = []

    for i in range(self.class_num):
        label_li.append(Dense(1, activation='sigmoid')(concat[::, i]))

    labels_li = tf.stack(label_li, axis=1)

    labels_li = tf.squeeze(labels_li, [2])

    adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rates, beta_1=self.beta_1, beta_2=self.beta_2)

    model = Model(inputs=inputs, outputs=labels_li)

    if summary:
        model.summary()
        
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def Single_CAML(self, summary=True):
    
    inputs = Input((self.max_len,))
    
    if self.exist_emb:
        embedding = Embedding(self.vocab_size, self.embedding_size, weights=[self.pretrain_vec], trainable=False)(inputs)
    else:
        embedding = Embedding(self.vocab_size, self.embedding_size)(inputs)
        
    cnn1 = Conv1D(self.filter_num, self.kernel_size, padding=self.padding, activation=self.activation)(embedding)
    att1 = CNN_Attention(self.class_num)(cnn1)
    att1 = Dropout(self.dropout)(att1)

    label_li = []

    for i in range(self.class_num):
        label_li.append(Dense(1, activation='sigmoid')(att1[::, i]))

    labels_li = tf.stack(label_li, axis=1)

    labels_li = tf.squeeze(labels_li, [2])

    adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rates, beta_1=self.beta_1, beta_2=self.beta_2)

    model = Model(inputs=inputs, outputs=labels_li)

    if summary:
        model.summary()
        
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model