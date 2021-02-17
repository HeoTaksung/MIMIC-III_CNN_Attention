import cnn_attention
from tensorflow.keras.callbacks import EarlyStopping


class Label_Classification(object):
    def __init__(self, model_name='EnCAML', max_len=200, class_num=2, vocab_size=10000, embedding_size=100,
                 filter_num=300, kernel_size=4, padding='same', activation=None, dropout=0.2, batch_size=16,
                 learning_rates=0.0001, beta_1=0.9, beta_2=0.999, epochs=100,
                 X_train=None, y_train=None, X_test=None, y_test=None, pretrain_vec=None, exist_emb=True):

        self.model_name = model_name
        self.max_len = max_len
        self.class_num = class_num
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rates = learning_rates
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epochs = epochs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pretrain_vec = pretrain_vec
        self.exist_emb = exist_emb
    
    def train(self):
        if self.model_name == 'EnCAML':
            model = cnn_attention.Multi_CAML(self, summary=True)
        elif self.model_name == 'SWAM_CAML' or self.model_name == 'CAML':
            model = cnn_attention.Single_CAML(self, summary=True)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4, restore_best_weights=True)
        
        model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                  validation_split=0.2, callbacks=[es])
        
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        
        print("TEST Loss : {:.6f}".format(test_loss))
        print("TEST ACC : {:.6f}".format(test_acc))
