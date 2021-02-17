import re
import numpy as np


def preprocess_text(sentence):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def load_data(rdr):
    sentence = []
    label = []
    for idx, line in enumerate(rdr):
        if idx == 0:
            continue
        if len(line[1]) == 0:
            continue
        sentence.append(preprocess_text(line[1]))
        label.append([line[i] for i in range(2, len(line))])
    return sentence, label


def embedding_load(embed_model_dir, word_index, embedding_size):
    exist_emb = False

    if not dir:
        return None, exist_emb

    embedding_index = dict()

    with open(embed_model_dir, 'r', encoding='utf-8-sig') as f:
        for line in f:
            # Exception to first line in word2vec model.
            if len(line.split()) == 2:
                continue
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors
        f.close()
        print('--- Finished make embedding index ---')
    print('Found %s word vectors.' % len(embedding_index))

    pretrain_vec = np.zeros((len(word_index)+1, embedding_size))
    embed_cnt = 0

    for word, i in word_index.items():
        embedded_vector = embedding_index.get(word)
        if embedded_vector is not None:
            pretrain_vec[i] = embedded_vector
            embed_cnt += 1

    print('Created Embedded Matrix: %s word vectors.' % embed_cnt)

    return pretrain_vec, exist_emb
