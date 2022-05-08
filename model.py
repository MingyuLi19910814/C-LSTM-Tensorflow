import numpy as np
import tensorflow as tf
from gensim import downloader
def get_clstm(seq_len,
              word2idx,
              embedding_dim,
              keep_prob,
              filter_sizes,
              filter_num,
              lstm_hidden_size,
              lstm_num_layers,
              num_class,
              l2_reg_lambda,
              refine,
              google_vocabulary):
    vocab_dim = word2idx.__len__() + 1
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab_dim, embedding_dim))
    cnt_word_not_in_vocab = 0
    for word, i in word2idx.items():
        if word in google_vocabulary:
            embedding_matrix[i] = google_vocabulary[word]
        else:
            cnt_word_not_in_vocab += 1
    print('word num = {}, cnt_word_not_in_vocab = {}'.format(word2idx.__len__(), cnt_word_not_in_vocab))

    # (-1, max_len)
    inputs = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32)
    # (-1, max_len, embedding_dim)
    x = tf.keras.layers.Embedding(input_dim=vocab_dim,
                                  output_dim=embedding_dim,
                                  weights=[embedding_matrix],
                                  input_length=seq_len,
                                  name='embedding_layer', trainable=refine)(inputs)
    # (-1, max_len, embedding_dim, 1)
    x = tf.expand_dims(x, axis=-1)
    # (-1, max_len, embedding_dim, 1)
    conv_input = tf.keras.layers.Dropout(keep_prob)(x)
    conv_outputs = []
    max_feature_len = seq_len - max(filter_sizes) + 1
    for filter_size in filter_sizes:
        # (-1, max_len - filter_size + 1, 1, filter_num)
        x = tf.keras.layers.Conv2D(filter_num, kernel_size=(filter_size, embedding_dim), activation='relu')(conv_input)
        # (-1, max_len - filter_size + 1, filter_num)
        x = x[:, :max_feature_len, 0, :]
        conv_outputs.append(x)

    if conv_outputs.__len__() > 1:
        # (-1, max_feature_len, filter_num * n)
        x = tf.concat(conv_outputs, axis=-1)
    else:
        x = conv_outputs[0]
    lstm_cells = [tf.keras.layers.LSTMCell(lstm_hidden_size, dropout=(1 - keep_prob)) for _ in range(lstm_num_layers)]
    # final_state: (-1, lstm_hidden_size)
    final_state = tf.keras.layers.RNN(lstm_cells, return_state=False)(x)
    # (-1, num_class)
    outputs = tf.keras.layers.Dense(num_class,
                                    activation='softmax',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                                    bias_regularizer=tf.keras.regularizers.l2(l2_reg_lambda))(final_state)
    # outputs = tf.keras.layers.Dense(1,
    #                                 activation='sigmoid',
    #                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
    #                                 bias_regularizer=tf.keras.regularizers.l2(l2_reg_lambda))(final_state)
    model = tf.keras.Model(inputs, outputs)
    return model