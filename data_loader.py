import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf


def embed(texts, tokenizer, max_len):
    text_ids = tokenizer.texts_to_sequences(texts)
    text_id_pad = tf.keras.preprocessing.sequence.pad_sequences(text_ids, max_len).astype(np.int32)
    return text_id_pad

def clean_data(sentences):
    return sentences
    for i in range(sentences.shape[0]):
        sentences[i] = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentences[i])
        sentences[i] = re.sub(r"\'s", " \'s", sentences[i])
        sentences[i] = re.sub(r"\'ve", " \'ve", sentences[i])
        sentences[i] = re.sub(r"n\'t", " n\'t", sentences[i])
        sentences[i] = re.sub(r"\'re", " \'re", sentences[i])
        sentences[i] = re.sub(r"\'d", " \'d", sentences[i])
        sentences[i] = re.sub(r"\'ll", " \'ll", sentences[i])
        sentences[i] = re.sub(r",", " , ", sentences[i])
        sentences[i] = re.sub(r"!", " ! ", sentences[i])
        sentences[i] = re.sub(r"\(", " \( ", sentences[i])
        sentences[i] = re.sub(r"\)", " \) ", sentences[i])
        sentences[i] = re.sub(r"\?", " \? ", sentences[i])
        sentences[i] = re.sub(r"\s{2,}", " ", sentences[i])
    return sentences

def load_sst2_data(data_dir, max_len):
    train_path = os.path.join(data_dir, 'train.tsv')
    val_path = os.path.join(data_dir, 'dev.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')
    assert os.path.isfile(train_path) \
           and os.path.isfile(val_path) \
           and os.path.isfile(test_path)
    train_text_and_label = pd.read_csv(train_path, delimiter='\t', header=None).to_numpy()
    val_text_and_label = pd.read_csv(val_path, delimiter='\t', header=None).to_numpy()
    test_text_and_label = pd.read_csv(test_path, delimiter='\t', header=None).to_numpy()
    train_x = clean_data(train_text_and_label[:, 0])
    val_x = clean_data(val_text_and_label[:, 0])
    test_x = clean_data(test_text_and_label[:, 0])
    train_y = train_text_and_label[:, 1].astype(np.int32)
    val_y = val_text_and_label[:, 1].astype(np.int32)
    test_y = test_text_and_label[:, 1].astype(np.int32)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    all_texts = np.concatenate([train_x, val_x, test_x], axis=0)
    tokenizer.fit_on_texts(all_texts)
    train_x = embed(train_x, tokenizer, max_len)
    val_x = embed(val_x, tokenizer, max_len)
    test_x = embed(test_x, tokenizer, max_len)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer

def decode_sst5_label(labels):
    res = np.zeros(labels.shape[0], dtype=np.int32)
    for idx, label in enumerate(labels):
        res[idx] = int(label[-1]) - 1
    return res

def load_sst5_data(data_dir, max_len):
    train_path = os.path.join(data_dir, 'sst_train.csv')
    val_path = os.path.join(data_dir, 'sst_dev.csv')
    test_path = os.path.join(data_dir, 'sst_test.csv')
    assert os.path.isfile(train_path) \
           and os.path.isfile(val_path) \
           and os.path.isfile(test_path)
    train_text_and_label = pd.read_csv(train_path, delimiter='\t', header=None).to_numpy()
    val_text_and_label = pd.read_csv(val_path, delimiter='\t', header=None).to_numpy()
    test_text_and_label = pd.read_csv(test_path, delimiter='\t', header=None).to_numpy()
    train_x = clean_data(train_text_and_label[:, 1])
    val_x = clean_data(val_text_and_label[:, 1])
    test_x = clean_data(test_text_and_label[:, 1])
    train_y = decode_sst5_label(train_text_and_label[:, 0])
    val_y = decode_sst5_label(val_text_and_label[:, 0])
    test_y = decode_sst5_label(test_text_and_label[:, 0])
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    all_texts = np.concatenate([train_x, val_x, test_x], axis=0)
    tokenizer.fit_on_texts(all_texts)
    train_x = embed(train_x, tokenizer, max_len)
    val_x = embed(val_x, tokenizer, max_len)
    test_x = embed(test_x, tokenizer, max_len)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer
