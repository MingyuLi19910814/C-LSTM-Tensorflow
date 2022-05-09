import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from model import get_clstm
from data_loader import load_sst2_data, load_sst5_data
from constants import SST2_DATA_DIR, SST5_DATA_DIR, TOKENIZER_PATH, EMBEDDING_DIM, SAVED_MODEL_DIR
import pickle
from gensim import downloader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SST-5', choices=['SST-2', 'SST-5'])
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs')
parser.add_argument('--num_class', type=int, default=5, help='Number of class')
parser.add_argument('--max_len', type=int, default=48)
parser.add_argument('--filter_sizes', type=str, default='3', help='CNN filter sizes')
parser.add_argument('--num_filters', type=int, default=150, help='Number of filters per filter size')
parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units in the LSTM cell')
parser.add_argument('--num_layers', type=int, default=1, help='Number of the LSTM cells')
parser.add_argument('--keep_prob', type=float, default=0.5, help='Dropout keep probability')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg_lambda', type=float, default=1e-3, help='L2 regularization lambda')

def train(args):
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    if args.dataset == 'SST-5':
        (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer = load_sst5_data(SST5_DATA_DIR, args.max_len)
    elif args.dataset == 'SST-2':
        (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer = load_sst2_data(SST2_DATA_DIR, args.max_len)
    else:
        raise NotImplementedError("custom dataset is not implemented yet")
    print('tokenizer shape = {}'.format(tokenizer.word_index.__len__()))
    # save the tokenizer to file so that it could be used for inference
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print('loading google news 300 word2vec')
    google_vocabulary = downloader.load('word2vec-google-news-300')
    print('     Done')
    # generate the C-LSTM model with one convolutional layer and one LSTM layer as described in the paper
    model = get_clstm(seq_len=args.max_len,
                      word2idx=tokenizer.word_index,
                      embedding_dim=EMBEDDING_DIM,
                      keep_prob=args.keep_prob,
                      filter_sizes=list(map(int, args.filter_sizes.split(','))),
                      filter_num=args.num_filters,
                      lstm_hidden_size=args.hidden_size,
                      lstm_num_layers=args.num_layers,
                      num_class=args.num_class,
                      l2_reg_lambda=args.l2_reg_lambda,
                      google_vocabulary=google_vocabulary)

    ckpt_path = os.path.join(SAVED_MODEL_DIR, 'best_model.ckpt')
    # callback for saving the best model
    save_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                       save_weights_only=True,
                                                       save_best_only=True)
    # the paper only mentioned to use SGD with RMS.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc'])
    model.fit(train_x,
              train_y,
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              validation_data=(val_x, val_y),
              callbacks=[save_callback])
    print('*' * 20 + ' test ' + '*' * 20)
    # load the weight with minimum val_loss
    model.load_weights(ckpt_path)
    _, test_accuracy = model.evaluate(test_x, test_y)
    print('test acc = {}'.format(test_accuracy))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)

