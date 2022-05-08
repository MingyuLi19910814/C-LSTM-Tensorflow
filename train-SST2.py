import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from model import get_clstm
from data_loader import load_sst2_data, load_sst5_data
from constants import SST2_DATA_DIR, TOKENIZER_PATH_2, EMBEDDING_DIM, SAVED_MODEL_DIR_2
import pickle
from gensim import downloader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs')
parser.add_argument('--num_class', type=int, default=2, help='Number of class')
parser.add_argument('--max_len', type=int, default=16)
parser.add_argument('--filter_sizes', type=str, default='3', help='CNN filter sizes. For CNN, C-LSTM.')
parser.add_argument('--num_filters', type=int, default=150, help='Number of filters per filter size. For CNN, C-LSTM.')
parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units in the LSTM cell')
parser.add_argument('--num_layers', type=int, default=1, help='Number of the LSTM cells')
parser.add_argument('--keep_prob', type=float, default=0.5, help='Dropout keep probability')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg_lambda', type=float, default=1e-3, help='L2 regularization lambda')
parser.add_argument('--refine', action='store_true')

HP_MAX_LEN = [24, 32, 40, 48, 56]
HP_BATCH_SIZES = [16, 32, 40, 48, 64, 128]
HP_LEARNING_RATE = [1e-5, 3e-4, 1e-4]

def train_test_model(args, hparams, google_vocabulary):
    (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer = load_sst2_data(SST2_DATA_DIR, hparams['max_len'])
    with open(TOKENIZER_PATH_2, 'wb') as f:
        pickle.dump(tokenizer, f)
    model = get_clstm(seq_len=hparams['max_len'],
                      word2idx=tokenizer.word_index,
                      embedding_dim=EMBEDDING_DIM,
                      keep_prob=args.keep_prob,
                      filter_sizes=list(map(int, args.filter_sizes.split(','))),
                      filter_num=args.num_filters,
                      lstm_hidden_size=args.hidden_size,
                      lstm_num_layers=args.num_layers,
                      num_class=args.num_class,
                      l2_reg_lambda=args.l2_reg_lambda,
                      refine=args.refine,
                      google_vocabulary=google_vocabulary
                      )

    ckpt_path = os.path.join(SAVED_MODEL_DIR_2, 'best_model.ckpt')
    if args.refine:
        model.load_weights(ckpt_path)
    save_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_best_only=True)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=hparams['learning_rate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc'])
    model.fit(train_x,
              train_y,
              batch_size=hparams['batch_size'],
              epochs=args.num_epochs,
              validation_data=(val_x, val_y),
              callbacks=[save_callback])

    model.load_weights(ckpt_path)
    _, test_accuracy = model.evaluate(test_x, test_y)
    return test_accuracy

def train(args):
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    run_id = 0
    print('loading google news 300 word2vec')
    google_vocabulary = downloader.load('word2vec-google-news-300')
    print('     Done')
    log_file = './grid_search_result_2.txt'
    best_acc = 0
    for max_len in HP_MAX_LEN:
        for batch_size in HP_BATCH_SIZES:
            for learning_rate in HP_LEARNING_RATE:
                hparams = {
                    'max_len': max_len,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
                print('----start trial %s' % run_id)
                print(hparams)
                acc = train_test_model(args, hparams, google_vocabulary)
                best_acc = max(best_acc, acc)
                with open(log_file, 'a') as f:
                    f.write('hparam: {}, acc: {}, best acc = {}\n'.format(hparams, acc, best_acc))

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)

