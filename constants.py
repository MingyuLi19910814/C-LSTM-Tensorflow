import os


EMBEDDING_DIM = 300
SAVED_MODEL_DIR = './saved_model/'
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(SAVED_MODEL_DIR, 'tokenizer.bin')

SAVED_MODEL_DIR_2 = './saved_model_2/'
os.makedirs(SAVED_MODEL_DIR_2, exist_ok=True)
TOKENIZER_PATH_2 = os.path.join(SAVED_MODEL_DIR_2, 'tokenizer.bin')
SST2_DATA_DIR = './data/SST-2'
SST5_DATA_DIR = './data/SST-5'