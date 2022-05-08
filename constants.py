import os


EMBEDDING_DIM = 300
SAVED_MODEL_DIR = './saved_model/'
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(SAVED_MODEL_DIR, 'tokenizer.bin')
SST2_DATA_DIR = './data/SST-2'
SST5_DATA_DIR = './data/SST-5'