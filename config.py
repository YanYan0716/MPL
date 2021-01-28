# about dataset
IMG_SIZE = 32
BATCH_SIZE = 64
LABEL_FILE_PATH = './dataset/label/data.csv'
UNLABEL_FILE_PATH = './dataset/unlabel/data.csv'
AUGMENT_MAGNITUDE = 16
SHUFFLE_SIZE = BATCH_SIZE * 16

# about model
NUM_XLA_SHARDS = -1
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99
DROPOUT_RATE = 0.2
NUM_CLASSES = 10

# about training
SAVE_EVERY = 1000

# about UdaCrossEntroy
UDA_DATA = 7
