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
MAX_STEP = 100

# about UdaCrossEntroy
UDA_DATA = 7
LABEL_SMOOTHING = 0.15
UDA_TEMP = 0.7
UDA_THRESHOLD = 0.6

# about learning rate
MPL_STUDENT_LR = 0.1
MPL_STUDENT_LR_WARMUP_STEPS = 5000
MPL_STUDENT_LR_WAIT_STEPS = 3000
LR_DECAY_TYPE = 'exponential'  # constant, exponential, cosine
NUM_DECAY_STEPS = 750
LR_DECAY_RATE = 0.97

# about optimizer
OPTIM_TYPE = 'sgd'  # sgd, momentum, rmsprop