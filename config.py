import tensorflow as tf

# about dataset
IMG_SIZE = 32
BATCH_SIZE = 64
# LABEL_FILE_PATH = '/content/cifar/label4000.csv' # google
# UNLABEL_FILE_PATH = '/content/cifar/train.csv'

_MAX_LEVEL = 10
CUTOUT_CONST = 40.
TRANSLATE_CONST = 100.
REPLACE_COLOR = [128, 128, 128]


LABEL_FILE_PATH = '../input/cifar10/cifar/label4000.csv'  # kaggle
UNLABEL_FILE_PATH = '../input/cifar10/cifar/train.csv'


AUGMENT_MAGNITUDE = 16
SHUFFLE_SIZE = BATCH_SIZE * 16
DATA_LEN = 4000  # 数据集的总长度

# about model
NUM_XLA_SHARDS = -1
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.999
DROPOUT_RATE = 0.
DROPOUT = 0.
NUM_CLASSES = 10
NUM_CLASS = 10

# about training
LOG_EVERY = 20
SAVE_EVERY = 5
TEA_SAVE_PATH = './weights/T'
STD_SAVE_PATH = './weights/S'

MAX_EPOCHS = 19200
MAX_STEPS = MAX_EPOCHS * (int(DATA_LEN / BATCH_SIZE)-1)
UDA_WEIGHT = 8  # uda的权重
UDA_STEPS = 20000
TEST_EVERY = 2
GRAD_BOUND = 1e9

# continue train
TEA_CONTINUE = True
STD_CONTINUE = True
TEA_LOAD_PATH = '../input/weights/weights/T'
STD_LOAD_PATH = '../input/weights/weights/S'
CONTINUE_EPOCH = 599

# about testing
# TEST_FILE_PATH = '/content/cifar/test.csv'
TEST_FILE_PATH = '../input/cifar10/cifar/test.csv'
TEST_MODEL_PATH = './weights/S'

# about UdaCrossEntroy
UDA_DATA = 7
LABEL_SMOOTHING = 0.15
UDA_TEMP = 0.7
UDA_THRESHOLD = 0.6

# about learning rate
STUDENT_LR = 0.005  # student
STUDENT_LR_WARMUP_STEPS = 10000
STUDENT_LR_WAIT_STEPS = 20000
TEACHER_LR = 0.007  # teacher
TEACHER_LR_WARMUP_STEPS = 10000
TEACHER_NUM_WAIT_STEPS = 0

LR_DECAY_TYPE = 'cosine'  # constant, exponential, cosine
NUM_DECAY_STEPS = 3000
LR_DECAY_RATE = 0.97

# about optimizer
OPTIM_TYPE = 'sgd'  # sgd, momentum, rmsprop
WEIGHT_DECAY = 5e-4


# dtype
DTYPE = tf.float32