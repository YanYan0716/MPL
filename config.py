# about dataset
IMG_SIZE = 32
BATCH_SIZE = 32
LABEL_FILE_PATH = '/content/food-11/labeled_1.csv'
UNLABEL_FILE_PATH = '/content/food-11/unlabeled.csv'
AUGMENT_MAGNITUDE = 16
SHUFFLE_SIZE = BATCH_SIZE * 4
DATA_LEN = 9867  # 数据集的总长度

# about model
NUM_XLA_SHARDS = -1
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99
DROPOUT_RATE = 0.2
NUM_CLASSES = 11

# about training
LOG_EVERY = 10
SAVE_EVERY = 1000
TEA_SAVE_PATH = './weights/T'
STD_SAVE_PATH = './weights/S'
MAX_EPOCHS = 100
MAX_STEPS = MAX_EPOCHS * (int(DATA_LEN / BATCH_SIZE)-1)

UDA_WEIGHT = 8  # uda的权重
UDA_STEPS = 5000

# about UdaCrossEntroy
UDA_DATA = 7
LABEL_SMOOTHING = 0.15
UDA_TEMP = 0.7
UDA_THRESHOLD = 0.6

# about learning rate
STUDENT_LR = 0.1  # student
STUDENT_LR_WARMUP_STEPS = 5000
STUDENT_LR_WAIT_STEPS = 3000
TEACHER_LR = 0.1  # teacher
TEACHER_LR_WARMUP_STEPS = 5000
TEACHER_NUM_WAIT_STEPS = 0

LR_DECAY_TYPE = 'cosine'  # constant, exponential, cosine
NUM_DECAY_STEPS = 750
LR_DECAY_RATE = 0.97

# about optimizer
OPTIM_TYPE = 'sgd'  # sgd, momentum, rmsprop
