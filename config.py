# about dataset
IMG_SIZE = 32
BATCH_SIZE = 1
LABEL_FILE_PATH = './dataset/label/data.csv'
UNLABEL_FILE_PATH = './dataset/unlabel/data.csv'
AUGMENT_MAGNITUDE = 16
SHUFFLE_SIZE = BATCH_SIZE * 16
DATA_LEN = 10 # 数据集的总长度

# about model
NUM_XLA_SHARDS = -1
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99
DROPOUT_RATE = 0.2
NUM_CLASSES = 10

# about training
SAVE_EVERY = 1000
MAX_EPOCHS = 100
MAX_STEPS = MAX_EPOCHS*(DATA_LEN/BATCH_SIZE)

UDA_WEIGHT = 8  #uda的权重
UDA_STEPS = 5000

# about UdaCrossEntroy
UDA_DATA = 7
LABEL_SMOOTHING = 0.15
UDA_TEMP = 0.7
UDA_THRESHOLD = 0.6
GLOBAL_STEP = 1

# about learning rate
STUDENT_LR = 0.1  # student
STUDENT_LR_WARMUP_STEPS = 5000
STUDENT_LR_WAIT_STEPS = 3000
STUDENT_LR_DECAY_TYPE = 'exponential'  # constant, exponential, cosine
TEACHER_LR = 0.1  # teacher
TEACHER_LR_WARMUP_STEPS = 5000
TEACHER_NUM_WARMUP_STEPS = 0

NUM_DECAY_STEPS = 750
LR_DECAY_RATE = 0.97

# about optimizer
OPTIM_TYPE = 'sgd'  # sgd, momentum, rmsprop