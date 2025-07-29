NORMAL_FLAG = False #归一化
eps=1e-6
IMAGE_DEPTH = 32
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

MIN_BOUND = 0.0
MAX_BOUND = 1.0

CROSS_VAL = False

FOLD =10

TEST_FOLD = [0,1,2,3,4,5,6,7,8,9]

TRAIN_DATA_PATH = 'D:/lung_nodule_project/DATASET/LUNA16_patch/DATA_10_FOLD/original'
VALIDATE_DATA_PATH = 'D:/lung_nodule_project/DATASET/LUNA16_patch/DATA_10_FOLD/original'
TEST_DATA_PATH = 'D:/lung_nodule_project/DATASET/LUNA16_patch/DATA_10_FOLD/original/fold{}'

MODEL_WEIGHT = None
NEW_LOSS = FAlse
ALPHA = 0.45
GAMMA =2

LOG_PATH =
