
# 0 not visage
# 1 visage
TRAIN_DATA = 'data/start_deep/train_images/'
TRAIN_DATA_NOT_FACE = 'data/start_deep/train_images/0'
TRAIN_DATA_FACE = 'data/start_deep/train_images/1'
CLASSIFIED_TRAIN_DATA = 'data/start_deep/posneg.txt'
CLASSIFIED_TRAIN_DATA_RANDOM = 'data/start_deep/posneg_r.txt'
CLASSIFIED_TRAIN_DATA_RANDOMTEST = 'data/start_deep/testposneg_r2'

CLASSIFIED_TRAIN_DATA_55000 = 'data/start_deep/train55000.txt'
CLASSIFIED_VALID_DATA_36000 = 'data/start_deep/valid36000'

TEST_DATA = 'data/start_deep/'
TEST_DATA_GOOGLE = 'test_images/googlefaces_test'
TEST_DATA_GOOGLE_2 = 'test_images/google_images02_36x36'
TEST_DATA_YALE = 'test_images/yalefaces_test'
TEST_DATA_FACE = 'new_set'
CLASSIFIED_TEST_DATA = 'data/start_deep/testposneg.txt'

CLASSIFIED_TEST_DATA_GOOGLE = 'data/start_deep/google_posneg.txt'
CLASSIFIED_TEST_DATA_GOOGLE2 = 'data/start_deep/google2_posneg.txt'
CLASSIFIED_TEST_DATA_YALE = 'data/start_deep/yale_posneg.txt'

USE_TUTO = False
SAVE_NOT_DETECTED = False
SAVE_MODEL = False
MODEL_DIR = 'models/'
LOAD_MODEL = False
LOAD_MODEL_NAME = ''

SCHEDULER = False
WORKERS = 1
BATCH_SIZE = 16
NB_ITERATIONS = 30
MODEL_NAME = 'project_over'

LEARNING_RATE = 0.01
# MOMENTUM = 0.2
MOMENTUM = 0.2
#MOMENTUM = 0.9

IMAGE_PATH = "./bigimage/nasa.jpg"
CONFIDENCE = 5.9
