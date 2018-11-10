
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
CLASSIFIED_TEST_DATA = 'data/start_deep/testposneg.txt'

CLASSIFIED_TEST_DATA_GOOGLE = 'data/start_deep/google_posneg.txt'
CLASSIFIED_TEST_DATA_GOOGLE2 = 'data/start_deep/google2_posneg.txt'
CLASSIFIED_TEST_DATA_YALE = 'data/start_deep/yale_posneg.txt'

SAVE_NOT_DETECTED = True
SAVE_MODEL = True
MODEL_DIR = 'models/'
LOAD_MODEL = False

SCHEDULER = False
WORKERS = 1
BATCH_SIZE = 10
NB_ITERATIONS = 10
MODEL_NAME = 'autostop'

# LEARNING_RATE = 0.000001
LEARNING_RATE = 0.001
# MOMENTUM = 0.2
MOMENTUM = 0.5
# MOMENTUM = 0.9
