import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_PATH = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")


TARGET_COL = "label"          # 0 = normal, 1 = anomalie
ATTACK_CAT_COL = "attack_cat"  
ID_COL = "id"


CATEGORICAL_COLS = ["proto", "service", "state"]


RANDOM_STATE = 42


EPOCHS = 15
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1

os.makedirs(RESULTS_DIR, exist_ok=True)
