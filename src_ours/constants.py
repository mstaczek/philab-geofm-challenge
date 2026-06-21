TRAIN_INPUT_FOLDERS = [
    "alphaearth_emb",
    "terramind_s1_emb",
    "terramind_s2_emb",
    "tessera_emb",
    "thor_s1_emb",
    "thor_s2_emb",
]

TEST_INPUT_FOLDERS = [
    "alphaearth_test_emb",
    "terramind_test_s1_emb",
    "terramind_test_s2_emb",
    "tessera_test_emb",
    "thor_test_s1_emb",
    "thor_test_s2_emb",
]

TRAIN_DATASET_FOLDERS = {
    "alphaearth": "alphaearth_emb",
    "terraminds1": "terramind_s1_emb",
    "terraminds2": "terramind_s2_emb",
    "tessera": "tessera_emb",
    "thors1": "thor_s1_emb",
    "thors2": "thor_s2_emb",
}

TEST_DATASET_FOLDERS = {
    "alphaearth": "alphaearth_test_emb",
    "terraminds1": "terramind_test_s1_emb",
    "terraminds2": "terramind_test_s2_emb",
    "tessera": "tessera_test_emb",
    "thors1": "thor_test_s1_emb",
    "thors2": "thor_test_s2_emb",
}

LABEL_FOLDER = "labels"

SOURCE_ROOT_TIF = "data/embed2heights/data"
SOURCE_ROOT_NPY = "data/embed2heights_npy"
HEIGHT_NORM_CONSTANT = 30.0

