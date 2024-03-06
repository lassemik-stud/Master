EXPECTED_DATASETS_FOLDER = "../datasets/"
EXPECTED_PREPROCESSED_DATASETS_FOLDER = "../preprocessed_datasets/"

EXPECTED_PREPROCESSED_DATASET_FILES = {
    'pan13-test': "pan13-test.jsonl",
    'pan13-train': "pan13-train.jsonl",
    'pan20-train-small': "pan20-train-small.jsonl",
    'pan20-train-large': "pan20-train-large.jsonl",
    'pan20-test': "pan20-test.jsonl",
    'pan21-test': "pan21-test.jsonl"
}

EXPECTED_DATASETS_FOLDERS_PAN13 = {
    'pan13-test': "pan13-authorship-verification-test-corpus2-2013-05-29",
    'pan13-train': "pan13-authorship-verification-training-corpus-2013-02-01"
}

EXPECTED_DATASETS_FILES_PAN13 = {
    'pan13-test-truth': EXPECTED_DATASETS_FOLDERS_PAN13['pan13-test'] + "/truth.txt",
    'pan13-train-truth': EXPECTED_DATASETS_FOLDERS_PAN13['pan13-train'] + "/truth.txt"
}

EXPECTED_DATASETS_FILES_PAN20 = {
    'pan20-test' : "pan20-authorship-verification-test.jsonl",
    'pan20-test-truth' : "pan20-authorship-verification-test-truth.jsonl",
    'pan20-train-large': "pan20-authorship-verification-training-large.jsonl",
    'pan20-train-large-truth': "pan20-authorship-verification-training-large-truth.jsonl",
    'pan20-train-small':"pan20-authorship-verification-training-small.jsonl",
    'pan20-train-small-truth':"pan20-authorship-verification-training-small-truth.jsonl",
    'pan21-test':"pan21-authorship-verification-test.jsonl",
    'pan21-test-truth':"pan21-authorship-verification-test-truth.jsonl"
}

EXPECTED_DATASETS_FILES_PAN23 = {
    'pan23-train': "pan23_authorship-verification-training-dataset",
    'pan23-test': "pan23-authorship-verification-test-dataset"
}