# Retrieves the authors that are both in TRAIN and TEST with 30 texts. 
# Used for baseline 1 and 2 to create a new dataset using the specified author.

DATASET_PATH=../../datasets/
PAN20_TRAIN_TRUTH=pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl
PAN20_TEST_TRUTH=pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl

TRAIN_AUTHOR_FILE=pan20_train_authors.txt
TEST_AUTHOR_FILE=pan20_test_authors.txt
SIMILAR_AUTHOR_FILE=pan20_similar_authors.txt

jq '.authors.[]' $DATASET_PATH$PAN20_TRAIN_TRUTH | sort | uniq -c | awk '$1 == 30 { print $2 }' | sed 's/"//g' | sort > $TRAIN_AUTHOR_FILE
jq '.authors.[]' $DATASET_PATH$PAN20_TEST_TRUTH  | sort | uniq -c | awk '$1 == 30 { print $2 }' | sed 's/"//g' | sort > $TEST_AUTHOR_FILE

grep -f $TRAIN_AUTHOR_FILE $TEST_AUTHOR_FILE > $SIMILAR_AUTHOR_FILE

rm $TRAIN_AUTHOR_FILE $TEST_AUTHOR_FILE