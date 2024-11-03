from experiments.b1.pan20_baseline_1_LR import pan20_b1_LR_tfidf, pan20_b1_LR_dependency
from experiments.b1.pan20_baseline_1_NB import pan20_b1_NB_tfidf, pan20_b1_NB_dependency
from experiments.b1.pan20_baseline_1_SVM import pan20_b1_SVM_tfidf, pan20_b1_SVM_dependency
from pipe import run_experiment

_NAME = 'pan20_b1_LR_dependency'
run_experiment(pan20_b1_LR_dependency(_NAME), _NAME)

_NAME = 'pan20_b1_NB_dependency'
run_experiment(pan20_b1_NB_dependency(_NAME), _NAME)

_NAME = 'pan20_b1_SVM_dependency'
run_experiment(pan20_b1_SVM_dependency(_NAME), _NAME)

_NAME = 'pan20_b1_LR_tfidf'
run_experiment(pan20_b1_LR_tfidf(_NAME), _NAME)

_NAME = 'pan20_b1_NB_tfidf'
run_experiment(pan20_b1_NB_tfidf(_NAME), _NAME) 

_NAME = 'pan20_b1_SVM_tfidf'
run_experiment(pan20_b1_SVM_tfidf(_NAME), _NAME)