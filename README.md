# reverse_turing_test
Interpretability in NLP classifiers


File descriptions:

## Helper Scripts: 
* `features.py` contains the featurizing functions that can be used
* `models.py` contains the code for the MLP  using for classification
* `utils.py` contains the evaluation function

## Preprocessing:
* `featurize_downstream.py` calls featurization functions for the downstream tasks (sentiment binary and finegrained)
* `featurize_probing.py` calls featurization for the probing tasks
* `feats_downstream.sh` runs python script for all sets (train, test, dev)
* `feats_probing.sh` runs python featurization script for all probing tasks

## Training:
* `train_classifier_downstream.py` initiates the training for the specified downstream task
* `train_classifier_probing.py` initiates the training for the specified probing task
* `train_probing.sh` runs the python training script for probing tasks. Note: A different classifier architecture is used for sentence length



