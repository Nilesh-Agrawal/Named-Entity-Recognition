# Named-Entity-Recognition
This repository contains both deep sequence tagging model which uses word level + Char level embedding with CRF.
Also it contains feature level CRF model which uses pycrfsuite with additional features added.

1. To run CRF Model which uses pycrfsuite.
pip install pycrfsuite
And then run python Notebook NLP_Assignment3_NER.ipynb

2. To run LSTM Word+Char CRF Model
pip install tensorflow
To train model run file generate_model
Then to evaluate model on private test file run python evaluate_test.py "testfilepath". Since model is already there you don't need to train it.
