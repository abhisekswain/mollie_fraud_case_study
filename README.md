# Mollie - fraud case study

This repository consists the solution for the mollie fraud case study.

The python scripts can be run in the following order:

## 1. Create features
If you haven't done so download the data/csv file and place it in the data folder. Change to the src folder as all the python scripts are there. To generate the necessary data frames for training and inference you need to run the `feature_creation.py` file first. 
```bash
python feature_creation.py
```
To train the model with cross validation run the command:
```bash
python train_crossval.py
```
Since its training on a large dataset and doing a grid search over mulitple models, it will take a long time to run. Thus, the best model from a previous run in the jupyter notebook is provided in the code already.

## 4. Generating results on test set
Run the `main.py` file to generate results on the test set. The `main.py` uses the already trained model to do inference on a test set and outputs the precision, recall and F1 score for the test set. You can also run the notebook, which does the same thing. 
To run the `main.py` file:
```bash
python main.py
```


