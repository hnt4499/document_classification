# DBpedia document classification

# Dependencies
```
transformers
sentencepiece
tqdm
loguru
PyYAML
pandas
scikit-learn
```

# Step-by-step to reproduce the baseline results

Note: all paths (either in the jupyter notebook file, in the config file or as argument when executing commands) should be modified accordingly.

## 1. Preprocess train and test data

1. First download the datasets and place it in `data/orig/` directory. Create empty directories `data/processed/`.
2. Execute all cells in `1_EDA.ipynb` to perform EDA (Exploratory Data Analysis) and data preprocessing (tokenization + finding spans) and stratified sampling (10-fold).

The purpose of splitting training to 10 folds was to train on 8 fold (`train`), evaluate on 1 fold (`val`) and test on 1 fold (`test`) for each experiment. Models trained on each "split" will then be used to generate raw predictions (e.g., probabilities, etc.). Combining these raw predictions (i.e., ensemble) will likely give us better results than single model.
Personally, I only experimented with several splits and have not done any ensemble.

## 2. Train and evaluate
Execute
```
python train.py -c work_dirs/config.yaml
```
to train using baseline config. Training data will then be saved under `work_dirs/yyyymmdd_hhmmss/`.

## 3. Evaluate

To evaluate model performance on any fold (i.e., with groundtruth labels), execute, for example:
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/split_6.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
```
The command above load config from `work_dirs/yyyymmdd_hhmmss/config.yaml`, load model from `work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth` (best model), load data from `data/processed/split_6.json` (6-th fold) and save prediction results to `work_dirs/yyyymmdd_hhmmss/test_results.csv`. Accuracy will be printed out as well.

To generate prediction results for test set (i.e., without groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/test_processed.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
```
