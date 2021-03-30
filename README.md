# shopee_code_league_2021

# Dependencies
```
transformers
sentencepiece
tqdm
loguru
PyYAML
```

# Step-by-step to reproduce the baseline results (~91% on public leaderboard)

Note: all paths (either in the jupyter notebook file, in the config file or as argument when executing commands) should be modified accordingly.

## 1. Preprocess train and test data

1. First download the datasets and place it in `data/orig/` directory. Create empty directories `data/processed/` and `data/processed/splits`.
2. Execute all cells in `1_EDA.ipynb` to perform EDA (Exploratory Data Analysis) and data preprocessing (tokenization + finding spans) and stratified sampling (10-fold).

The purpose of splitting training to 10 folds was to train on 8 fold (`train`), evaluate on 1 fold (`val`) and test on 1 fold (`test`) for each experiment. Models trained on each "split" will then be used to generate raw predictions (i.e., probabilities, etc.). Combining these raw predictions (i.e., ensemble) will likely give us better results than single model.
However, given that I am just testing the problem statement, I only experimented with several splits and have not done any ensemble.

## 2. Train and evaluate
Execute
```
python train.py -c work_dirs/config.yaml
```
to train using baseline config. Training data will then be saved under `work_dirs/yyyymmdd_hhmmss/`.

## 3. Evaluate

To evaluate model performance on any fold (i.e., with groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/splits/fold_6.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
```
The command above load config from `work_dirs/yyyymmdd_hhmmss/config.yaml`, load model from `work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth` (best model), load data from `data/processed/splits/fold_6.json` (6-th fold) and save prediction results to `work_dirs/yyyymmdd_hhmmss/test_results.csv`. Accuracy will be printed out as well.

To generate prediction results for test set (i.e., without groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/test_processed.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
  --confidence-threshold 0.51
```
Notes:
- `--post-processing`: If set to True, remove predictions where one span contains the other (keep the one with higher score).
- `--confidence-threshold`: This is set to 0.51 to account for model behavior: many predictions with exactly 0.5 confidence score are incorrect.

Using the baseline config, you will be able to achieve ~91% accuracy on public leaderboard.
