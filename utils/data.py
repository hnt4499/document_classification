import json
import random
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding as HfDataCollatorWithPadding

from .utils import from_config


class CustomDataset(Dataset):
    """Dataset for DBpedia classification task.

    Parameters
    ----------
    tokenizer
        Tokenizer.
    paths : str or list[str]
        Paths to the json data files.
    max_word_count: int
        Maximum number of words (default: None).
    min_word_count: int
        Minimum number of words (default: 50)
    p_augmentation : float
        Probability of performing augmentation (random boundary augmentation) (default: 1.0).
    """
    @from_config(requires_all=True)
    def __init__(self, tokenizer, paths, mapping_path, max_word_count=150, min_word_count=50, p_augmentation=1.0):
        super(CustomDataset, self).__init__()
        assert max_word_count > min_word_count

        self.tokenizer = tokenizer
        self.paths = paths
        self.mapping_path = mapping_path

        self.max_word_count = max_word_count
        self.min_word_count = min_word_count
        self.p_augmentation = p_augmentation

        # Read input
        if isinstance(paths, str):
            paths = [paths]

        dfs = [pd.read_csv(path) for path in paths]
        self.df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        self.df["wiki_name"] = self.df["wiki_name"].str.replace("_", " ")  # preprocess

        # Mapping
        with open(mapping_path, "r") as fin:
            self.mapping = json.load(fin)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        text, title, l1, l2, l3 = data[["text", "wiki_name", "l1", "l2", "l3"]]
        text = text.split()

        # Augmentation
        do_augmentation = random.random() < self.p_augmentation
        if do_augmentation and len(text) > self.min_word_count:
            start = random.randint(0, len(text) - self.min_word_count)
            end = random.randint(start + self.min_word_count, min(start + self.max_word_count, len(text)))
            text = text[start:end]
        elif len(text) > self.max_word_count:  # during inference
            text = text[:self.max_word_count]
        text = " ".join(text)

        # Tokenize and encode
        input_ids = self.tokenizer.encode(title, text_pair=text)

        # Category labels
        l1 = self.mapping["l1"]["cls2idx"][l1]
        l2 = self.mapping["l2"]["cls2idx"][l2]
        l3 = self.mapping["l3"]["cls2idx"][l3]

        return {
            "input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "l1_cls_gt": l1, "l2_cls_gt": l2, "l3_cls_gt": l3
        }


class DataCollatorWithPadding(HfDataCollatorWithPadding):
    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Wrap the parent's call function by selectively pass dict of list/tensors to it"""

        selected_features, filtered_features = [], []
        for feature in features:
            selected_feature, filtered_feature = {}, {}
            for key, value in feature.items():
                if isinstance(value, (list, torch.Tensor)):
                    selected_feature[key] = value
                else:
                    filtered_feature[key] = value

            selected_features.append(selected_feature)
            filtered_features.append(filtered_feature)

        # Collate selected features
        selected_features = super(DataCollatorWithPadding, self).__call__(selected_features)

        # Collate filtered features
        all_keys = [tuple(filtered_feature.keys()) for filtered_feature in filtered_features]
        assert len(set(all_keys)) == 1
        all_keys = all_keys[0]
        collated_filtered_features = {}

        for key in all_keys:
            collated_filtered_feature = [filtered_feature[key] for filtered_feature in filtered_features]
            try:
                collated_filtered_feature = torch.tensor(collated_filtered_feature)
            except Exception:
                pass
            collated_filtered_features[key] = collated_filtered_feature

        # Combine everything together
        collated_features = {}
        collated_features.update(selected_features)
        collated_features.update(collated_filtered_features)

        return collated_features
