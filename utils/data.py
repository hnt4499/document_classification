import json
import random
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding as HfDataCollatorWithPadding

from .utils import from_config


def is_no_overlap(interval_1, interval_2):
    """Assert whether two intervals overlap"""
    x1, y1 = interval_1
    x2, y2 = interval_2

    # Ignore if not present
    if x1 == -1 or x2 == -1:
        return True

    if x1 < x2:
        return y1 <= x2
    else:
        return x1 >= y2


class CustomDataset(Dataset):
    """Dataset for Shopee Code League 2021 - Address Elements Extraction task.

    Parameters
    ----------
    tokenizer
        Tokenizer.
    paths : str or list[str]
        Paths to the json data files.
    p_augmentation : float
        Probability of performing random replacement on POI/street data.
    cls_token : str
        String representation of "cls" token. Might be "<s>" or "[CLS]", depending on the tokenizer.
    sep_token : str
        String representation of "sep" token. Might be "</s>" or "[SEP]", depending on the tokenizer.
    """
    @from_config(requires_all=True)
    def __init__(self, tokenizer, paths, p_augmentation=0.2, cls_token="<s>", sep_token="</s>"):
        super(CustomDataset, self).__init__()
        self.p_augmentation = p_augmentation

        self.tokenizer = tokenizer
        self._cls_idx = tokenizer.convert_tokens_to_ids(cls_token)
        self._sep_idx = tokenizer.convert_tokens_to_ids(sep_token)

        # Read input
        if isinstance(paths, str):
            paths = [paths]

        dfs = []
        for path in paths:
            with open(path, "r") as fin:
                df = json.load(fin)
            df = pd.DataFrame.from_dict(df, orient="index")
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        self._for_training = "poi_start" in self.df.columns

    def __len__(self):
        return len(self.df)

    def _get_item_for_training(self, idx):
        data_info = self.df.iloc[idx]

        # Base infos
        info = data_info[["poi_start", "poi_end", "street_start", "street_end"]].copy()
        info["tokens"] = data_info["raw_address_tok"]
        info["new_poi_idx"] = -1
        info["new_street_idx"] = -1

        # Data augmentation by token replacement
        if not is_no_overlap((info["poi_start"], info["poi_end"] + 1), (info["street_start"], info["street_end"] + 1)):
            do_augmentation = False  # don't augment in case POI and street overlap (very rare)
        else:
            do_augmentation = random.random() < self.p_augmentation

        if do_augmentation:
            to_iterate = [("poi_start", "poi_end", "poi_tok", "new_poi_idx"),
                          ("street_start", "street_end", "street_tok", "new_street_idx")]

            for i, (this_start_str, this_end_str, this_col_name, new_name) in enumerate(to_iterate):
                # Retrieve info
                this_orig_start, this_orig_end = info[this_start_str], info[this_end_str]

                # Check if this is present
                if this_orig_start == -1:
                    continue

                # Sample new tokens
                orig_span_length = this_orig_end - this_orig_start + 1
                while True:
                    new_toks = self.df.sample(n=1).iloc[0]
                    # Continue if it results in too short string
                    if len(new_toks) == 0 and len(info["tokens"]) - orig_span_length < 3:
                        continue

                    info[new_name] = new_toks.name
                    new_toks = new_toks[this_col_name]
                    new_span_length = len(new_toks)
                    break

                # Augment
                tokens = info["tokens"][:this_orig_start] + new_toks + info["tokens"][this_orig_end + 1:]
                if len(new_toks) > 0:
                    new_start = this_orig_start
                    new_end = new_start + new_span_length - 1
                else:
                    new_start = -1
                    new_end = -1

                # Save
                info[this_start_str] = new_start
                info[this_end_str] = new_end
                info["tokens"] = tokens

                # Modify index of the other token
                other_start_str, other_end_str, other_col_name, _ = to_iterate[1 - i]
                other_start, other_end = info[other_start_str], info[other_end_str]
                if other_start > this_orig_end:
                    offset = new_span_length - orig_span_length
                    other_start += offset
                    other_end += offset

                    info[other_start_str] = other_start
                    info[other_end_str] = other_end

        # Tokens
        token_idxs = [self._cls_idx] + self.tokenizer.convert_tokens_to_ids(info["tokens"]) + [self._sep_idx]
        poi_start, poi_end = info["poi_start"], info["poi_end"]
        street_start, street_end = info["street_start"], info["street_end"]
        new_poi_idx, new_street_idx = info["new_poi_idx"], info["new_street_idx"]

        # CLS and SEP were added
        if poi_start != -1:
            poi_start += 1
            poi_end += 1
        if street_start != -1:
            street_start += 1
            street_end += 1

        # POI
        has_poi = (poi_start != -1)

        # Street
        has_street = (street_start != -1)

        return {
            "input_ids": token_idxs, "attention_mask": [1] * len(token_idxs),
            # POI
            "poi_start": poi_start, "poi_end": poi_end,
            "new_poi_idx": new_poi_idx, "has_poi": has_poi,
            # Street
            "street_start": street_start, "street_end": street_end,
            "new_street_idx": new_street_idx, "has_street": has_street,
            # Original info, for debugging
            "orig": data_info.to_dict()
        }

    def _get_item_for_testing(self, idx):
        data_info = self.df.iloc[idx]

        # Tokens
        token_idxs = [self._cls_idx] + self.tokenizer.convert_tokens_to_ids(
            data_info["raw_address_tok"]) + [self._sep_idx]

        return {
            "input_ids": token_idxs, "attention_mask": [1] * len(token_idxs),
        }

    def __getitem__(self, idx):
        if self._for_training:
            return self._get_item_for_training(idx)
        else:
            return self._get_item_for_testing(idx)


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
