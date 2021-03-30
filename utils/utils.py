import time
import inspect
from functools import partial

import torch
import pandas as pd
from loguru import logger


def to_device(x, device):
    if not isinstance(x, dict):
        return x

    new_x = {}

    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            new_v = v.to(device)
        elif isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            new_v = [i.to(device) for i in v]
        else:
            new_v = v

        new_x[k] = new_v

    return new_x


def aggregate_dict(x):
    """Aggregate a list of dict to form a new dict"""
    agg_x = {}

    for ele in x:
        assert isinstance(ele, dict)

        for k, v in ele.items():
            if k not in agg_x:
                agg_x[k] = []

            if isinstance(v, (tuple, list)):
                agg_x[k].extend(list(v))
            else:
                agg_x[k].append(v)

    # Stack if possible
    new_agg_x = {}
    for k, v in agg_x.items():
        try:
            v = torch.cat(v, dim=0)
        except Exception:
            pass
        new_agg_x[k] = v

    return new_agg_x


def raise_or_warn(action, msg):
    if action == "raise":
        raise ValueError(msg)
    else:
        logger.warning(msg)


class ConfigComparer:
    """Compare two config dictionaries. Useful for checking when resuming from
    previous session."""

    _to_raise_error = [
        "model->model_name_or_path"
    ]
    _to_warn = [
        "model->config_name", "model->tokenizer_name", "model->cache_dir", "model->freeze_base_model", "model->fusion",
        "model->lambdas"
    ]

    def __init__(self, cfg_1, cfg_2):
        self.cfg_1 = cfg_1
        self.cfg_2 = cfg_2

    def compare(self):
        for components, action in \
                [(self._to_raise_error, "raise"), (self._to_warn, "warn")]:
            for component in components:
                curr_scfg_1, curr_scfg_2 = self.cfg_1, self.cfg_2  # subconfigs
                for key in component.split("->"):
                    if key not in curr_scfg_1 or key not in curr_scfg_2:
                        raise ValueError(
                            f"Component {component} not found in config file.")
                    curr_scfg_1 = curr_scfg_1[key]
                    curr_scfg_2 = curr_scfg_2[key]
                if curr_scfg_1 != curr_scfg_2:
                    msg = (f"Component {component} is different between "
                           f"two config files\nConfig 1: {curr_scfg_1}\n"
                           f"Config 2: {curr_scfg_2}.")
                    raise_or_warn(action, msg)
        return True


def collect(config, args, collected):
    """Recursively collect each argument in `args` from `config` and write to
    `collected`."""
    if not isinstance(config, dict):
        return

    keys = list(config.keys())
    for arg in args:
        if arg in keys:
            if arg in collected:  # already collected
                raise RuntimeError(f"Found repeated argument: {arg}")
            collected[arg] = config[arg]

    for key, sub_config in config.items():
        collect(sub_config, args, collected)


def from_config(main_args=None, requires_all=False):
    """Wrapper for all classes, which wraps `__init__` function to take in only
    a `config` dict, and automatically collect all arguments from it. An error
    is raised when duplication is found. Note that keyword arguments are still
    allowed, in which case they won't be collected from `config`.

    Parameters
    ----------
    main_args : str
        If specified (with "a->b" format), arguments will first be collected
        from this subconfig. If there are any arguments left, recursively find
        them in the entire config. Multiple main args are to be separated by
        ",".
    requires_all : bool
        Whether all function arguments must be found in the config.
    """
    global_main_args = main_args
    if global_main_args is not None:
        global_main_args = global_main_args.split(",")
        global_main_args = [args.split("->") for args in global_main_args]

    def decorator(init):
        init_args = inspect.getfullargspec(init)[0][1:]  # excluding self

        def wrapper(self, config=None, main_args=None, **kwargs):
            # Add config to self
            if config is not None:
                self.config = config

            # Get config from self
            elif getattr(self, "config", None) is not None:
                config = self.config

            if main_args is None:
                main_args = global_main_args
            else:
                # Overwrite global_main_args
                main_args = main_args.split(",")
                main_args = [args.split("->") for args in main_args]

            collected = kwargs  # contains keyword arguments
            not_collected = [arg for arg in init_args if arg not in collected]
            # Collect from main args
            if config is not None and main_args is not None \
                    and len(not_collected) > 0:
                for main_arg in main_args:
                    sub_config = config
                    for arg in main_arg:
                        if arg not in sub_config:
                            break  # break when `main_args` is invalid
                        sub_config = sub_config[arg]
                    else:
                        collect(sub_config, not_collected, collected)
                    not_collected = [arg for arg in init_args
                                     if arg not in collected]
                    if len(not_collected) == 0:
                        break
            # Collect from the rest
            not_collected = [arg for arg in init_args if arg not in collected]
            if config is not None and len(not_collected) > 0:
                collect(config, not_collected, collected)
            # Validate
            if requires_all and (len(collected) < len(init_args)):
                not_collected = [arg for arg in init_args
                                 if arg not in collected]
                raise RuntimeError(
                    f"Found missing argument(s) when initializing "
                    f"{self.__class__.__name__} class: {not_collected}.")
            # Call function
            return init(self, **collected)
        return wrapper
    return decorator


class Timer:
    def __init__(self):
        self.global_start_time = time.time()
        self.start_time = None
        self.last_interval = None
        self.accumulated_interval = None

    def start(self):
        assert self.start_time is None
        self.start_time = time.time()

    def end(self):
        assert self.start_time is not None
        self.last_interval = time.time() - self.start_time
        self.start_time = None

        # Update accumulated interval
        if self.accumulated_interval is None:
            self.accumulated_interval = self.last_interval
        else:
            self.accumulated_interval = (
                0.9 * self.accumulated_interval + 0.1 * self.last_interval)

    def get_last_interval(self):
        return self.last_interval

    def get_accumulated_interval(self):
        return self.accumulated_interval

    def get_total_time(self):
        return time.time() - self.global_start_time


def post_process(poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds,
                 confidence_threshold=0.5, post_processing=False):
    """Perform post processing.
    1. Mask predictions that model is not confident about by -1 (i.e., not present).
    2. If `post_processing=True`:
        a. Remove predictions where one span contains the other (keep the one with higher score).
    """
    # Mask predictions that model is not confident about by -1 (i.e., not present)
    has_poi_preds_all = (poi_existence_preds >= confidence_threshold)  # (B,)
    has_street_preds_all = (street_existence_preds >= confidence_threshold)  # (B,)

    negative_one = torch.tensor(-1).to(poi_span_preds)
    poi_span_preds = torch.where(has_poi_preds_all.unsqueeze(-1), poi_span_preds, negative_one)  # (B, 2)
    street_span_preds = torch.where(
        has_street_preds_all.unsqueeze(-1), street_span_preds, negative_one)  # (B, 2)

    # Remove predictions where one span contains the other (keep the one with higher score)
    if post_processing:
        poi_start_preds, poi_end_preds = poi_span_preds[:, 0], poi_span_preds[:, 1]
        street_start_preds, street_end_preds = street_span_preds[:, 0], street_span_preds[:, 1]

        mask_containing = ((poi_start_preds >= street_start_preds) & (poi_end_preds <= street_end_preds) |
                           (poi_start_preds < street_start_preds) & (poi_end_preds >= street_end_preds))  # (B,)
        mask_containing = mask_containing.unsqueeze(-1)  # (B, 1)
        mask_confidence = (poi_existence_preds > street_existence_preds).unsqueeze(-1)  # (B, 1)

        poi_span_preds = torch.where(mask_containing & (~mask_confidence), negative_one, poi_span_preds)
        street_span_preds = torch.where(mask_containing & mask_confidence, negative_one, street_span_preds)

    return poi_span_preds, street_span_preds


def compute_metrics_from_inputs_and_outputs(inputs, outputs, tokenizer, confidence_threshold=0.5, save_csv_path=None,
                                            post_processing=False, show_progress=False):
    if isinstance(inputs, dict):
        inputs = [inputs]
    if isinstance(outputs, dict):
        outputs = [outputs]

    input_ids_all = []
    has_gt = "poi_start" in inputs[0]

    poi_span_preds_all, street_span_preds_all = [], []
    poi_existence_preds_all, street_existence_preds_all = [], []
    if has_gt:
        poi_span_gt_all, street_span_gt_all = [], []

    if show_progress:
        from tqdm import tqdm
    else:
        tqdm = lambda x, **kwargs: x

    for inputs_i, outputs_i in tqdm(zip(inputs, outputs), desc="Processing predictions"):  # by batch
        input_ids = inputs_i["input_ids"]
        input_ids_all.append(input_ids)

        # Groundtruths
        if has_gt:
            poi_start_gt, poi_end_gt = inputs_i["poi_start"], inputs_i["poi_end"]
            street_start_gt, street_end_gt = inputs_i["street_start"], inputs_i["street_end"]
            # Stack
            poi_span_gt = torch.stack([poi_start_gt, poi_end_gt], dim=-1)  # (B, 2)
            street_span_gt = torch.stack([street_start_gt, street_end_gt], dim=-1)  # (B, 2)

        # Predictions
        poi_span_preds, street_span_preds = outputs_i["poi_span_preds"], outputs_i["street_span_preds"]  # (B, L, 2)
        poi_span_preds = poi_span_preds.argmax(dim=1)  # (B, 2)
        street_span_preds = street_span_preds.argmax(dim=1)  # (B, 2)

        poi_existence_preds = outputs_i["poi_existence_preds"]  # (B,)
        street_existence_preds = outputs_i["street_existence_preds"]  # (B,)

        # Aggregate
        poi_span_preds_all.append(poi_span_preds)
        poi_existence_preds_all.append(poi_existence_preds)
        if has_gt:
            poi_span_gt_all.append(poi_span_gt)

        street_span_preds_all.append(street_span_preds)
        street_existence_preds_all.append(street_existence_preds)
        if has_gt:
            street_span_gt_all.append(street_span_gt)

    # Combine results
    poi_span_preds_all = torch.cat(poi_span_preds_all, dim=0)  # (N, 2), where N is length of the dataset
    poi_existence_preds_all = torch.cat(poi_existence_preds_all, dim=0)  # (N,)
    if has_gt:
        poi_span_gt_all = torch.cat(poi_span_gt_all, dim=0)  # (N, 2)

    street_span_preds_all = torch.cat(street_span_preds_all, dim=0)  # (N, 2)
    street_existence_preds_all = torch.cat(street_existence_preds_all, dim=0)  # (N,)
    if has_gt:
        street_span_gt_all = torch.cat(street_span_gt_all, dim=0)  # (N, 2)

    # Post process
    poi_span_preds_all, street_span_preds_all = post_process(
        poi_span_preds_all, poi_existence_preds_all, street_span_preds_all, street_existence_preds_all,
        confidence_threshold=confidence_threshold, post_processing=post_processing)

    # Calculate accuracy
    if has_gt:
        poi_span_correct_all = (poi_span_gt_all.int() == poi_span_preds_all.int()).all(-1)  # (N,)
        poi_span_acc = poi_span_correct_all.sum() / float(len(poi_span_correct_all))  # scalar

        street_span_correct_all = (street_span_gt_all.int() == street_span_preds_all.int()).all(-1)  # (N,)
        street_span_acc = street_span_correct_all.sum() / float(len(street_span_correct_all))  # scalar

        total_correct_all = poi_span_correct_all & street_span_correct_all
        total_acc = total_correct_all.sum() / float(len(total_correct_all))  # scalar

        acc = {"total_acc": total_acc, "poi_acc": poi_span_acc, "street_acc": street_span_acc}

    # Generate prediction csv if needed
    if save_csv_path is not None:
        assert sum(len(i) for i in input_ids_all) == len(poi_span_preds_all) == len(poi_existence_preds_all) \
            == len(street_span_preds_all) == len(street_existence_preds_all)
        if has_gt:
            assert len(poi_span_preds_all) == len(poi_span_gt_all) == len(street_span_gt_all)

        decode = partial(tokenizer.decode, skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        input_i, input_j = 0, -1
        records = []

        for i, (poi_span_pred, poi_existence_pred, street_span_pred, street_existence_pred) \
                in enumerate(zip(poi_span_preds_all, poi_existence_preds_all,
                                 street_span_preds_all, street_existence_preds_all)):
            # If has groundtruths
            if has_gt:
                poi_span_gt = poi_span_gt_all[i]
                street_span_gt = street_span_gt_all[i]
            # Get index of the `input_ids_all`
            input_j += 1
            if input_j >= len(input_ids_all[input_i]):
                input_i += 1
                input_j = 0
            input_ids = input_ids_all[input_i][input_j].tolist()
            record = {
                "raw_address": decode(input_ids),
            }

            if has_gt:
                to_iterate = [
                    (poi_span_pred, poi_existence_pred, poi_span_gt, "poi"),
                    (street_span_pred, street_existence_pred, street_span_gt, "street")
                ]

                for (pred_start, pred_end), conf_score, (gt_start, gt_end), col_name in to_iterate:
                    gt_str = "" if gt_start == -1 else decode(input_ids[gt_start:gt_end + 1])
                    if pred_end < pred_start:
                        pred_str = "[INVALID]"
                    else:
                        pred_str = "" if pred_start == - 1 else decode(input_ids[pred_start:pred_end + 1])
                    record.update({
                        f"{col_name}_gt": gt_str, f"{col_name}_pred": pred_str,
                        f"has_{col_name}": round(conf_score.cpu().item(), 6),
                    })
            else:
                to_iterate = [
                    (poi_span_pred, poi_existence_pred, "POI"),
                    (street_span_pred, street_existence_pred, "street")
                ]

                for (pred_start, pred_end), conf_score, col_name in to_iterate:
                    if pred_end < pred_start:
                        pred_str = ""
                    else:
                        pred_str = "" if pred_start == - 1 or conf_score < confidence_threshold \
                            else decode(input_ids[pred_start:pred_end + 1])
                    record.update({f"{col_name}": pred_str, f"has_{col_name}": round(conf_score.cpu().item(), 6)})

            records.append(record)

        df = pd.DataFrame.from_records(records)
        if not has_gt:

            def transform(row):
                if row["POI"] == "" and row["street"] == "":
                    return ""
                return f"{row['POI']}/{row['street']}"
            # Generate to the correct format
            df["POI/street"] = df.apply(transform, axis=1)
            df["id"] = df.index
        df.to_csv(save_csv_path, index=False)

    if has_gt:
        return acc
