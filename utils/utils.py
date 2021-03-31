import time
import inspect
from functools import partial

import torch
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support


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


def compute_metrics_from_inputs_and_outputs(inputs, outputs, tokenizer, save_csv_path=None, show_progress=False):
    if isinstance(inputs, dict):
        inputs = [inputs]
    if isinstance(outputs, dict):
        outputs = [outputs]

    input_ids_all = []
    has_gt = "l1_cls_gt" in inputs[0]

    l1_cls_preds_all, l2_cls_preds_all, l3_cls_preds_all = [], [], []
    l1_probs_preds_all, l2_probs_preds_all, l3_probs_preds_all = [], [], []
    if has_gt:
        l1_cls_gt_all, l2_cls_gt_all, l3_cls_gt_all = [], [], []

    if show_progress:
        from tqdm import tqdm
    else:
        tqdm = lambda x, **kwargs: x

    for inputs_i, outputs_i in tqdm(zip(inputs, outputs), desc="Processing predictions"):  # by batch
        input_ids = inputs_i["input_ids"]
        input_ids_all.append(input_ids)

        # Groundtruths
        if has_gt:
            l1_cls_gt, l2_cls_gt, l3_cls_gt = inputs_i["l1_cls_gt"], inputs_i["l2_cls_gt"], inputs_i["l3_cls_gt"]
            l1_cls_gt_all.append(l1_cls_gt)
            l2_cls_gt_all.append(l2_cls_gt)
            l3_cls_gt_all.append(l3_cls_gt)

        # Predictions
        l1_cls_preds = outputs_i["l1_cls_preds"]
        l1_probs_preds, l1_cls_preds = l1_cls_preds.max(dim=1)  # (B,)
        l1_cls_preds_all.append(l1_cls_preds)
        l1_probs_preds_all.append(l1_probs_preds)

        l2_cls_preds = outputs_i["l2_cls_preds"]
        l2_probs_preds, l2_cls_preds = l2_cls_preds.max(dim=1)  # (B,)
        l2_cls_preds_all.append(l2_cls_preds)
        l2_probs_preds_all.append(l2_probs_preds)

        l3_cls_preds = outputs_i["l3_cls_preds"]
        l3_probs_preds, l3_cls_preds = l3_cls_preds.max(dim=1)  # (B,)
        l3_cls_preds_all.append(l3_cls_preds)
        l3_probs_preds_all.append(l3_probs_preds)

    # Combine results
    l1_cls_preds_all = torch.cat(l1_cls_preds_all, dim=0)  # (N,), where N is length of the dataset
    l1_probs_preds_all = torch.cat(l1_probs_preds_all, dim=0)  # (N,)

    l2_cls_preds_all = torch.cat(l2_cls_preds_all, dim=0)  # (N,)
    l2_probs_preds_all = torch.cat(l2_probs_preds_all, dim=0)  # (N,)

    l3_cls_preds_all = torch.cat(l3_cls_preds_all, dim=0)  # (N,)
    l3_probs_preds_all = torch.cat(l3_probs_preds_all, dim=0)  # (N,)

    if has_gt:
        l1_cls_gt_all = torch.cat(l1_cls_gt_all, dim=0)  # (N,)
        l2_cls_gt_all = torch.cat(l2_cls_gt_all, dim=0)  # (N,)
        l3_cls_gt_all = torch.cat(l3_cls_gt_all, dim=0)  # (N,)

    # Calculate metrics
    if has_gt:
        metrics = {}
        preds_data = [
            ("l1", l1_cls_preds_all, l1_cls_gt_all),
            ("l2", l2_cls_preds_all, l2_cls_gt_all),
            ("l3", l3_cls_preds_all, l3_cls_gt_all),
        ]
        for level, level_preds, level_gt in preds_data:
            for t in ["micro", "macro"]:
                precision, recall, f1, support = precision_recall_fscore_support(
                    level_gt.cpu().numpy(), level_preds.cpu().numpy(), average=t, zero_division=1)
                metrics[f"{level}_precision_{t}"] = precision
                metrics[f"{level}_recall_{t}"] = recall
                metrics[f"{level}_f1_{t}"] = f1

    # Generate prediction csv if needed
    if save_csv_path is not None:
        decode = partial(tokenizer.decode, skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        input_i, input_j = 0, -1
        records = []

        for i, (l1_cls_pred, l1_prob_pred, l2_cls_pred, l2_prob_pred, l3_cls_pred, l3_prob_pred) \
                in enumerate(zip(l1_cls_preds_all, l1_probs_preds_all,
                                 l2_cls_preds_all, l2_probs_preds_all,
                                 l3_cls_preds_all, l3_probs_preds_all)):
            # If has groundtruths
            if has_gt:
                l1_cls_gt = l1_cls_gt_all[i]
                l2_cls_gt = l2_cls_gt_all[i]
                l3_cls_gt = l3_cls_gt_all[i]
            # Get index of the `input_ids_all`
            input_j += 1
            if input_j >= len(input_ids_all[input_i]):
                input_i += 1
                input_j = 0
            input_ids = input_ids_all[input_i][input_j].tolist()
            record = {
                "text": decode(input_ids),
            }

            if has_gt:
                to_iterate = [
                    (l1_cls_pred, l1_prob_pred, l1_cls_gt, "l1"),
                    (l2_cls_pred, l2_prob_pred, l2_cls_gt, "l2"),
                    (l3_cls_pred, l3_prob_pred, l3_cls_gt, "l3"),
                ]

                for cls_pred, prob_pred, cls_gt, col_name in to_iterate:
                    record.update({
                        f"{col_name}_gt": cls_gt, f"{col_name}_pred": cls_pred,
                        f"{col_name}_pred_prob": prob_pred,
                    })
            else:
                to_iterate = [
                    (l1_cls_pred, l1_prob_pred, "l1"),
                    (l2_cls_pred, l2_prob_pred, "l2"),
                    (l3_cls_pred, l3_prob_pred, "l3"),
                ]

                for cls_pred, prob_pred, col_name in to_iterate:
                    record.update({
                        f"{col_name}_pred": cls_pred,
                        f"{col_name}_pred_prob": prob_pred,
                    })

            records.append(record)

        df = pd.DataFrame.from_records(records)
        df.to_csv(save_csv_path, index=False)

    if has_gt:
        return metrics
