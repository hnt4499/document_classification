import os
import sys
import json
import datetime
from pathlib import Path
from shutil import copy, SameFileError

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import yaml
from loguru import logger
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.model import model_classes
from utils.data import CustomDataset, DataCollatorWithPadding
from utils.utils import (from_config, to_device, compute_metrics_from_inputs_and_outputs,
                         ConfigComparer, Timer)


class Trainer:
    @from_config(requires_all=True)
    def __init__(self, config_path):
        self.action = self.config["action"]
        # Get save dir
        self._get_save_dir()
        # Get logger
        self._get_logger()
        # Print config
        logger.info(f"Config:\n{json.dumps(self.config, indent=2)}")

        # Initialize models, optimizers and load state dicts (if possible)
        self._initialize_models()

        # Initialize dataloaders
        logger.info("Initializing dataloaders...")
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        self._initialize_dataloaders(collate_fn)

        # Scheduler needs to be initialized after dataloaders since we need more info from dataset
        if self.action == "training":
            self._initialize_scheduler()

        # Copy config
        if self.save_dir is not None:
            copy_from = os.path.realpath(config_path)
            copy_to = os.path.realpath(
                os.path.join(self.save_dir, "config.yaml"))
            try:
                copy(copy_from, copy_to)
            except SameFileError:
                pass

        # Set additional attributes
        self._set_epoch(self.start_epoch - 1)  # training not yet started
        self.config["trainer"] = self
        self._best_acc = -float("inf")
        self._no_improve = 0
        self._stop = False
        self._is_best = False

    @from_config(requires_all=True)
    def _get_save_dir(self, work_dir, resume_from):
        # Get save directory
        if self.action == "training":
            if resume_from is None:
                if work_dir is not None:
                    curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_dir = os.path.join(work_dir, curr_time)
                    os.makedirs(save_dir, exist_ok=True)
                else:
                    save_dir = None
            else:
                save_dir = os.path.realpath(resume_from)
                assert os.path.exists(save_dir)
        elif self.action == "evaluation":
            save_dir = None
        else:
            raise ValueError(f"Unrecognized action: {self.action}")

        self.config["training"]["save_dir"] = self.save_dir = save_dir

    def _get_logger(self):
        # Get logger
        logger.remove()  # remove default handler
        logger.add(
            sys.stderr, colorize=True,
            format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {message}")
        if self.save_dir is not None:
            logger_path = os.path.join(self.save_dir, "training.log")
            logger.add(logger_path, mode="a",
                       format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")
            logger.info(f"Working directory: {self.save_dir}")
        self.logger = logger

    @from_config(requires_all=True)
    def _initialize_models(self, learning_rate, weight_decay, load_from, resume_from, device):
        """Initialize models and optimizer(s), and load state dictionaries, if
        possible."""
        # Get model class
        model_class = self.config["model"].get("model_class", None)
        if model_class is None:
            model_class = "BertForAddressExtractionWithTwoSeparateHeads"  # default model class
        model_init = model_classes[model_class]
        # Initialize backbone model
        logger.info("Initializing model...")
        from_pretrained = load_from is not None or resume_from is not None
        self.device = torch.device(device)
        self.model = model_init(self.config, from_pretrained=from_pretrained).to(self.device)
        self.tokenizer = self.model.tokenizer

        # Initialize optimizer
        if isinstance(learning_rate, str):
            learning_rate = eval(learning_rate)
        self.optimizer = AdamW(
            [params for params in self.model.parameters()
             if params.requires_grad],
            lr=learning_rate, weight_decay=weight_decay)

        # Load from a pretrained model
        self.start_epoch = 0

        if resume_from is not None:
            # Ensure that the two configs match (with some exclusions)
            with open(os.path.join(self.save_dir, "config.yaml"), "r") as conf:
                resume_config = yaml.load(conf, Loader=yaml.FullLoader)

            # Load the most recent saved model
            model_list = Path(self.save_dir).glob("checkpoint*.pth")
            resume_from = max(
                model_list, key=os.path.getctime)  # last saved model
            logger.info(f"Loading most recent saved model at {resume_from}")
            # Get some more info for resuming training
            _, last_name = os.path.split(resume_from)
            last_name, _ = os.path.splitext(last_name)
            self.start_epoch = int(last_name.split("_")[-1]) + 1

            compare_config = ConfigComparer(self.config, resume_config)
            compare_config.compare()
        if load_from is not None:
            logger.info(f"Loading pretrained model from {load_from}")

        if from_pretrained:
            load_from_path = resume_from if resume_from is not None else load_from

            checkpoint = torch.load(load_from_path, map_location=self.device)
            self.model.load_state_dict(
                checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.resume_from = resume_from
        self.load_from = load_from

    def _initialize_dataloaders(self, collate_fn):
        self.dataloaders = {}
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]
        batch_size_multiplier = self.config["training"].get(
            "batch_size_multiplier", 1.0)

        if self.action == "training":
            for set_name, set_info in self.config["data"].items():
                if set_name not in ["train", "val", "test"]:
                    continue

                if set_name == "train":
                    shuffle = True
                    bs = batch_size
                    p_augmentation = set_info["p_augmentation"]
                else:
                    shuffle = False if set_name == "test" else True
                    bs = round(batch_size * batch_size_multiplier)
                    p_augmentation = 0.0

                dataset = CustomDataset(
                    self.config, tokenizer=self.tokenizer,
                    paths=set_info["paths"], p_augmentation=p_augmentation)
                self.dataloaders[set_name] = DataLoader(
                    dataset, batch_size=bs, shuffle=shuffle,
                    collate_fn=collate_fn, num_workers=num_workers)

        elif self.action == "evaluation":
            if self.config["data_path"] is None and not ("val" in self.config["data"]):
                raise ValueError("Either argument `data_path` or `val` value in the config file must be specified.")

            if self.config["data_path"] is None:
                data_path = self.config["data"]["val"]["paths"]
            else:
                data_path = self.config["data_path"]
            dataset = CustomDataset(
                self.config, tokenizer=self.tokenizer, paths=data_path, p_augmentation=0.0)
            self.dataloaders["eval"] = DataLoader(
                dataset, batch_size=round(batch_size * batch_size_multiplier),
                shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

        else:
            raise ValueError(f"Unrecognized action: {self.action}")

    @from_config(requires_all=True)
    def _initialize_scheduler(self, lr_warmup):
        # Initialize scheduler
        updates_total = (self.config["training"]["num_epochs"] - self.start_epoch + 1) * len(self.dataloaders["train"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=round(lr_warmup * updates_total), num_training_steps=updates_total)

    def _save_models(self, filename=None):
        # Save model
        if self.save_dir is not None:
            if filename is None:
                filename = f"checkpoint_{self.epoch}.pth"
            save_path = os.path.join(self.save_dir, filename)
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}.")

    def _set_epoch(self, epoch):
        self.epoch = self.config["epoch"] = epoch

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, num_epochs, max_grad_norm=None,
                        debugging=False, post_processing=False):
        """Train the model for one epoch."""
        model.train()
        timer = Timer()

        print(
            ("{:25}" + "|" + "{:^45}" + "|" + "{:^45}" + "|").format("", "POI", "street")
        )
        print(
            ("{:25}" + "|" + "{:^15}" * 3 + "|" + "{:^15}" * 3 + "|").format(
                "", "span_loss", "existence_loss", "acc", "span_loss", "existence_loss", "acc")
        )

        total = 10 if debugging else len(dataloader)
        with tqdm(dataloader, total=total) as t:
            if num_epochs is not None:
                description = f"Training ({self.epoch}/{num_epochs})"
            else:
                description = "Training"
            t.set_description(description)

            for i, data in enumerate(t):
                timer.start()

                data = to_device(data, self.device)
                optimizer.zero_grad()

                # Forward
                output = model(**data)
                losses = output["losses"]

                # Calculate batch accuracy
                acc = compute_metrics_from_inputs_and_outputs(
                    inputs=data, outputs=output, tokenizer=self.tokenizer, save_csv_path=None,
                    confidence_threshold=self.config["evaluation"]["confidence_threshold"],
                    post_processing=post_processing)
                losses.update(acc)

                # Update tqdm with training information
                to_tqdm = []  # update tqdm
                for name in ["poi", "street"]:
                    for loss_type in ["span_loss", "existence_loss", "acc"]:
                        n = f"{name}_{loss_type}"
                        loss_n = losses[n]

                        if torch.isnan(loss_n):
                            to_tqdm.append("nan")
                        else:
                            to_tqdm.append(f"{loss_n.item():.3f}")

                des = (
                    "{:25}" + "|" + "{:^15}" * 3 + "|" + "{:^15}" * 3 + "|"
                ).format(description, *to_tqdm)
                t.set_description(des)

                # Backward
                losses["total_loss"].backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                timer.end()

                # Break when reaching 10 iterations when debugging
                if debugging and i == 9:
                    break

        logger.info(f"{description} took {timer.get_total_time():.2f}s.")
        return

    def evaluate_one_epoch(self, model, dataloader, prefix, debugging=False, save_csv_path=None,
                           post_processing=False, show_progress=False):
        """Evaluate the model for one epoch."""
        model.eval()
        tot_inp, tot_outp = [], []

        with torch.no_grad():
            total = 10 if debugging else len(dataloader)
            with tqdm(dataloader, total=total) as t:
                t.set_description(prefix)

                for i, data in enumerate(t):
                    # Input
                    data = to_device(data, self.device)
                    tot_inp.append(data)

                    # Forward
                    output = model(**data)
                    tot_outp.append(output)

                    # Break when reaching 10 iterations when debugging
                    if debugging and i == 9:
                        break

        acc = compute_metrics_from_inputs_and_outputs(
            inputs=tot_inp, outputs=tot_outp, tokenizer=self.tokenizer, save_csv_path=save_csv_path,
            confidence_threshold=self.config["evaluation"]["confidence_threshold"], post_processing=post_processing,
            show_progress=show_progress)

        if acc is not None:
            self._record_metrics(acc)

            to_log = [f"{k}: {v.item():.3f}" for k, v in acc.items()]
            logger.info(f"{prefix}: {', '.join(to_log)}")

        model.train()
        return

    def _record_metrics(self, acc):
        total_acc = acc["total_acc"]

        if self._best_acc < total_acc:
            self._best_acc = total_acc
            self._is_best = True
            self._no_improve = 0
        else:
            self._is_best = False
            self._no_improve += 1

        early_stopping = self.config["training"]["early_stopping"]
        self._stop = (self._no_improve > early_stopping) if early_stopping is not None else False

    @from_config(main_args="training", requires_all=True)
    def _train(self, num_epochs, debugging=False, max_grad_norm=None, post_processing=False):

        if self.load_from is not None or self.resume_from is not None:
            self.evaluate_one_epoch(
                self.model, self.dataloaders["val"], debugging=debugging,
                prefix="Validation (before training)")

        # Start training and evaluating
        for epoch in range(self.start_epoch, num_epochs):
            self._set_epoch(epoch)

            # Train
            self.train_one_epoch(
                self.model, self.dataloaders["train"], self.optimizer, self.scheduler, num_epochs=num_epochs,
                max_grad_norm=max_grad_norm, debugging=debugging, post_processing=post_processing)

            # Evaluate
            self.evaluate_one_epoch(
                self.model, self.dataloaders["val"], debugging=debugging, post_processing=post_processing,
                prefix=f"Validation (epoch: {epoch}/{num_epochs})")

            # Checkpoint
            self._save_models()

            # Best model
            if self._is_best:
                self._save_models(filename="checkpoint_best.pth")

            # Early stopping
            if self._stop:
                early_stopping = self.config["training"]["early_stopping"]
                logger.info(f"Model not improved over {early_stopping} "
                            f"epochs. Stopping training...")
                break

        # Test
        self.evaluate_one_epoch(
            self.model, self.dataloaders["test"], debugging=False, prefix="Test", post_processing=post_processing)
        logger.info("Training finished.")

    def train(self):
        return self._train(self.config)

    def eval(self):
        assert self.action == "evaluation"
        return self.evaluate_one_epoch(
            self.model, self.dataloaders["eval"], prefix="Evaluation",
            post_processing=self.config["evaluation"]["post_processing"], show_progress=True,
            debugging=False, save_csv_path=self.config["save_csv_path"])
