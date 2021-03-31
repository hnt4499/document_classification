import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from .utils import from_config


model_classes = {}


def register_model(cls):
    model_classes[cls.__name__] = cls
    return cls


@register_model
class BertForDBpediaDocumentClassification(nn.Module):
    """Bert model for DBpedia classification task.

    Parameters
    ----------
    model_name_or_path : str
        Path to pretrained model or model identifier from huggingface.co/models.
    config_name : str
        Pretrained config name or path if not the same as model_name.
    tokenizer_name : str
        Pretrained tokenizer name or path if not the same as model_name.
    cache_dir : str
        Path to directory to store the pretrained models downloaded from huggingface.co.
    from_pretrained : bool
        Whether intializing model from pretrained model (other than the pretrained model from huggingface). If yes,
        avoid loading pretrained model from huggingface to save time.
    freeze_base_model : bool
        Whether to freeze the base BERT model.
    fusion : str
        One of ["max_pooling", "average_pooling", "sum"]. How the hidden states from each timestep will be fused
        together to produce a single vector used for binary classifiers (for exist/non-exist of POI/street).
        According to http://arxiv.org/abs/1909.07755, max pooling works best.
    lambdas : list[float]
        Loss weights. Final loss will be computed as: `lambda[0] * l1_loss + lambda[1] * l2_loss + lambda[2] * l3_loss`
    """
    @from_config(main_args="model", requires_all=True)
    def __init__(self, model_name_or_path, mapping_path, config_name=None, tokenizer_name=None, cache_dir=None,
                 from_pretrained=False, freeze_base_model=False, fusion="max_pooling", lambdas=[1, 1, 1]):
        super(BertForDBpediaDocumentClassification, self).__init__()
        # Initialize config, tokenizer and model (feature extractor)
        self.base_model_config = AutoConfig.from_pretrained(
            config_name if config_name is not None else model_name_or_path,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
        )
        if from_pretrained:
            self.base_model = AutoModel.from_config(config=self.base_model_config)
        else:
            self.base_model = AutoModel.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=self.base_model_config,
                cache_dir=cache_dir,
            )

        # Fusion
        if fusion not in ["max_pooling", "average_pooling", "sum"]:
            raise ValueError(f"Invalid fusion value. Expected one of ['max_pooling', 'average_pooling', 'sum'], got "
                             f"'{fusion}' instead.")
        self.fusion = fusion

        assert len(lambdas) == 3
        self.lambdas = lambdas

        # Freeze
        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # Intialize layers
        with open(mapping_path, "r") as fin:
            self.mapping = json.load(fin)
        self._initialize_layers()

    def _initialize_layers(self):
        self.l1_classifier = nn.Linear(self.base_model_config.hidden_size, len(self.mapping["l1"]["cls2idx"]))
        self.l2_classifier = nn.Linear(self.base_model_config.hidden_size, len(self.mapping["l2"]["cls2idx"]))
        self.l3_classifier = nn.Linear(self.base_model_config.hidden_size, len(self.mapping["l3"]["cls2idx"]))

    def fusion_layer(self, inp, mask, dim):
        """Fuse model predictions across the sequence length dimension"""
        # max pooling and sum can be handled easily
        if self.fusion in ["max_pooling", "sum"]:
            func = torch.max if self.fusion == "max_pooling" else torch.sum
            epsilon = torch.tensor(1e-16).to(inp)
            inp = torch.where(mask, inp, epsilon)
            inp = func(inp, dim=dim)
            if not isinstance(inp, torch.Tensor):
                inp = inp[0]
        # average pooling
        elif self.fusion == "average_pooling":
            assert inp.shape == mask.shape
            new_inp = []
            for inp_i, mask_i in zip(inp, mask):
                new_inp.append(inp_i[mask_i].mean())
            inp = torch.tensor(new_inp).to(inp)
        else:
            raise ValueError(f"Invalid fusion value. Expected one of ['max_pooling', 'average_pooling', 'sum'], got "
                             f"'{self.fusion}' instead.")

        return inp

    def _get_predictions(self, hidden_states, attention_mask):
        # Fuse
        hidden_states = self.fusion_layer(hidden_states, attention_mask.unsqueeze(-1), dim=1)  # (B, H)

        # Classify
        l1_cls_preds = self.l1_classifier(hidden_states)  # (B, C1)
        l2_cls_preds = self.l2_classifier(hidden_states)  # (B, C2)
        l3_cls_preds = self.l3_classifier(hidden_states)  # (B, C3)

        return l1_cls_preds, l2_cls_preds, l3_cls_preds

    def _compute_losses(
        self,
        # Predictions
        l1_cls_preds, l2_cls_preds, l3_cls_preds,
        # Groundtruths
        l1_cls_gt, l2_cls_gt, l3_cls_gt,
    ):
        """Compute losses (including total loss) given loss weights"""

        l1_cls_loss = F.cross_entropy(l1_cls_preds, l1_cls_gt)
        l2_cls_loss = F.cross_entropy(l2_cls_preds, l2_cls_gt)
        l3_cls_loss = F.cross_entropy(l3_cls_preds, l3_cls_gt)

        # Total loss
        total_loss = 0
        for weight, loss in \
                zip(self.lambdas, [l1_cls_loss, l2_cls_loss, l3_cls_loss]):
            total_loss += weight * loss

        return {"total_loss": total_loss,
                "l1_cls_loss": l1_cls_loss, "l2_cls_loss": l2_cls_loss, "l3_cls_loss": l3_cls_loss}

    def forward(self, input_ids, attention_mask=None, l1_cls_gt=None, l2_cls_gt=None, l3_cls_gt=None, **kwargs):
        """Forward logic.

        input_ids : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Indices of input sequence tokens in the vocabulary (i.e.,
            encoded).
        attention_mask : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``.
        """
        # Base forward (feature extractor)
        hidden_states = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # (B, L, H)
        attention_mask = attention_mask.bool()

        l1_cls_preds, l2_cls_preds, l3_cls_preds = self._get_predictions(hidden_states, attention_mask)

        outp = {
            "l1_cls_preds": l1_cls_preds,  # (B, C1)
            "l2_cls_preds": l2_cls_preds,  # (B, C2)
            "l3_cls_preds": l3_cls_preds,  # (B, C3)
        }

        # Get loss if training (i.e., some tensors are not provided)
        if l1_cls_gt is not None:
            # Compute loss
            losses = self._compute_losses(
                # Predictions
                l1_cls_preds, l2_cls_preds, l3_cls_preds,
                # Groundtruths
                l1_cls_gt, l2_cls_gt, l3_cls_gt,
            )
            outp["losses"] = losses

        return outp
