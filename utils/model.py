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
class BertForAddressExtractionWithTwoSeparateHeads(nn.Module):
    """Bert model for for Shopee Code League 2021 - Address Elements Extraction task.
    Model structure:
                                            |--- span_start
                                        |---|
                         |---- POI -----|   |--- span_end
                         |              |
                         |              |--- existence
    feature_extractor ---|
                         |                  |--- span_start
                         |              |---|
                         |              |   |--- span_end
                         |---- street --|
                                        |--- existence

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
        Loss weights. Final loss will be computed as: `lambda[0] * poi_span_loss + lambda[1] * poi_existence +
        lambda[2] * street_span_loss + lambda[3] * street_existence `
    """
    @from_config(main_args="model", requires_all=True)
    def __init__(self, model_name_or_path, config_name=None, tokenizer_name=None, cache_dir=None,
                 from_pretrained=False, freeze_base_model=False, fusion="max_pooling", lambdas=[1, 1, 1, 1]):
        super(BertForAddressExtractionWithTwoSeparateHeads, self).__init__()
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

        # Additional layers
        self._initialize_layers()

        # Fusion
        if fusion not in ["max_pooling", "average_pooling", "sum"]:
            raise ValueError(f"Invalid fusion value. Expected one of ['max_pooling', 'average_pooling', 'sum'], got "
                             f"'{fusion}' instead.")
        self.fusion = fusion

        assert len(lambdas) == 4
        self.lambdas = lambdas

        # Freeze
        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False

    def _initialize_layers(self):
        self.poi_span_classifier = nn.Linear(self.base_model_config.hidden_size, 2)
        self.poi_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.street_span_classifier = nn.Linear(self.base_model_config.hidden_size, 2)
        self.street_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

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

    def _get_predictions(self, hidden_states, attention_mask, poi_start, street_start):
        # POI
        poi_span_preds = self.poi_span_classifier(hidden_states)  # (B, L, 2)
        poi_existence_preds = self.poi_existence(hidden_states).squeeze(-1)  # (B, L)
        poi_existence_preds = self.fusion_layer(poi_existence_preds, attention_mask, dim=1)  # (B,)
        poi_existence_preds = self.sigmoid(poi_existence_preds)  # turn into probabilities

        # Street
        street_span_preds = self.street_span_classifier(hidden_states)  # (B, L, 2)
        street_existence_preds = self.street_existence(hidden_states).squeeze(-1)  # (B, L)
        street_existence_preds = self.fusion_layer(street_existence_preds, attention_mask, dim=1)  # (B,)
        street_existence_preds = self.sigmoid(street_existence_preds)  # turn into probabilities

        return poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds

    def _compute_losses(
        self,
        # Predictions
        poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds,
        # Groundtruths
        poi_start, poi_end, has_poi, street_start, street_end, has_street,
    ):
        """Compute losses (including total loss) given loss weights"""

        # POI span loss
        poi_span_gt = torch.stack([poi_start, poi_end], dim=-1)  # (B, 2)
        if has_poi.any():
            poi_span_loss = F.cross_entropy(poi_span_preds[has_poi], poi_span_gt[has_poi])
        else:
            poi_span_loss = 0.0

        # POI existence loss
        poi_existence_loss = F.binary_cross_entropy(poi_existence_preds, has_poi.float())

        # Street span loss
        street_span_gt = torch.stack([street_start, street_end], dim=-1)  # (B, 2)
        if has_street.any():
            street_span_loss = F.cross_entropy(street_span_preds[has_street], street_span_gt[has_street])
        else:
            street_span_loss = 0.0

        # Street existence loss
        street_existence_loss = F.binary_cross_entropy(street_existence_preds, has_street.float())

        # Total loss
        total_loss = 0
        for weight, loss in \
                zip(self.lambdas, [poi_span_loss, poi_existence_loss, street_span_loss, street_existence_loss]):
            total_loss += weight * loss

        return {"total_loss": total_loss,
                "poi_span_loss": poi_span_loss, "poi_existence_loss": poi_existence_loss,
                "street_span_loss": street_span_loss, "street_existence_loss": street_existence_loss}

    def forward(self, input_ids, attention_mask=None, poi_start=None, poi_end=None, street_start=None, street_end=None,
                has_poi=None, has_street=None, **kwargs):
        """Forward logic.

        input_ids : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Indices of input sequence tokens in the vocabulary (i.e.,
            encoded).
        attention_mask : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``.
        poi_start : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of start indices of POI (person of interest) in the input
            sequence. `-1` if not present.
        poi_end : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of end indices (**inclusive**) of POI (person of
            interest) in the input sequence. `-1` if not present.
        street_start : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of start indices of street in the input sequence. `-1` if
            not present.
        street_end : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of end indices (**inclusive**) of street in the input
            sequence. `-1` if not present.
        has_poi : torch.Tensor (optional)
            Tensor of shape (batch_size,). Mask to indicate whether there is POI info in a input sequence. This can
            also be obtained by masking `poi_start != -1`.
        has_street : torch.Tensor (optional)
            Tensor of shape (batch_size,). Mask to indicate whether there is street info in a input sequence. This can
            also be obtained by masking `street_start != -1`.
        """
        # Base forward (feature extractor)
        hidden_states = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # (B, L, H)
        attention_mask = attention_mask.bool()

        poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds = self._get_predictions(
            hidden_states, attention_mask, poi_start, street_start)

        # Need to mask predictions with attention mask so that padding does not affect the probabilities
        epsilon = torch.tensor(1e-16).to(poi_span_preds)
        poi_span_preds = torch.where(attention_mask.unsqueeze(-1), poi_span_preds, epsilon)  # (B, L, 2)
        street_span_preds = torch.where(attention_mask.unsqueeze(-1), street_span_preds, epsilon)  # (B, L, 2)

        outp = {"poi_span_preds": poi_span_preds,  # (B, L, 2)
                "poi_existence_preds": poi_existence_preds,  # (B,)
                "street_span_preds": street_span_preds,  # (B, L, 2)
                "street_existence_preds": street_existence_preds,  # (B,)
                "attention_mask": attention_mask,  # (B, L)
                }

        # Get loss if training (i.e., some tensors are not provided)
        if poi_start is not None:
            # Get mask
            if has_poi is None:
                has_poi = (poi_start != -1)
            if has_street is None:
                has_street = (street_start != -1)

            # Compute loss
            losses = self._compute_losses(
                # Predictions
                poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds,
                # Groundtruths
                poi_start, poi_end, has_poi, street_start, street_end, has_street
            )
            outp["losses"] = losses

        return outp


@register_model
class BertForAddressExtractionWithTwoLinkedHeads(BertForAddressExtractionWithTwoSeparateHeads):
    """
    Model structure:


                                       |--- span_start ---|
                         |---- POI ----|                  |--- existence
                         |             |--- span_end -----|
    feature_extractor ---|
                         |             |--- span_start ---|
                         |-- street ---|                  |--- existence
                                       |--- span_end -----|
    """
    def _initialize_layers(self):
        self.poi_span_classifier = nn.Linear(self.base_model_config.hidden_size, 2)
        self.poi_existence = nn.Linear(2, 1)
        self.street_span_classifier = nn.Linear(self.base_model_config.hidden_size, 2)
        self.street_existence = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_predictions(self, hidden_states, attention_mask, poi_start, street_start):
        # POI
        poi_span_preds = self.poi_span_classifier(hidden_states)  # (B, L, 2)
        poi_existence_preds = self.poi_existence(poi_span_preds).squeeze(-1)  # (B, L)
        poi_existence_preds = self.fusion_layer(poi_existence_preds, attention_mask, dim=1)  # (B,)
        poi_existence_preds = self.sigmoid(poi_existence_preds)  # turn into probabilities

        # Street
        street_span_preds = self.street_span_classifier(hidden_states)  # (B, L, 2)
        street_existence_preds = self.street_existence(street_span_preds).squeeze(-1)  # (B, L)
        street_existence_preds = self.fusion_layer(street_existence_preds, attention_mask, dim=1)  # (B,)
        street_existence_preds = self.sigmoid(street_existence_preds)  # turn into probabilities

        return poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds


@register_model
class BertForAddressExtractionWithTwoDependentHeads(BertForAddressExtractionWithTwoSeparateHeads):
    """
    Model structure:

                                        |--- span_start --- span_end
                         |---- POI -----|
                         |              |--- existence
    feature_extractor ---|
                         |              |--- span_start --- span_end
                         |---- street --|
                                        |--- existence

    where each span_end prediction is conditioned on the corresponding span_start prediction.
    """
    def _initialize_layers(self):
        self.poi_span_start_classifier = nn.Linear(self.base_model_config.hidden_size, 1)
        self.poi_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.street_span_start_classifier = nn.Linear(self.base_model_config.hidden_size, 1)
        self.street_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_predictions(self, hidden_states, attention_mask, poi_start, street_start):
        # hidden_state: (B, L, H), where H is hidden size.
        # poi_start, poi_end: (B,)

        # POI
        poi_span_start_preds = self.poi_span_start_classifier(hidden_states).squeeze(-1)  # (B, L)
        poi_span_start_preds_idxs = poi_span_start_preds.argmax(-1)  # (B,)
        if self.training:
            poi_span_start_preds_idxs = poi_start  # (B,)
        # For each training example, take the hidden state of the span start prediction.
        poi_span_start_preds_states = hidden_states[
            list(range(len(hidden_states))),
            poi_span_start_preds_idxs]  # (B, H)
        poi_span_end_preds = torch.matmul(
            hidden_states, poi_span_start_preds_states.unsqueeze(-1)).squeeze(-1)  # (B, L)
        poi_span_preds = torch.stack([poi_span_start_preds, poi_span_end_preds], dim=-1)  # (B, L, 2)

        poi_existence_preds = self.poi_existence(hidden_states).squeeze(-1)  # (B, L)
        poi_existence_preds = self.fusion_layer(poi_existence_preds, attention_mask, dim=1)  # (B,)
        poi_existence_preds = self.sigmoid(poi_existence_preds)  # turn into probabilities

        # Street
        street_span_start_preds = self.street_span_start_classifier(hidden_states).squeeze(-1)  # (B, L)
        street_span_start_preds_idxs = street_span_start_preds.argmax(-1)  # (B,)
        if self.training:
            street_span_start_preds_idxs = street_start  # (B,)
        # For each training example, take the hidden state of the span start prediction.
        street_span_start_preds_states = hidden_states[
            list(range(len(hidden_states))),
            street_span_start_preds_idxs]  # (B, H)
        street_span_end_preds = torch.matmul(
            hidden_states, street_span_start_preds_states.unsqueeze(-1)).squeeze(-1)  # (B, L)
        street_span_preds = torch.stack([street_span_start_preds, street_span_end_preds], dim=-1)  # (B, L, 2)

        street_existence_preds = self.street_existence(hidden_states).squeeze(-1)  # (B, L)
        street_existence_preds = self.fusion_layer(street_existence_preds, attention_mask, dim=1)  # (B,)
        street_existence_preds = self.sigmoid(street_existence_preds)  # turn into probabilities

        return poi_span_preds, poi_existence_preds, street_span_preds, street_existence_preds
