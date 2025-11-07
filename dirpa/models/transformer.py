from typing import Any, Literal, cast

import torch
import torch.nn as nn
from eurocropsml.dataset.base import DataItem

from dirpa.models.base import Model, ModelBuilder, ModelConfig
from dirpa.models.positional_encoding import PositionalEncoding


class TransformerBackbone(nn.Module):
    """Transformer Backbone Encoder with multi-headed self-attention.

    Args:
        in_channels: Number of input channels.
        d_model: Number of input features.
            Input dimension of query, key, and value's linear layers.
        encoder_layer: Instance of TransformerEncoderLayer class.
            It consists of self-attention and a feedforward network.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        t: Period to use for positional encoding.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_enc_len: int,
        t: int = 1000,
    ):
        self.d_model = d_model

        super().__init__()
        self._in_layernorm = nn.LayerNorm(in_channels)
        self._in_linear = torch.nn.Linear(in_channels, d_model)
        self._pos_encoding = PositionalEncoding(d_model, pos_enc_len, t)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute model (backbone) forward pass."""
        x = x.float()

        x = self._in_layernorm(x)
        x = self._in_linear(x)
        enc_output: torch.Tensor = self._pos_encoding(x, kwargs.get("dates"))
        # TODO: double-check if thi is correct
        if (mask := kwargs.get("mask")) is not None and mask.dim() == x.dim():
            # masking of full time steps
            mask = mask.any(-1)
        encoder_out = self._encoder(enc_output, src_key_padding_mask=mask)

        return cast(torch.Tensor, encoder_out)


class TransformerConfig(ModelConfig):
    """Config for transformer model.

    Args:
        n_heads: Number of heads in the multi-head attention models.
        in_channels: Number of input channels.
        d_model: Number of input features. Input dimension of query, key, and value's linear layers.
        dim_fc: Dimensionality of query, key, and value used as input to the multi-head attention.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
    """

    n_heads: int
    in_channels: int
    d_model: int
    dim_fc: int
    num_layers: int
    pos_enc_len: int

    model_builder: Literal["TransformerModelBuilder"] = "TransformerModelBuilder"


class TransformerModel(Model):
    """Model architecture for pre-training and fine-tuning with transformers."""

    def forward(self, ipt: DataItem) -> torch.Tensor:
        out = self.backbone(ipt.data, **ipt.meta_data)
        match self.head:
            case nn.Linear():
                return cast(torch.Tensor, self.head(out.mean(1)))
            case _:
                raise NotImplementedError


class TransformerModelBuilder(ModelBuilder):
    """Transformer with multi-headed self-attention.

    Args:
        config: Transformer model config.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_fc,
            batch_first=True,
        )

    def build_backbone(self) -> TransformerBackbone:
        return TransformerBackbone(
            in_channels=self.config.in_channels,
            d_model=self.config.d_model,
            encoder_layer=self._encoder_layer,
            num_layers=self.config.num_layers,
            pos_enc_len=self.config.pos_enc_len,
        )

    def build_classification_head(self, num_classes: int) -> nn.Linear:
        return nn.Linear(self.config.d_model, num_classes)

    def build_classification_model(
        self, num_classes: int, device: torch.device
    ) -> Model:
        backbone = self.build_backbone()
        head = self.build_classification_head(num_classes)
        return TransformerModel(backbone=backbone, head=head, device=device)
