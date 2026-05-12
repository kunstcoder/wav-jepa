"""Compact WavJEPA-style model components."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wav_jepa_minimal.config import WAVJEPA_CONV_LAYERS_SPEC, MaskConfig, WavJepaConfig
from wav_jepa_minimal.masking import sample_context_target_masks


class ConvPatchEncoder(nn.Module):
    """Raw waveform encoder using the upstream WavJEPA convolution spec."""

    def __init__(
        self,
        in_channels: int,
        conv_layers_spec: tuple[tuple[int, int, int], ...] = WAVJEPA_CONV_LAYERS_SPEC,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for index, (out_channels, kernel, stride) in enumerate(conv_layers_spec):
            layers.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    bias=conv_bias,
                )
            )
            if index == 0:
                layers.append(nn.GroupNorm(out_channels, out_channels, affine=True))
            layers.append(nn.GELU())
            current_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.embedding_dim = conv_layers_spec[-1][0]

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(
                f"Expected audio shape [batch, samples] or [batch, channels, samples], "
                f"got {tuple(audio.shape)}"
            )
        features = self.network(audio)
        return features.transpose(1, 2)


class TransformerStack(nn.Module):
    """Small batch-first transformer encoder with learnable positional embeddings."""

    def __init__(self, config: WavJepaConfig, max_patches: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, max_patches, config.encoder_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.encoder_dim * config.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
            layer_norm_eps=1e-6,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.norm = nn.LayerNorm(config.encoder_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.position[:, : tokens.size(1)]
        return self.norm(self.encoder(tokens))


class Predictor(nn.Module):
    """Predict target latent vectors from context latents."""

    def __init__(self, config: WavJepaConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.encoder_dim, config.predictor_dim),
            nn.GELU(),
            nn.LayerNorm(config.predictor_dim),
            nn.Linear(config.predictor_dim, config.encoder_dim),
        )

    def forward(self, context_tokens: Tensor) -> Tensor:
        return self.network(context_tokens)


class WavJepaModel(nn.Module):
    """Minimal JEPA learner for raw waveforms."""

    def __init__(self, config: WavJepaConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = ConvPatchEncoder(config.in_channels)
        self.feature_norm = nn.LayerNorm(self.feature_extractor.embedding_dim)
        if self.feature_extractor.embedding_dim != config.encoder_dim:
            self.feature_to_encoder = nn.Linear(
                self.feature_extractor.embedding_dim, config.encoder_dim
            )
        else:
            self.feature_to_encoder = nn.Identity()
        max_patches = self._infer_patch_count(config)
        self.context_encoder = TransformerStack(config, max_patches=max_patches)
        self.target_encoder = TransformerStack(config, max_patches=max_patches)
        self.predictor = Predictor(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.encoder_dim))
        self.mask_config = MaskConfig(
            context_mask_prob=config.context_mask_prob,
            context_mask_length=config.context_mask_length,
            target_prob=config.target_prob,
            target_length=config.target_length,
            target_masks_per_context=config.target_masks_per_context,
            ratio_cutoff=config.ratio_cutoff,
        )
        self._copy_context_to_target()

    def _encode_features(self, audio: Tensor) -> Tensor:
        tokens = self.feature_extractor(audio)
        tokens = self.feature_norm(tokens)
        return self.feature_to_encoder(tokens)

    def _infer_patch_count(self, config: WavJepaConfig) -> int:
        samples = int(config.sample_rate * config.process_seconds)
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, samples)
            return self._encode_features(dummy).size(1)

    def _copy_context_to_target(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, step: int | None = None) -> None:
        """Apply EMA update from context encoder to target encoder."""

        decay = self._ema_decay(step)
        for context, target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target.data.mul_(decay).add_(context.data, alpha=1.0 - decay)

    def _ema_decay(self, step: int | None = None) -> float:
        if step is None or step >= self.config.ema_anneal_end_step:
            return self.config.ema_end_decay
        delta = self.config.ema_end_decay - self.config.ema_decay
        progress = step / self.config.ema_anneal_end_step
        return self.config.ema_decay + delta * progress

    def forward_loss(self, audio: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Compute masked latent prediction loss for one batch."""

        tokens = self._encode_features(audio)
        batch_size, patch_count, _ = tokens.shape
        context_mask, target_mask = sample_context_target_masks(
            batch_size=batch_size,
            length=patch_count,
            config=self.mask_config,
            device=tokens.device,
        )

        masked_tokens = torch.where(context_mask.unsqueeze(-1), tokens, self.mask_token)
        context_latents = self.context_encoder(masked_tokens)
        predictions = self.predictor(context_latents)

        with torch.no_grad():
            target_latents = self.target_encoder(tokens)

        selected_predictions = predictions[target_mask]
        selected_targets = target_latents[target_mask]
        loss = torch.nn.functional.smooth_l1_loss(selected_predictions, selected_targets)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "target_fraction": float(target_mask.float().mean().detach().cpu()),
            "context_fraction": float(context_mask.float().mean().detach().cpu()),
        }
        return loss, metrics

    @torch.no_grad()
    def embed(self, audio: Tensor) -> Tensor:
        """Return utterance-level embeddings for KNN or downstream probes."""

        tokens = self._encode_features(audio)
        latents = self.context_encoder(tokens)
        return latents.mean(dim=1)
