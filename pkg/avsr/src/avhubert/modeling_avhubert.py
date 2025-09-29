import logging
from dataclasses import dataclass
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerationConfig, GenerationMode
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.hubert.modeling_hubert import HubertEncoder, HubertEncoderStableLayerNorm
from transformers.utils import ModelOutput

from .configuration_avhubert import AVHubertConfig
from .configuration_resnet import ResEncoderConfig
from .decoder import AVHubertDecoder, AVHubertDecoderStableLayerNorm
from .modeling_resnet import ResEncoder

logger = logging.getLogger(__name__)

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def compute_mask_indices(
    shape: tuple[int, int],
    padding_mask: torch.Tensor | None,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length) + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    batch_indexes, starts, ends = [], [], []
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
        vals, run_starts, run_lengths = find_runs(mask[i])
        start_indices, lengths = run_starts[vals], run_lengths[vals]
        starts.append(start_indices)
        ends.append(start_indices + lengths)
        batch_indexes.append(np.zeros([len(start_indices)]) + i)
    return (
        mask,
        np.concatenate(starts).astype(np.int64),
        np.concatenate(ends).astype(np.int64),
        np.concatenate(batch_indexes).astype(np.int64),
    )


@dataclass
class AVHubertOutput:
    last_hidden_state: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    mask_indices: Optional[torch.Tensor] = None


class GradMultiply(torch.autograd.Function):
    """https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/grad_multiply.py"""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(AudioFeatureExtractor, self).__init__()
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, T, F]
        return einops.rearrange(x, "b t f -> b f t")  # [B, F, T]


class VideoFeatureExtractor(nn.Module):
    def __init__(self, config: ResEncoderConfig, output_dim: int) -> None:
        super(VideoFeatureExtractor, self).__init__()
        self.resnet = ResEncoder(config=config)
        self.proj = nn.Linear(
            in_features=self.resnet.backend_out,
            out_features=output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(einops.rearrange(x, "b t c h w -> b c t h w"))  # [B, F, T]
        x = self.proj(einops.rearrange(x, "b f t -> b t f"))  # [B, T, F]
        return einops.rearrange(x, "b t f -> b f t")  # [B, F, T]


class AVHubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AVHubertConfig
    base_model_prefix = "avhubert"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if is_deepspeed_zero3_enabled():
                import deepspeed

                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                if hasattr(module, "parametrizations"):
                    nn.init.kaiming_normal_(module.parametrizations.weight.original0.data)
                    nn.init.kaiming_normal_(module.parametrizations.weight.original1.data)
                nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


class AVHubertModel(AVHubertPreTrainedModel):
    def __init__(self, config: AVHubertConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.feat2tar_ratio = config.label_rate / config.sample_rate

        # feature extractor
        resnet_config = ResEncoderConfig(relu_type=config.resnet_relu_type)
        self.feature_extractor_audio = AudioFeatureExtractor(
            input_dim=config.audio_feat_dim,
            output_dim=config.encoder_embed_dim,
        )
        self.feature_extractor_video = VideoFeatureExtractor(config=resnet_config, output_dim=config.encoder_embed_dim)

        # modalities
        self.modality_dropout, self.audio_dropout = (
            config.modality_dropout,
            config.audio_dropout,
        )
        self.encoder_embed_dim = config.encoder_embed_dim
        if config.modality_fuse == "concat":
            embed = config.encoder_embed_dim * 2
        elif config.modality_fuse == "add":
            embed = config.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(embed, config.encoder_embed_dim) if embed != config.encoder_embed_dim else None
        )

        # feature mask
        self.mask_prob_image, self.mask_prob_audio = (
            config.mask_prob_image,
            config.mask_prob_audio,
        )
        self.mask_length_image, self.mask_length_audio = (
            config.mask_length_image,
            config.mask_length_audio,
        )

        # dropout
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(config.audio_feat_dim).uniform_()
            if config.masking_type == "input"
            else torch.FloatTensor(config.encoder_embed_dim).uniform_()
        )

        # transformer encoder
        transformer_config = config.encoder_config
        if transformer_config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config=transformer_config)
        else:
            self.encoder = HubertEncoder(config=transformer_config)
        self.layer_norm = nn.LayerNorm(embed)

    def forward_features(
        self,
        x: torch.Tensor,
        extractor: AudioFeatureExtractor | VideoFeatureExtractor,
    ) -> torch.Tensor:
        if self.config.feature_grad_mult > 0:
            features = extractor(x)
            if self.config.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.config.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(x)
        return features

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def apply_input_mask(self, x, padding_mask):
        B, T, C = x.shape[:3]  # [B, T, C, H, W] or [B, T, C]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:
            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)

            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                if mask_indices.any():
                    x[mask_indices] = self.mask_emb
            elif self.config.selection_type == "same_other_seq":
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.config.selection_type == "same_seq":
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end - start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start - length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T - 1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64) + batch_index)
                batch_indexes, other_indexes = (
                    np.concatenate(batch_indexes_),
                    np.concatenate(other_indexes),
                )
                x[mask_indices] = x[batch_indexes, other_indexes]

        else:
            mask_indices = None

        if self.config.mask_channel_prob > 0:
            logger.info("No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, (
            "masking prob/length for image/audio be same for feature masking"
        )
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb.to(x.dtype)
        else:
            mask_indices = None

        if self.config.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.config.mask_channel_prob,
                self.config.mask_channel_length,
                self.config.mask_channel_selection,
                self.config.mask_channel_other,
                no_overlap=self.config.no_mask_channel_overlap,
                min_space=self.config.mask_channel_min_space,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_features: bool = False,
        output_mask_indices: bool = False,
    ) -> ModelOutput:
        if mask and self.config.masking_type == "input" and self.training:  # input-level mask
            pixel_values, mask_indices_video = self.apply_input_mask(pixel_values, padding_mask)
            input_values, mask_indices_audio = self.apply_input_mask(input_values, padding_mask)
            if mask_indices_audio is None and mask_indices_video is None:
                raise ValueError
            elif mask_indices_audio is None:
                mask_indices_audio = torch.zeros_like(mask_indices_video).bool()
            else:
                mask_indices_video = torch.zeros_like(mask_indices_audio).bool()
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            input_values, pixel_values, mask_indices = input_values, pixel_values, None

        if input_values is not None and pixel_values is None:
            features_audio = self.forward_features(input_values, self.feature_extractor_audio)  # [B, F, T]
            features_video = torch.zeros_like(features_audio)  # [B, F, T]
        elif input_values is None and pixel_values is not None:
            features_video = self.forward_features(pixel_values, self.feature_extractor_video)  # [B, F, T]
            features_audio = torch.zeros_like(features_video)  # [B, F, T]
        elif input_values is not None and pixel_values is not None:
            features_audio = self.forward_features(input_values, self.feature_extractor_audio)  # [B, F, T]
            features_video = self.forward_features(pixel_values, self.feature_extractor_video)  # [B, F, T]
        else:
            raise ValueError("Either `input_values` or `pixel_values` must be passed")

        # dropout
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                else:
                    features_video = 0 * features_video
        # fuse modality
        if self.config.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.config.modality_fuse == "add":
            features = features_audio + features_video

        features_raw = features.clone()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        else:
            padding_mask = torch.zeros(features.size()[:2], dtype=torch.bool, device=features.device)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if mask and self.config.masking_type == "feature" and self.training:  # feature-level mask
            x, mask_indices = self.apply_feature_mask(features, padding_mask)
        else:
            x = features

        # transformer encoder
        encoder_out = self.encoder(
            hidden_states=x,
            attention_mask=~padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return AVHubertOutput(
            last_hidden_state=encoder_out.last_hidden_state,
            padding_mask=padding_mask,
            hidden_states=encoder_out.hidden_states,
            attentions=encoder_out.attentions,
            features=features_raw if output_features else None,
            mask_indices=mask_indices if output_mask_indices else None,
        )


class AVHubertForConditionalGeneration(AVHubertPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: AVHubertConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.config = config
        if config.apply_mask is None:
            config.apply_mask = False

        self.avhubert = AVHubertModel(config=config)
        if config.freeze_base_model:
            self.freeze_base_model()
        if config.freeze_feature_encoder:
            self.freeze_feature_encoder()
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `AVHubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim, padding_idx=config.pad_token_id)
        transformer_config = config.decoder_config
        if transformer_config.do_stable_layer_norm:
            self.decoder = AVHubertDecoderStableLayerNorm(config=transformer_config)
        else:
            self.decoder = AVHubertDecoder(config=transformer_config)

        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        if config.share_decoder_input_output_embed:
            # If this model shares lm head weights with the token embeddings,
            # you can access lm head weights that is the same as the token embeddings but
            # the token embeddings are directly refered to instead of lm heads when training!
            self.lm_head.weight = self.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0, std=config.decoder_embed_dim**-0.5)

        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        for param in self.avhubert.feature_extractor_audio.parameters():
            param.requires_grad = False
        for param in self.avhubert.feature_extractor_video.parameters():
            param.requires_grad = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.avhubert.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> ModelOutput:
        encoder_outs = self.avhubert(
            input_values=input_values,
            pixel_values=pixel_values,
            padding_mask=padding_mask,
            mask=self.config.apply_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        embed_tokens = self.embed_tokens(decoder_input_ids)
        encoder_attention_mask = (~padding_mask.bool()).long()
        hidden_states = self.decoder(
            inputs_embeds=embed_tokens,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.config.share_decoder_input_output_embed:
            logits = F.linear(hidden_states.last_hidden_state, weight=self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            decoder_hidden_states=hidden_states.hidden_states,
            decoder_attentions=hidden_states.attentions,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outs.last_hidden_state,
            encoder_hidden_states=encoder_outs.hidden_states,
            encoder_attentions=encoder_outs.attentions,
        )

    def _get_generation_mode(
        self,
        generation_config: GenerationConfig,
        assistant_model: PreTrainedModel | None,
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None or generation_config.prompt_lookup_num_tokens is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor = None,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
            decoder_attention_mask = torch.ones_like(input_ids)
        return {
            "input_values": input_values,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "padding_mask": padding_mask,
        }
