from typing import Literal

from transformers import PretrainedConfig, HubertConfig

EXTRACTOR_MODE_CHOICES = Literal["default", "layer_norm"]
MASKING_DISTRIBUTION_CHOICES = Literal["static", "uniform", "normal", "poisson"]


class AVHubertConfig(PretrainedConfig):
    model_type: str = "avhubert"

    def __init__(
        self,
        label_rate: int = 100,
        input_modality: str = "image",
        extractor_mode: EXTRACTOR_MODE_CHOICES = "default",
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        encoder_layerdrop: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        final_dropout: float = 0.0,
        final_dim: int = 0,
        untie_final_proj: bool = False,
        layer_norm_first: bool = False,
        conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        conv_bias: bool = False,
        logit_temp: float = 0.1,
        target_glu: bool = False,
        feature_grad_mult: float = 1.0,
        mask_length_audio: int = 10,
        mask_prob_audio: float = 0.65,
        mask_length_image: int = 10,
        mask_prob_image: float = 0.65,
        mask_selection: MASKING_DISTRIBUTION_CHOICES = "static",
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = "static",
        mask_channel_other: float = 0.0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        latent_temp: tuple[float, float, float] = (2, 0.5, 0.999995),
        skip_masked: bool = False,
        skip_nomask: bool = False,
        resnet_relu_type: str = "prelu",
        resnet_weights: str | None = None,
        sim_type: str = "cosine",
        sub_encoder_layers: int = 0,
        audio_feat_dim: int = 104,
        modality_dropout: float = 0.0,
        audio_dropout: float = 0.0,
        modality_fuse: str = "concat",
        selection_type: str = "same_other_seq",
        masking_type: str = "input",
        decoder_embed_dim: int = 768,
        decoder_ffn_embed_dim: int = 3072,
        decoder_layers: int = 6,
        decoder_layerdrop: float = 0.0,
        decoder_attention_heads: int = 4,
        decoder_learned_pos: bool = False,
        decoder_normalize_before: bool = False,
        no_token_positional_embeddings: bool = False,
        decoder_dropout: float = 0.1,
        decoder_attention_dropout: float = 0.1,
        decoder_activation_dropout: float = 0.0,
        max_target_positions: int = 2048,
        share_decoder_input_output_embed: bool = False,
        no_scale_embedding: bool = True,
        sample_rate: int = 25,
        num_labels: int = 100,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: list[float] | None = None,
        initializer_range: float = 0.02,
        do_stable_layer_norm: bool = False,
        apply_mask: bool | None = None,
        vocab_size: int | None = None,
        freeze_feature_encoder: bool = False,
        freeze_base_model: bool = False,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = False,
        ctc_loss_weight: float = 0.3,
        special_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_rate = label_rate
        self.input_modality = input_modality
        self.extractor_mode = extractor_mode
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        self.final_dropout = final_dropout
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        self.layer_norm_first = layer_norm_first
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_bias = conv_bias
        self.logit_temp = logit_temp
        self.target_glu = target_glu
        self.feature_grad_mult = feature_grad_mult
        self.mask_length_audio = mask_length_audio
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_image = mask_length_image
        self.mask_prob_image = mask_prob_image
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.latent_temp = latent_temp
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.resnet_relu_type = resnet_relu_type
        self.resnet_weights = resnet_weights
        self.sim_type = sim_type
        self.sub_encoder_layers = sub_encoder_layers
        self.audio_feat_dim = audio_feat_dim
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.modality_fuse = modality_fuse
        self.selection_type = selection_type
        self.masking_type = masking_type
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_learned_pos = decoder_learned_pos
        self.decoder_normalize_before = decoder_normalize_before
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.max_target_positions = max_target_positions
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.no_scale_embedding = no_scale_embedding
        self.sample_rate = sample_rate
        self.num_labels = num_labels
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.initializer_range = initializer_range
        self.do_stable_layer_norm = do_stable_layer_norm
        self.apply_mask = apply_mask
        self.vocab_size = vocab_size
        self.freeze_feature_encoder = freeze_feature_encoder
        self.freeze_base_model = freeze_base_model
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_loss_weight = ctc_loss_weight
        self.special_ids = special_ids

    @property
    def encoder_config(self) -> HubertConfig:
        return HubertConfig(
            hidden_size=self.encoder_embed_dim,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.encoder_attention_heads,
            intermediate_size=self.encoder_ffn_embed_dim,
            hidden_act=self.activation_fn,
            hidden_dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            layerdrop=self.encoder_layerdrop,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.conv_pos,
            num_conv_pos_embedding_groups=self.conv_pos_groups,
            feat_extract_activation="gelu",
            do_stable_layer_norm=self.do_stable_layer_norm,
            max_position_embeddings=self.max_target_positions,
            learned_pos=self.decoder_learned_pos,
            share_input_output_embed=self.share_decoder_input_output_embed,
        )

    @property
    def decoder_config(self) -> HubertConfig:
        return HubertConfig(
            hidden_size=self.decoder_embed_dim,
            num_hidden_layers=self.decoder_layers,
            num_attention_heads=self.decoder_attention_heads,
            intermediate_size=self.decoder_ffn_embed_dim,
            hidden_act=self.activation_fn,
            hidden_dropout=self.decoder_dropout,
            activation_dropout=self.decoder_activation_dropout,
            attention_dropout=self.decoder_attention_dropout,
            layerdrop=self.decoder_layerdrop,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.conv_pos,
            num_conv_pos_embedding_groups=self.conv_pos_groups,
            feat_extract_activation="gelu",
            do_stable_layer_norm=self.do_stable_layer_norm,
            max_position_embeddings=self.max_target_positions,
            learned_pos=self.decoder_learned_pos,
            share_input_output_embed=self.share_decoder_input_output_embed,
        )
