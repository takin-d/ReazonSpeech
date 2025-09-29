from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoFeatureExtractor

from .avhubert import (
    AVHubertConfig,
    AVHubertModel,
    AVHubertForConditionalGeneration,
    AVHubertFeatureExtractor,
    AVHubertProcessor,
)

__all__ = [
    "AVHubertConfig",
    "AVHubertModel",
    "AVHubertForConditionalGeneration",
    "AVHubertFeatureExtractor",
    "AVHubertProcessor",
]

AutoConfig.register(AVHubertConfig.model_type, AVHubertConfig)
AutoModel.register(AVHubertConfig, AVHubertModel)
AutoModelForSpeechSeq2Seq.register(AVHubertConfig, AVHubertForConditionalGeneration)
AutoFeatureExtractor.register(AVHubertConfig.model_type, AVHubertFeatureExtractor)
AutoProcessor.register(AVHubertConfig.model_type, AVHubertProcessor)
