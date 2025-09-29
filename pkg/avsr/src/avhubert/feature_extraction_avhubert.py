import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from python_speech_features import logfbank
from transformers.feature_extraction_utils import BatchFeature
from transformers import FeatureExtractionMixin


class AVHubertFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_values", "pixel_values"]

    def __init__(
        self,
        max_sample_size: int | None = None,
        normalize: bool = True,
        stack_order_audio: int = 4,
        image_crop_size: int = 88,
        image_mean: float = 0.421,
        image_std: float = 0.165,
        sr: int = 16_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_sample_size = max_sample_size
        self.normalize = normalize
        self.stack_order_audio = stack_order_audio
        self.image_crop_size = image_crop_size
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.CenterCrop(image_crop_size),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([image_mean], [image_std]),
            ]
        )
        self.sr = sr

    def _load_video(self, video: str | np.ndarray):
        if isinstance(video, str):
            cap = cv2.VideoCapture(video)
            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frames_np = np.stack(frames, axis=0)
        else:
            frames_np = video
        return torch.from_numpy(frames_np).unsqueeze(dim=1)

    def _load_audio(self, audio: str | np.ndarray):
        def stacker(feats, stack_order):
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        sr = None
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16_000)
        if sr is None:
            sr = self.sr
        fbank = logfbank(audio, samplerate=sr).astype(np.float32)
        fbank = stacker(fbank, self.stack_order_audio)
        return torch.from_numpy(fbank)

    def _align_time_steps(
        self,
        audio: list[np.ndarray],
        video: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        aligned_indices = []
        for sample_audio, sample_video in zip(audio, video):
            diff = len(sample_audio) - len(sample_video)
            if diff != 0:
                aligned_indices.append(
                    torch.arange(0, len(sample_audio)).float() * len(sample_video) / len(sample_audio)
                )
            else:
                aligned_indices.append(torch.arange(0, len(sample_audio)))
        return (
            audio,
            [
                sample[torch.clamp(torch.floor(indices), max=sample.shape[0] - 1).long()]
                for sample, indices in zip(video, aligned_indices)
            ],
        )

    def __call__(
        self,
        raw_audio: np.ndarray | str | list[np.ndarray] | list[str] | None = None,
        raw_video: np.ndarray | str | list[np.ndarray] | list[str] | None = None,
    ) -> BatchFeature:
        if not isinstance(raw_audio, list):
            raw_audio = [raw_audio]
        if not isinstance(raw_video, list):
            raw_video = [raw_video]

        audio = [self._load_audio(sample) if sample is not None else None for sample in raw_audio]
        video = [self._load_video(sample) if sample is not None else None for sample in raw_video]
        for batch_idx in range(len(audio)):
            sample_a = audio[batch_idx]
            sample_v = video[batch_idx]
            assert sample_a is not None or sample_v is not None
            if sample_a is None:
                sample_a = torch.zeros((sample_v.shape[0], 26 * self.stack_order_audio))
                audio[batch_idx] = sample_a
            elif sample_v is None:  # 25 fps
                sample_v = torch.zeros((sample_a.shape[0], 1, self.image_crop_size, self.image_crop_size))
                video[batch_idx] = sample_v

        audio, video = self._align_time_steps(audio, video)
        max_length = max(len(data) for data in audio)
        input_values = []
        pixel_values = []
        padding_mask = []
        for feat_audio, feat_video in zip(audio, video):
            remainder_length = max_length - len(feat_audio)
            audio_remainder = torch.zeros(
                size=(remainder_length,) + feat_audio.size()[1:],
                dtype=feat_audio.dtype,
            )
            video_remainder = torch.zeros(
                size=(remainder_length,) + feat_video.size()[1:],
                dtype=feat_video.dtype,
            )

            feat_audio = torch.cat((feat_audio, audio_remainder))
            feat_video = torch.cat((feat_video, video_remainder))
            if self.max_sample_size:
                feat_audio = feat_audio[: self.max_sample_size]
                feat_video = feat_video[: self.max_sample_size]
            pad_mask = torch.zeros(max_length)
            pad_mask[max_length - remainder_length :] = 1

            input_values.append(feat_audio)
            pixel_values.append(feat_video)
            padding_mask.append(pad_mask)

        input_values = torch.stack(input_values)
        batch = BatchFeature(
            {
                "input_values": (
                    F.layer_norm(input_values, input_values.shape[2:]) if self.normalize else input_values
                ),
                "pixel_values": self.transforms(torch.stack(pixel_values)),
                "padding_mask": torch.stack(padding_mask).bool(),
            }
        )
        return batch

    def to_dict(self):
        output = super().to_dict()
        output["transforms"] = self._transforms_to_dict(output["transforms"])
        return output

    def _transforms_to_dict(self, transforms: transforms.Compose):
        output = []
        for component in transforms.__dict__["transforms"]:
            name = component.__class__.__name__
            component_dict = {"transforms_type": name}
            for k, v in component.__dict__.items():
                if k.startswith("_"):
                    continue
                component_dict[k] = str(v)
            output.append(component_dict)
        return output
