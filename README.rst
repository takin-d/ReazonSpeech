============
ReazonSpeech
============

This repository provides access to the main user tooling of ReazonSpeech and AVista projects.

**ReazonSpeech**

* Building the world’s largest open Japanese speech corpus.
* Project page: https://research.reazon.jp/projects/ReazonSpeech/

**AVista 🐦‍🔥 -- Toward a new vista of Human–Robot Interaction**

* Project for noise‑robust multimodal speech recognition for human–robot interaction and beyond.
* "AVista" stands for **A**\udio‑**VI**\sual **S**\peech **T**\ranscription & **A**\lignment, and the name contains the Latin words "avis", meaning bird, and "vista", meaning view or sight.

Install
=======

Installation instructions live in each package’s README. See Packages below.

Packages
========

`reazonspeech.k2.asr <pkg/k2-asr>`_

* Next-gen Kaldi model that is very fast and accurate.
* The total number of parameters is 159M. Requires `sherpa-onnx <https://github.com/k2-fsa/sherpa-onnx>`_.
* Also contains a bilingual (ja-en) model, which is highly accurate at language detection in bilingual settings of Japanese and English.
* For development: "ja-en-mls-5k" model trained on 5k hours of ReazonSpeech and MLS English data each

`reazonspeech.nemo.asr <pkg/nemo-asr>`_

* Implements a fast, accurate speech recognition based on FastConformer-RNNT.
* The total number of parameters is 619M. Requires `Nvidia Nemo <https://github.com/NVIDIA/NeMo>`_.

`reazonspeech.espnet.asr <pkg/espnet-asr>`_

* Speech recognition with a Conformer-Transducer model.
* The total number of parameters is 120M. Requires `ESPnet <https://github.com/espnet/espnet>`_.

`reazonspeech.avsr <pkg/avsr>`_

* Audio‑Visual Speech models for AVista 🐦‍🔥, including pretrained models.
* Follows the Hugging Face Transformers interface; supports Auto classes for quick use.

`reazonspeech.evaluation <pkg/evaluation>`_

* Provides a set of tools to evaluate ReazonSpeech models and other speech recognition models.

`reazonspeech.espnet.oneseg <pkg/espnet-oneseg>`_

* Provides a set of tools to analyze Japanese "one-segment" TV stream.
* Use this package to create Japanese audio corpus.

LICENSE
=======

::

    Copyright 2022-2025 Reazon Holdings, inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
