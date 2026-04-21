Comprehensive Architecture Blueprint: All-in-One Multi-Talker, Multi-Task Vietnamese Speech-to-Text (STT) System
1. Architectural Overview & System Paradigm
The transition from cascaded Speech-to-Text (STT) pipelines to unified end-to-end (E2E) architectures is imperative for solving the "cocktail party problem" in Vietnamese speech processing. Vietnamese—a tonal, monosyllabic language with high dialectal variance—presents unique acoustic challenges, especially in highly overlapped multi-party conversations.

This technical blueprint outlines a production-grade, multi-task, multi-talker ASR framework. The proposed topology is a unified computational graph that integrates Streaming Sortformer for permutation-resolved speaker diarization , a Zipformer-Transducer acoustic backbone (replacing Fast-Conformer), and a Prompt-based Multi-Task Learning (MTL) network inspired by SenseVoice for extracting Paralinguistic metadata (Speech Emotion Recognition, Voice Activity Detection, Gender Classification, and continuous Age Regression). Furthermore, the pipeline is optimized for real-time human-computer interaction via native End-of-Utterance (EOU) token emission and Inverse Text Normalization (ITN).   

2. Speaker Diarization: Scaling the Streaming Sortformer
Speaker diarization answers "who spoke when" and provides deterministic speaker-activity boundaries utilized by the downstream ASR decoder to isolate overlapping features. NVIDIA's Streaming Sortformer operates as a micro-tracker using a Transformer-encoder architecture to emit frame-level speaker activity probabilities directly.

2.1 Resolving the Permutation Problem: Sort Loss vs. PIL
Traditional E2E diarization relies on Permutation Invariant Loss (PIL), which suffers from O(N!) combinatorial explosion as the number of concurrent speakers N increases. Sortformer mitigates this by introducing Sort Loss (Arrival Time Sort - ATS Loss). ATS imposes an inductive bias by forcing the network to output speaker activations strictly sorted by their chronological arrival time in the audio segment. This temporal anchoring stabilizes gradient descent and allows cross-entropy-based training without strictly relying on permutation calculations.   

2.2 Overcoming the 4-Speaker Constraint (Architectural Surgery)
The default NVIDIA NeMo Sortformer checkpoints (e.g., diar_streaming_sortformer_4spk-v2.1) are architecturally hardcoded to a maximum of 4 speakers via the final dense classification layer (emitting 4 independent sigmoid logits). To fine-tune the model for N>4 (e.g., 5, 6, or dynamic multi-party meetings), architectural surgery and Hydra config modifications are mandatory.

Hydra Configuration Overrides:
To expand the network's capacity, target the following keys in the .yaml config:

model.max_num_of_spks: Change from 4 to N. This acts as a global variable.

model.train_ds.num_spks & model.validation_ds.num_spks: Set to N to ensure dataloaders yield N-channel binary matrices.

sortformer_modules.num_spks: Overwrite to N to initialize a new linear projection head with N output dimensions.

Fine-tuning Strategy:

Freeze Acoustic Encoders: Freeze the lower 18 layers of the Transformer encoder (initialized via NEST - NeMo Encoders for Speech Task) to preserve foundational acoustic feature extraction.   

Re-initialize Classification Head: The final Linear(hidden_dim, N) layer must be randomly initialized.

Loss Weighting: Adjust the hybrid loss weights (pil_weight and ats_weight). During the warm-up phase for N>4, temporarily increase ats_weight > pil_weight to enforce chronological sorting before optimizing for permutation matching, thus preventing gradient instability.

Hybrid Macro-Tracking Fallback: For unbounded environments (e.g., N≫6), utilize Sortformer strictly for short-window micro-tracking. The extracted frame-level embeddings and local segments should be passed to a sequential, lightweight clustering backend (e.g., online k-means or Multi-Scale Diarization Decoder - MSDD) to maintain global speaker identity.

3. Multitalker ASR Core: Zipformer & Speaker Kernel Injection
Standard ASR models fail during overlapping speech. To transcribe multi-talker audio, the framework adopts a Self-Speaker Adaptation methodology inspired by the Multitalker Parakeet Streaming architecture.

3.1 Self-Speaker Adaptation via Kernel Injection
Instead of requiring pre-enrolled speaker embeddings (target-speaker ASR), the model ingests the continuous speaker activity probabilities  
Y
^
 ∈ 
T×N
  directly from the Sortformer module.

Sinusoidal Kernel Functions: The speaker labels/timestamps predicted by Sortformer are embedded into the ASR encoder using sinusoidal kernel functions. This bridges timestamps and linguistic tokens.

Multi-Instance Decoding: The system dynamically spawns an independent ASR instance for each detected active speaker. Each instance processes the same acoustic features but applies attention masking guided by the injected speaker-specific kernels, effectively isolating the target speaker's phonetic trajectory from the overlapping background mixture.

Cache-Aware Streaming: To maintain an 80ms latency profile, the model employs a stateful cache-based inference mechanism, strictly limiting right-context lookahead.

3.2 Acoustic Modeling Upgrade: Zipformer Transducer
We replace the Fast-Conformer backbone with Zipformer, which provides faster convergence, reduced memory footprint, and better long-context dependency modeling.

Key Zipformer Architectural Innovations:

Component	Zipformer Modification	Engineering Impact
U-Net Style Downsampling	Processes intermediate stacks at lower frame rates (e.g., 50Hz → 25Hz → 12.5Hz → 6.25Hz) before upsampling back to 50Hz.	Drastically cuts the quadratic memory cost of self-attention, allowing for massive receptive fields necessary for resolving overlapped Vietnamese tonal dependencies.
BiasNorm	
Replaces LayerNorm. Equation: BiasNorm(x)=( 
RMS[x−b]
x
​
 )⋅exp(γ), where b is a learnable channel-wise bias.

Standard LayerNorm destroys vector length information. BiasNorm preserves amplitude/energy dynamics inherent to speech features.

Activations	Swish is replaced by SwooshR and SwooshL.	Mitigates vanishing gradients and prevents neuron saturation (dead channels) near zero.
ScaledAdam Optimizer	Scales updates by each tensor's current scale and explicitly learns the parameter scale.	Achieves significantly faster convergence on noisy, synthetic multi-talker datasets.
  
NeMo Integration: To deploy the Icefall/k2 native Zipformer inside NeMo, wrap the PyTorch Zipformer code in a custom encoder class inheriting from nemo.collections.asr.models.EncDecRNNTBPEModel. Ensure the final projection layer precisely matches model_defaults.enc_hidden expected by the Transducer Joint Network.

4. Multi-Task Learning (MTL) Framework for Paralinguistics
The "All-in-One" system expands beyond Character Error Rate (CER) optimization to concurrently predict Speech Emotion (SER), Gender, Voice Activity (VAD), and Age.

4.1 Prompt-Based Unified Modeling (Discrete Tasks)
Following the FunAudioLLM SenseVoice paradigm , discrete classification tasks are handled via prompt-based sequence generation rather than dedicated MLP heads.   

Query Injection: Task-specific embeddings (e.g., language_query, event_emo_query) are instantiated via self.embed and concatenated directly to the front of the Mel-filterbank acoustic features (fbank):
speech = torch.cat((input_query, speech), dim=1).   

Unified CTC Decoder: The shared Zipformer encoder feeds into a single CTC decoder layer (self.ctc.log_softmax(encoder_out)). The vocabulary is expanded to include special tokens like <|FEMALE|>, <|MALE|>, <|HAPPY|>, <|SAD|>, etc.   

VAD via CTC Alignment: VAD boundaries and timestamps are extracted directly from the CTC alignment peaks, removing the need for a standalone MarbleNet VAD model.

4.2 Continuous Regression Head (Age Estimation)
Age is a continuous variable. Discretizing it into bins (e.g., "20-30 years") destroys quantitative fidelity. Thus, an auxiliary regression head is grafted onto the shared Zipformer encoder.

Temporal Pooling: Frame-level encoder outputs (T×D) are condensed using GlobalAveragePooling1D (or attentive pooling) into a fixed-length utterance-level vector (1×D).   

MLP Regression Network: The pooled vector passes through a deep feed-forward block: Dense(128, ReLU) → Dense(1, Linear).   

Loss Formulation: Optimized via Mean Absolute Error (MAE) or Mean Squared Error (MSE). The gradients from this head flow back into the Zipformer, enriching the shared latent space with vocal-tract physiological features.

5. Interaction Optimization: EOU, ITN, and PnC
To facilitate seamless real-time dialogue agents (e.g., LLM-based voice bots), the ASR output must be instantly actionable and grammatically normalized.

5.1 Native End-of-Utterance (EOU) Detection
Traditional systems rely on VAD silence timeouts (often >500ms) to determine when a user stops speaking. We adopt the Parakeet-Realtime-EOU methodology.

Mechanism: The Transducer vocabulary is augmented with an <EOU> token. The model is explicitly trained to emit <EOU> immediately upon detecting semantic and acoustic completion of a sentence.

Latency: This drastically drops endpointing latency to 80ms – 160ms, enabling instantaneous downstream LLM response.

5.2 Inverse Text Normalization (ITN)
Vietnamese ASR yields raw outputs ("một trăm hai mươi nghìn đồng"). ITN maps this to ("120.000 VNĐ").

Approach A (Neural Prompting): Similar to FunASR-Nano, pass a conditioning prompt <|use_itn|> into the encoder. The seq2seq model directly learns to emit normalized text.

Approach B (Rule-based Fallback): For strict deterministic control over financial/date entities, utilize an open-source, zero-dependency Python 3.8+ regex and dictionary-based pipeline (e.g., VietNormalizer). It executes single-pass deterministic conversions outside the neural graph, eliminating hallucination risks.

5.3 Integrated Spelling Correction and Punctuation/Capitalization (PnC) Restoration
To bridge the gap between raw ASR transcription and NLP-ready text, a dedicated post-processing system must be deployed to correct spelling errors and restore punctuation/capitalization. Instead of relying exclusively on massive LLMs, the optimal approach is to fine-tune lightweight, pre-trained Vietnamese language models to leverage their inherent linguistic representations.

Dataset Generation Pipeline (Training & Testing):
A robust pipeline to synthesize error-infused datasets from clean corpora is required.

Rule-based Error Injection: Programmatically remove punctuation and lowercase all text. Inject synthetic spelling errors by simulating Vietnamese keyboard input methods (Telex/VNI typo patterns) and common phonological/homophone confusions (e.g., swapping "tr" and "ch", "s" and "x").

LLM-based Synthetic Generation: Utilize Large Language Models for aspect-based synthetic data generation to create complex, context-aware grammatical and spelling errors. This is particularly effective for modeling errors in specialized domains, such as legal or administrative texts.

Model Architecture & Fine-tuning:

Sequence-to-Sequence (Seq2Seq): Fine-tune generative models such as BARTpho (BARTpho-word/syllable) or ViT5. These models treat error correction and punctuation recovery as a unified translation task, demonstrating state-of-the-art performance in Vietnamese text generation and summarization.

Token Classification: Alternatively, fine-tune encoder-only models (e.g., PhoBERT) to predict punctuation marks as discrete classes per token, similar to deep punctuation prediction architectures.

Word-Level Confidence Scoring:
Providing a confidence score for each corrected word is critical for downstream human-in-the-loop review or programmatic filtering.

For Token Classification Models: The confidence score is naturally derived from the softmax probability distribution of the classification head for each token.

For Seq2Seq Models (BARTpho/ViT5): Extract token-level log-probabilities (logprobs) directly from the Beam Search decoding outputs. To align these sub-word generation probabilities with the final sentence, apply a sequence alignment algorithm (such as Levenshtein edit distance) to map the generated tokens back to the raw input words. The exponential of the token's log-probability serves as its reliable word-level confidence score.

6. Data Engineering & Training Dynamics
6.1 Synthetic Multi-Talker Generation
Since fully-annotated, overlapping multi-talker Vietnamese datasets are severely limited, large-scale synthetic generation is required.

Pipeline: Utilize the NeMo Multispeaker Simulator. Randomly sample N utterances from high-fidelity single-speaker datasets (VoxVietnam, VIVOS, ViSpeech).

Dynamic Overlap: Programmatically mix waveforms at varying Overlap Error Rates (OER) from 0% to 100%.

Acoustic Augmentation: Convolve mixtures with diverse Room Impulse Responses (RIRs) (e.g., shoebox, auditorium) and layer isotropic/point-source noise (MUSAN dataset) at randomized Signal-to-Noise Ratios (SNR) to ensure ecological validity.

Manifest Generation: The simulator natively dumps JSON-lines manifest files with exact millisecond timestamps for speaker boundaries, providing perfect ground truth for Sortformer training.

6.2 Multi-Objective Optimization
The global loss function is a dynamically weighted linear combination:

L 
Total
​
 =λ 
1
​
 L 
RNN−T
​
 +λ 
2
​
 L 
SortLoss
​
 +λ 
3
​
 L 
CTC_Prompt
​
 +λ 
4
​
 L 
Age_MAE
​
 
Note: Due to scale differences between Transducer log-probs, Cross-Entropy (CTC), and MAE (Age), implement dynamic uncertainty weighting or gradient normalization to prevent the dominant ASR task from causing catastrophic interference on paralinguistic heads.

6.3 Curriculum Fine-Tuning Strategy
Phase 1 (Acoustic Pretraining): Train the Zipformer-Transducer exclusively on clean, single-speaker Vietnamese data to stabilize the phonetic alignment.

Phase 2 (Multitalker Adaptation): Freeze the lower encoder layers. Activate Sortformer heads and inject Sinusoidal Kernel Functions. Train on the synthetic overlapping dataset with high λ 
1
​
 ,λ 
2
​
  weights.

Phase 3 (Paralinguistic SFT): Unfreeze prompt tokens (<|HAPPY|>, <|FEMALE|>) and the Age MLP regression head. Fine-tune on strictly annotated metadata corpora (e.g., VNEMOS, Common Voice 21) using a severely reduced learning rate for the main encoder to preserve ASR fidelity.


icml.cc
ICML Poster Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems
Mở trong cửa sổ mới

github.com
FunAudioLLM/SenseVoice: Multilingual Voice Understanding Model - GitHub
Mở trong cửa sổ mới

arxiv.org
Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems - arXiv
Mở trong cửa sổ mới

arxiv.org
Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens - arXiv
Mở trong cửa sổ mới

ristohinno.medium.com
Under the hood of zipformer. Fast and accurate ASR model | by Risto Hinno | Medium
Mở trong cửa sổ mới

huggingface.co
Sharris/age_detection_regression - Hugging Face
Mở trong cửa sổ mới

docs.nvidia.com
Speaker Diarization — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

reddit.com
NVIDIA Sortformer v2 (Speaker Diarization) ported to Rust/ONNX - Reddit
Mở trong cửa sổ mới

docs.nvidia.com
Customization — NVIDIA NIM Riva ASR
Mở trong cửa sổ mới

github.com
Question: How to extend Streaming Sortformer beyond 4 speakers? · Issue #14546 - GitHub
Mở trong cửa sổ mới

arxiv.org
CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
Mở trong cửa sổ mới

arxiv.org
TouchASP: Elastic Automatic Speech Perception that Everyone Can Touch - arXiv
Mở trong cửa sổ mới

researchgate.net
Identity, Gender, Age, and Emotion Recognition from Speaker Voice with Multi-task Deep Networks for Cognitive Robotics - ResearchGate
Mở trong cửa sổ mới

arxiv.org
[2410.03458] Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges
Mở trong cửa sổ mới

microsoft.com
Distant conversational speech recognition: Challenges and Opportunities - Microsoft
Mở trong cửa sổ mới

arxiv.org
LibriConvo: Simulating Conversations from Read Literature for ASR and Diarization - arXiv
Mở trong cửa sổ mới

pmc.ncbi.nlm.nih.gov
Real-time multilingual speech recognition and speaker diarization system based on Whisper segmentation - PMC
Mở trong cửa sổ mới

rev.com
Reverb Open-Source ASR and Diarization Models | Rev
Mở trong cửa sổ mới

docs.nvidia.com
Models — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

catalog.ngc.nvidia.com
STT En Conformer-Transducer Medium - NGC Catalog - NVIDIA
Mở trong cửa sổ mới

docs.nvidia.com
NeMo ASR Configuration Files — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

huggingface.co
README.md · nvidia/stt_ru_conformer_transducer_large at main - Hugging Face
Mở trong cửa sổ mới

github.com
[Help needed] Conformer-Transducer Streaming ASR with Microphone #5264 - GitHub
Mở trong cửa sổ mới

github.com
NeMo/tutorials/asr/Streaming_Multitalker_ASR.ipynb at main - GitHub
Mở trong cửa sổ mới

docs.nvidia.com
NeMo Speaker Diarization Configuration Files — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

huggingface.co
nvidia/diar_sortformer_4spk-v1 · Fine-tuning NeMo Sortformer for Custom Speaker Diarization - Hugging Face
Mở trong cửa sổ mới

docs.nvidia.com
Models — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

github.com
GitHub - pyannote/pyannote-audio: Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding
Mở trong cửa sổ mới

mdpi.com
Improving Speaker Diarization for Overlapped Speech with Texture-Aware Feature Fusion - MDPI
Mở trong cửa sổ mới

scalastic.io
Whisper and Pyannote: The Ultimate Solution for Speech Transcription
Mở trong cửa sổ mới

medium.com
Speaker diarization using Whisper ASR and Pyannote | by Ritesh - Medium
Mở trong cửa sổ mới

arxiv.org
CoVoMix: Advancing Zero-Shot Speech Generation for Human-like Multi-talker Conversations - arXiv.org
Mở trong cửa sổ mới

arxiv.org
ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts - arXiv
Mở trong cửa sổ mới

researchgate.net
(PDF) VNEMOS: Vietnamese Speech Emotion Inference Using Deep Neural Networks
Mở trong cửa sổ mới

github.com
GitHub - k2-fsa/sherpa-onnx: Speech-to-text, text-to-speech, speaker diarization, speech enhancement, source separation, and VAD using next-gen Kaldi with onnxruntime without Internet connection. Support embedded systems, Android, iOS, HarmonyOS, Raspberry Pi, RISC-V, RK NPU, Axera NPU, Ascend NPU, x86_64 servers, websocket server/client, support 12 programming languages
Mở trong cửa sổ mới

github.com
sherpa-onnx/nodejs-addon-examples/README.md at master - GitHub
Mở trong cửa sổ mới

pmc.ncbi.nlm.nih.gov
A Deep Learning Method Using Gender-Specific Features for Emotion Recognition - PMC
Mở trong cửa sổ mới

arxiv.org
FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs - arXiv.org
Mở trong cửa sổ mới

pdfs.semanticscholar.org
multi-head attention for speech emotion recognition with auxiliary learning of gender recognition - Semantic Scholar
Mở trong cửa sổ mới

docs.nvidia.com
End-to-End Speaker Diarization Configuration Files — NVIDIA ...
Mở trong cửa sổ mới

developer.nvidia.com
Identify Speakers in Meetings, Calls, and Voice Apps in Real-Time ...
Mở trong cửa sổ mới

docs.nvidia.com
Speaker Diarization — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

docs.nvidia.com
Speaker Diarization — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

github.com
NeMo/tutorials/speaker_tasks/Speaker_Diarization_Training.ipynb at main - GitHub
Mở trong cửa sổ mới

arxiv.org
Generating Data with Text-to-Speech and Large-Language Models for Conversational Speech Recognition - arXiv.org
Mở trong cửa sổ mới

mdpi.com
Age and Gender Recognition Using a Convolutional Neural Network with a Specially Designed Multi-Attention Module through Speech Spectrograms - MDPI
Mở trong cửa sổ mới

huggingface.co
funasr/fsmn-vad - Hugging Face
Mở trong cửa sổ mới

github.com
GitHub - modelscope/FunASR: A Fundamental End-to-End Speech Recognition Toolkit and Open Source SOTA Pretrained Models, Supporting Speech Recognition, Voice Activity Detection, Text Post-processing etc.
Mở trong cửa sổ mới

youtube.com
Real Time Age And Gender Recognition Using Pre-Trained Caffe models lPython Opencv|KNOWLEDGE DOCTOR| - YouTube
Mở trong cửa sổ mới

kaggle.com
common-voice-vi-21 - Kaggle
Mở trong cửa sổ mới

aclanthology.org
Gender and Dialect Classification for the Vietnamese Language - ACL Anthology
Mở trong cửa sổ mới

isca-archive.org
Vietnam-Celeb: a large-scale dataset for Vietnamese speaker recognition - ISCA Archive
Mở trong cửa sổ mới

data.mendeley.com
FPT Open Speech Dataset (FOSD) - Vietnamese - Mendeley Data
Mở trong cửa sổ mới

docs.nvidia.com
Models — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

docs.nvidia.com
Models — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

arxiv.org
VietSuperSpeech: A Large-Scale Vietnamese Conversational Speech Dataset for ASR Fine-Tuning in Chatbot, Customer Support, and Call Center Applications - arXiv.org
Mở trong cửa sổ mới

arxiv.org
[2210.15715] Simulating realistic speech overlaps improves multi-talker ASR - arXiv
Mở trong cửa sổ mới

isca-archive.org
ViCocktail: Automated Multi-Modal Data Collection for Vietnamese Audio-Visual Speech Recognition - ISCA Archive
Mở trong cửa sổ mới

ieeexplore.ieee.org
Speaker Identification in Multi-Talker Overlapping Speech Using Neural Networks - IEEE Xplore
Mở trong cửa sổ mới

isl.iar.kit.edu
Synthetic Conversations Improve Multi-Talker ASR - KIT
Mở trong cửa sổ mới

openreview.net
Zipformer: A faster and better encoder for automatic speech recognition - OpenReview
Mở trong cửa sổ mới

github.com
Multilingual Pretraining, Cross-Lingual Finetuning, and LID Heads with Zipformer · Issue #2052 · k2-fsa/icefall - GitHub
Mở trong cửa sổ mới

arxiv.org
[2310.11230] Zipformer: A faster and better encoder for automatic speech recognition - arXiv
Mở trong cửa sổ mới

k2-fsa.github.io
Zipformer-transducer-based Models - sherpa-onnx
Mở trong cửa sổ mới

k2-fsa.github.io
Recipes — icefall 0.1 documentation - GitHub Pages
Mở trong cửa sổ mới

icefall.readthedocs.io
How to create a recipe — icefall 0.1 documentation - Read the Docs
Mở trong cửa sổ mới

youtube.com
Step 3: Training the Zipformer ASR Model with Icefall | End-to-End Speech Recognition Tutorial - YouTube
Mở trong cửa sổ mới

huggingface.co
nvidia/diar_sortformer_4spk-v1 - Hugging Face
Mở trong cửa sổ mới

docs.nvidia.com
How to fine-tune a Riva NMT Bilingual model with Nvidia NeMo
Mở trong cửa sổ mới

galhever.medium.com
NeMo Toolkit in Different Languages | by Gal Hever - Medium
Mở trong cửa sổ mới

developer.nvidia.com
Multilingual and Code-Switched Automatic Speech Recognition with NVIDIA NeMo
Mở trong cửa sổ mới

github.com
Does non-English TTS training work properly now? #4606 - GitHub
Mở trong cửa sổ mới

github.com
Age Estimation by CNN Based Regression Model - GitHub
Mở trong cửa sổ mới

pmc.ncbi.nlm.nih.gov
Multiple Regression Modeling for Age Estimation by Assessment and Comparison of Spheno-Occipital Synchondrosis Fusion and Cervical Vertebral Maturation Stages - PMC
Mở trong cửa sổ mới

jcsce.vnu.edu.vn
The 2025 VLSP Task on Vietnamese Voice Conversion: Overview and Preliminary Results
Mở trong cửa sổ mới

huggingface.co
Vietnamese speech dataset - a doof-ferb Collection - Hugging Face
Mở trong cửa sổ mới

aclanthology.org
The 2025 VLSP Task on Vietnamese Voice Conversion: Overview and Preliminary Results - ACL Anthology
Mở trong cửa sổ mới

arxiv.org
Whisper based Cross-Lingual Phoneme Recognition between Vietnamese and English
Mở trong cửa sổ mới

docs.nvidia.com
NeMo ASR API — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

youtube.com
[Detailed Paper Reading] Zipformer: A faster and better encoder for automatic speech recognition - YouTube
Mở trong cửa sổ mới

docs.nvidia.com
NeMo ASR Configuration Files — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

arxiv.org
Zipformer: A faster and better encoder for automatic speech recognition - arXiv
Mở trong cửa sổ mới

github.com
wheevu/nemo-vietnamese-asr: Pipeline for harvesting, validating, and training Vietnamese speech recognition models. - GitHub
Mở trong cửa sổ mới

github.com
NeMo/tutorials/tools/Multispeaker_Simulator.ipynb at main - GitHub
Mở trong cửa sổ mới

colab.research.google.com
Finetuning FastPitch for a new speaker - Colab
Mở trong cửa sổ mới

docs.nvidia.com
Tutorials — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

docs.nvidia.com
Datasets — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

kaggle.com
VIVOS: Vietnamese Speech Corpus for ASR - Kaggle
Mở trong cửa sổ mới

openslr.org
Room Impulse Response and Noise Database - openslr.org
Mở trong cửa sổ mới

github.com
A list of publicly available room impulse response datasets and scripts to download them. - GitHub
Mở trong cửa sổ mới

arxiv.org
[2507.09750] MB-RIRs: a Synthetic Room Impulse Response Dataset with Frequency-Dependent Absorption Coefficients - arXiv
Mở trong cửa sổ mới

arxiv.org
VoxVietnam: a Large-Scale Multi-Genre Dataset for Vietnamese Speaker Recognition
Mở trong cửa sổ mới

pmc.ncbi.nlm.nih.gov
Gender and Age Estimation Methods Based on Speech Using Deep Neural Networks - PMC
Mở trong cửa sổ mới

arxiv.org
Fun-ASR Technical Report - arXiv.org
Mở trong cửa sổ mới

isca-archive.org
FunASR: A Fundamental End-to-End Speech Recognition Toolkit - ISCA Archive
Mở trong cửa sổ mới

github.com
Fun-ASR/docs/finetune.md at main - GitHub
Mở trong cửa sổ mới

researchgate.net
FunASR: A Fundamental End-to-End Speech Recognition Toolkit - ResearchGate
Mở trong cửa sổ mới

docs.nvidia.com
Models — NVIDIA NeMo Framework User Guide
Mở trong cửa sổ mới

github.com
ONNX Export Fails for Streaming Sortformer Model (diar_streaming_sortformer_4spk-v2) · Issue #15077 · NVIDIA-NeMo/NeMo - GitHub
Mở trong cửa sổ mới

github.com
BlackPlatinum/Human-Voice-Predictor: Real-time Gender and Age Recognition from Audio using CRNN - GitHub
Mở trong cửa sổ mới

github.com
SenseVoice.cpp/README-EN.md at main · lovemefan/SenseVoice.cpp · GitHub
Mở trong cửa sổ mới

huggingface.co
FunAudioLLM/SenseVoiceSmall - Hugging Face
Mở trong cửa sổ mới

labsites.rochester.edu
JOINT SPEAKER DIARIZATION AND RECOGNITION USING CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS Zhihan Zhou, Yichi Zhang, Student Me - University Lab Sites
Mở trong cửa sổ mới

online-journals.org
Convolutional Neural Network Architectures for Gender, Emotional Detection from Speech and Speaker Diarization - Online-Journals.org
Mở trong cửa sổ mới

isca-archive.org
Integrating Emotion Recognition with Speech Recognition and Speaker Diarisation for Conversations - ISCA Archive
Mở trong cửa sổ mới

arxiv.org
Emotion Recognition in Multi-Speaker Conversations through Speaker Identification, Knowledge Distillation, and Hierarchical Fusion - arXiv
Mở trong cửa sổ mới

aclanthology.org
Emotion Recognition in Multi-Speaker Conversations through Speaker Identification, Knowledge Distillation, and Hierarchical Fusion - ACL An

Reference code need to analyze previous work can brow from:
https://github.com/modelscope/FunASR
https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer
https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
https://github.com/FunAudioLLM/SenseVoice
https://huggingface.co/FunAudioLLM/SenseVoiceSmall