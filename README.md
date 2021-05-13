# VietASR (NVIDIA NeMo ToolKit)
⚡ Some experiment with [NeMo](https://github.com/NVIDIA/NeMo) ⚡
# Result
* Model: [QuartzNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#quartznet) is a smaller version of [Jaser](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#jasper)  model  
* I list the word error rate (WER) with and without LM of major ASR tasks.

| Task                   | CER (%) | WER (%) | +LM WER (%) |
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| VIVOS (TEST)            |  6.80 | 18.02 | 15.72 |
| VLSP2018                |  6.87 | 16.26 |  N/A  |
| VLSP2020 T1             | 14.73 | 30.96 |  N/A  |
| VLSP2020 T2             | 41.67 | 69.15 |  N/A  |

Model was trained with ~500 hours Vietnamese speech dataset, was collected from youtube, radio, call center(8k), text to speech data and some public dataset (vlsp, vivos, fpt). It is very small model (13M parameters) make it inference so fast ⚡  

# Installation
* ctcdecoder, [kemlm](https://github.com/kpu/kenlm) for LM Decode  
`pip install ds-ctcdecoder`
* and some python libraries: `torch, numpy, librosa, flask, flask_socketio, requests,...`

# Run Demo
* Vietnamese Model ([pretrained](model_vietasr2/checkpoints)): `python flask_upload_record_vn.py`  
* Video demo in Youtube: https://youtu.be/P3mhEngL1us  
[![Video demo](https://img.youtube.com/vi/P3mhEngL1us/maxresdefault.jpg)](https://youtu.be/P3mhEngL1us)  

* English Model ([pretrained](model_english)): `python flask_upload_record_en.py`  

# TODO   
* Conformer Model  
* Transformer LM instead of kenlm  
* Data augumentation: speed, noise, pitch shift, time shift,...  
* FastAPI
