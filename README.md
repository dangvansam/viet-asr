# VietASR (Vietnamese Automatic Speech Recognition)
⚡ Some experiment with [NeMo](https://github.com/NVIDIA/NeMo) ⚡
Result
------
* Model: [QuartzNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#quartznet) is a smaller version of [Jaser](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#jasper)  model  
* I list the word error rate (WER) with and without LM of major ASR tasks.

| Task                   | CER (%) | WER (%) | +LM WER (%) |
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| VIVOS (TEST)            |  6.80 | 18.02 | 15.72 |
| VLSP2018                |  6.87 | 16.26 |  N/A  |
| VLSP2020 T1             | 14.73 | 30.96 |  N/A  |
| VLSP2020 T2             | 41.67 | 69.15 |  N/A  |

Model was trained with ~500 hours Vietnamese speech dataset, was collected from youtube, radio, call center(8k), text to speech data and some public dataset (vlsp, vivos, fpt). It is very small model (13M parameters) make it inference so fast ⚡  

Installation
------------
+ Update & install linux libs:
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
```
+ Install [python>=3.8](https://www.python.org/downloads/release/python-385/)
* Python libs:
```bash
pip install -r requirements.txt
```
+ Install [torch 1.8.1](https://pytorch.org/get-started/previous-versions/#v181):
```bash
# cpu only, you can install CUDA version if you have NVidia GPU
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
+ Install [kemlm](https://github.com/kpu/kenlm) for LM decoding  
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

Transcribe audio file
--------
```bash
python infer.py audio_samples # will transcribe audio file in folder: audio_samples
```
Run web application
--------
* Vietnamese Model ([pretrained](model_vietasr2/checkpoints)): `python flask_upload_record_vn.py`  
* Video demo in Youtube:
   + v1: https://youtu.be/P3mhEngL1us  
   + v2: https://youtu.be/o9NpWi3VUHs  
   
[![Video demo](https://img.youtube.com/vi/P3mhEngL1us/maxresdefault.jpg)](https://youtu.be/P3mhEngL1us)  

* English Model ([pretrained](model_english)): `python flask_upload_record_en.py`  

TODO
------
* Conformer Model
* Data augumentation: speed, noise, pitch shift, time shift,...  
* FastAPI
* Add Dockerfile

Citation
--------
```
  @article{kuchaiev2019nemo,
    title={Nemo: a toolkit for building ai applications using neural modules},
    author={Kuchaiev, Oleksii and Li, Jason and Nguyen, Huyen and Hrinchuk, Oleksii and Leary, Ryan and Ginsburg, Boris and Kriman, Samuel and Beliaev, Stanislav and Lavrukhin, Vitaly and Cook, Jack and others},
    journal={arXiv preprint arXiv:1909.09577},
    year={2019}
  }
```
