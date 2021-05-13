# Demo VietASR (NVIDIA NeMo ToolKit)
# Installation
* ctcdecoder, [kemlm](https://github.com/kpu/kenlm) for LM Decode  
`pip install ds-ctcdecoder`
* apex: *https://github.com/NVIDIA/apex*
...
# Result
We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

| Task                   | CER (%) | WER (%) | Pretrained model|
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| VIVOS (TEST)            | .    | .     | [link](model_vietasr2/) |
| VLSP2018                | .    | .     | [link](model_vietasr2/) |
| VLSP2020 T1             | .    | .     | [link](model_vietasr2/) |
| VLSP2020 T2             | .    | .     | [link](model_vietasr2/) |

# Run demo
* vietnamese pretrained model  
`python flask_upload_record_vn.py`  
*Video youtube: https://youtu.be/P3mhEngL1us*  
[![Video demo](https://img.youtube.com/vi/P3mhEngL1us/maxresdefault.jpg)](https://youtu.be/P3mhEngL1us)  
* english pretrained model: `python flask_upload_record_en.py`  
