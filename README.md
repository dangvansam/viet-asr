# Demo VietASR (NVIDIA NeMo ToolKit)
# Installation
* ctcdecoder, [kemlm](https://github.com/kpu/kenlm) for LM Decode  
`pip install ds-ctcdecoder`
* apex: *https://github.com/NVIDIA/apex*
...
# Result
We list the word error rate (WER) with and without LM of major ASR tasks.

| Task                   | CER (%) | WER (%) | +LM WER (%) |
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| VIVOS (TEST)            |  6.80 | 18.02 | 15.72 |
| VLSP2018                |  6.87 | 16.26 |  N/A  |
| VLSP2020 T1             | 14.73 | 30.96 |  N/A  |
| VLSP2020 T2             | 41.67 | 69.15 |  N/A  |

# Run demo
* vietnamese pretrained model  
`python flask_upload_record_vn.py`  
*Video youtube: https://youtu.be/P3mhEngL1us*  
[![Video demo](https://img.youtube.com/vi/P3mhEngL1us/maxresdefault.jpg)](https://youtu.be/P3mhEngL1us)  
* english pretrained model: `python flask_upload_record_en.py`  
