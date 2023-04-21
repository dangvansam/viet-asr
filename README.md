# VIETASR - Vietnamese Automatic Speech Recognition

### A Vietnamese Automatic Speech Recognition (Speech to Text) Toolkit

### Installation
+ [Python](https://www.python.org/downloads) >= 3.8  
+ Install [pytorch](https://pytorch.org/get-started/previous-versions):
    ```bash
    # CPU version
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

    # CUDA version
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ```
+ Install python libs:

    ```bash
    pip install -r requirements.txt
    ```
+ Install [kenlm](https://github.com/kpu/kenlm) for beam search decode (Optional)
    ```bash
    # Only support Linux
    pip install https://github.com/kpu/kenlm/archive/master.zip
    ```

### Prepair Data

Perform on VIVOS dataset
1. Create meta file:
```bash
# Usage
python preprocessing/preprocess_vivos.py \
    --dataset_dir=<path_to_vivos_data_dir> \
    --subset=<train_or_test_set> \
    --save_filepath=<save_meta_filepath>

# Example:
# VIVOS train set
python preprocessing/preprocess_vivos.py \
    --dataset_dir=D:/data/VIVOS \
    --subset=train \
    --save_filepath=data/data.txt

# VIVOS test set
python preprocessing/preprocess_vivos.py \
    --dataset_dir=D:/data/VIVOS \
    --subset=test \
    --save_filepath=data/test.txt
```

2. Split data for valid set (Optional)
```bash
# Usage
python preprocessing/split_train_valid.py \
    --input_filepath=<path_to_meta_file> \
    --output_dir=<out_splited_meta_files>

# Example: split VIVOS train set to train and valid set
python preprocessing/split_train_valid.py \
    --dataset_dir=data/data.txt \
    --output_dir=data
```
### Training Model
To train a new model, use the below script:
```bash
# Usage
python train.py \
    --config=<path_to_config> \
    --output_dir=<training_output_dir> \
    --device=<training_device_cpu_or_cuda>

# Example:
python train.py \
    --config=config/conformer.yaml \
    --output_dir=exps/vivos/conformer_v1 \
    --device=cpu
```
### Testing Model
To test the model when training is done, use the below script:
```bash
# Usage
python test.py \
    --config=<path_to_config> \
    --test_meta_filepath=<path_to_test_meta_file> \
    --model_path=<path_to_model_trained> \
    --beam_size=<beam_width_beamsearch_decode> \
    --kenlm_path=<language_model_beamsearch_decode> \
    --kenlm_alpha=<lm_alpha_beamsearch_decode> \
    --kenlm_beta=<lm_beta_beamsearch_decode> \
    --device=<training_device_cpu_or_cuda>

# Example:
python test.py \
    --config=config/conformer.yaml \
    --test_meta_filepath=data/test.txt \
    --model_path=exps/vivos/conformer_v1/checkpoint.pt \
    --beam_size=1 \
    --kenlm_path=languege_model/kenlm_model.bin \
    --kenlm_alpha=0.5 \
    --kenlm_beta=1.5 \
    --device=cpu

# beam_size=1 means greedy decode, otherwise,  use beamsearch decode with language model, so --kenlm_path must be passed.
```
## Transcribe An Audio File
To transcribe an audio file on your computer:
```bash
# Usage
python transcribe.py \
    --config=<path_to_config> \
    --test_meta_filepath=<path_to_test_meta_file> \
    --model_path=<path_to_model_trained> \
    --beam_size=<beam_width_beamsearch_decode> \
    --kenlm_path=<language_model_beamsearch_decode> \
    --kenlm_alpha=<lm_alpha_beamsearch_decode> \
    --kenlm_beta=<lm_beta_beamsearch_decode> \
    --device=<training_device_cpu_or_cuda>

# Example:
python transcribe.py \
    --config=config/conformer.yaml \
    --test_meta_filepath=data/test.txt \
    --model_path=exps/vivos/conformer_v1/checkpoint.pt \
    --beam_size=10 \
    --kenlm_path=languege_model/kenlm_model.bin \
    --kenlm_alpha=0.5 \
    --kenlm_beta=1.5 \
    --device=cpu

# beam_size=1 means greedy decode, otherwise, use beamsearch decode with language model, so --kenlm_path must be passed.
```