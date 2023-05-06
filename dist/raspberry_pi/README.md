# Raspberry Pi App

## Install lib
```bash
pip install -r requirements.txt
```

## Run demo
```bash
sudo python dist/raspberry_pi/app_gradio.py \
    -c <config_path> \
    -m <model_path> \
    -l <language_model_path>

# Example:
sudo python dist/raspberry_pi/app_gradio.py \
    -c exps/all_data_vlsp/conformer_e84256_d44256_ctc07/config.yaml \
    -m exps/all_data_vlsp/conformer_e84256_d44256_ctc07/epoch_19.pt \
    -b 2
    -l language_model/lm.bin
# App will run on http://127.0.0.1:7860

```