# Few-Shot Table-to-Text Generation with Prototype Memory
Authors: Yixuan Su, Zaiqiao Meng, Simon Baker, and Nigel Collier

## 1. Data and Pre-trained Models:
### (1) [Download Data](https://drive.google.com/file/d/10Q0s6bHP4bhzxurlgrT1XKQ9hzCpPylw/view?usp=sharing)
> unzip the data.zip and replace it with the empty data folder
### (2) [Pre-trained Checkpoints](https://drive.google.com/file/d/1ip8muvfeI5IOFfOc6i-jRRz_BJZ5IsqN/view?usp=sharing)
> unzip the checkpoints.zip and replace it with empty checkpoints folder

## 2. Prototype Selector
### (1) Enviornment Installation: 
> pip install -r prototype_selector_requirements.txt
### (2) Training of Few-Shot-k setting for humans dataset: 
> ./prototype_selector/sh_folder/training/human/human-few-shot-k.sh
### (3) Inference of Few-Shot-k setting for humans dataset:
> ./prototype_selector/sh_folder/inference/human/inference_human-few-shot-k.sh
