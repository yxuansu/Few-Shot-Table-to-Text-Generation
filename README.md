# Few-Shot Table-to-Text Generation with Prototype Memory
Authors: Yixuan Su, Zaiqiao Meng, Simon Baker, and Nigel Collier

Code for EMNLP 2021 paper [Few-Shot Table-to-Text Generation with Prototype Memory](https://arxiv.org/abs/2108.12516)

## 1. Download Data and Pre-trained Models:
### (1) Download Data [link](https://drive.google.com/file/d/10Q0s6bHP4bhzxurlgrT1XKQ9hzCpPylw/view?usp=sharing)
```yaml
unzip the data.zip and replace it with the empty data folder
```
### (2) Pre-trained Checkpoints [link](https://drive.google.com/file/d/1ip8muvfeI5IOFfOc6i-jRRz_BJZ5IsqN/view?usp=sharing)
```yaml
unzip the checkpoints.zip and replace it with empty checkpoints folder
```

## 2. Prototype Selector
### (1) Enviornment Installation: 
```yaml
pip install -r prototype_selector_requirements.txt
```
### (2) Training of Few-Shot-k setting for humans dataset: 
```yaml
cd ./prototype_selector/sh_folder/training/human/
chmod +x ./human-few-shot-k.sh
./human-few-shot-k.sh
```
### (3) Inference of Few-Shot-k setting for humans dataset:
```yaml
cd ./prototype_selector/sh_folder/inference/human/
chmod +x ./inference_human-few-shot-k.sh
./inference_human-few-shot-k.sh
```

## 3. Generator
### (1) Enviornment Installation: 
```yaml
pip install -r generator_requirements.txt
```
### (2) Training of Few-Shot-k setting for humans dataset: 
```yaml
cd ./generator/training/human/
chmod +x ./human-few-shot-k.sh
./human-few-shot-k.sh
```
### (3) Inference of Few-Shot-k setting for humans dataset:
```yaml
cd ./generator/inference/human/
chmod +x ./human-few-shot-k-inference.sh
./human-few-shot-k-inference.sh
```

## 4. Citation
If you find our paper and code useful, please kindly cite our paper:

```bibtex
@inproceedings{su-etal-2021-shot-table,
    title = "Few-Shot Table-to-Text Generation with Prototype Memory",
    author = "Su, Yixuan  and
      Meng, Zaiqiao  and
      Baker, Simon  and
      Collier, Nigel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.77",
    pages = "910--917",
    abstract = "Neural table-to-text generation models have achieved remarkable progress on an array of tasks. However, due to the data-hungry nature of neural models, their performances strongly rely on large-scale training examples, limiting their applicability in real-world applications. To address this, we propose a new framework: Prototype-to-Generate (P2G), for table-to-text generation under the few-shot scenario. The proposed framework utilizes the retrieved prototypes, which are jointly selected by an IR system and a novel prototype selector to help the model bridging the structural gap between tables and texts. Experimental results on three benchmark datasets with three state-of-the-art models demonstrate that the proposed framework significantly improves the model performance across various evaluation metrics.",
}
```

