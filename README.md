# CMR
Code repository for SIGIR 2024 paper 
"[Contrast then Memorize: Semantic Neighbor Retrieval-Enhanced Inductive Multimodal Knowledge Graph Completion](https://dl.acm.org/doi/abs/10.1145/3626772.3657838)".

The paper is available at [arXiv](https://arxiv.org/pdf/2407.02867) or [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3626772.3657838).

In this paper, 
we study the inductive multimodal knowledge graph completion (IMKGC) task. 
We propose to 1) unified cross-modal contrastive learning, 2) knowledge representation memorization, and 3) semantic-neighbor retrieval-enhanced inference, for effective IMKGC.

## Requirements
* python>=3.9
* torch>=1.12
* transformers>=4.24
* faiss>=1.7

All experiments are run with 1 or 2 A6000 (48GB) GPUs (usually, the larger batch size brings better performance for contrastive learning).

## Dataset
You can download the FB15k237_ind and WN18RR_ind datasets from [BLP](https://github.com/dfdazac/blp). 
The WN9_ind is the inductive version of WN9 with the `inductive_split.py` from BLP.

The entity descriptions are from [KG-BERT](https://github.com/yao8839836/kg-bert).

The entity images are from [MMKB](https://github.com/mniepert/mmkb), [RSME](https://github.com/wangmengsd/RSME), or [MKGformer](https://github.com/zjunlp/MKGformer).

Since some entities from the above sources have no image, we manually crawl their images from 1) [Wikidata](https://www.wikidata.org), 2) Google/Bing Search their names with the help of [icrawler](https://github.com/hellock/icrawler).

The PLM is `bert-base-uncased` and `vit-base-patch16-224` from [huggingface](https://huggingface.co/). 

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

### Preprocess

Put the dataset in `./data/TASK_NAME` directory and images in `./img_data/TASK_NAME` directory. 

Put the downloaded PLMs in `./PLMs/` directory.

Then preprocess the dataset.
```
python preprocess.py 
```

### Training
Train the model with the following script, taking FB15K237_ind dataset as an example. 
We put our default parameters in `config.py`. 
```
CUDA_VISIBLE_DEVICES='0' python main.py --model-dir ./ckpt/fb15k237/ --pretrained-model ./PLMs/bert-base-uncased --pooling mean --lr 1e-5 --use-link-graph --task FB15k237_ind --batch-size 768 --print-freq 20 --additive-margin 0.0 --finetune-t --pre-batch 2 --epochs 50 --workers 4 --max-to-keep 5 --use-amp --mm --prefix 4
```

### Evaluation
Evaluate the trained model with the following script:
```
CUDA_VISIBLE_DEVICES='0' python evaluate.py --is-test --eval-model-path ./ckpt/fb15k237/YOUR_MODEL_DIR/model_best.mdl --task FB15k237_ind --pretrained-model ./PLMs/bert-base-uncased --batch-size 1024 --mm --prefix 4 --knn_topk 32 
```

## Citation

Please cite our paper as follows if you find our work useful.

```
@inproceedings{zhao2024CMR,
author = {Zhao, Yu and Zhang, Ying and Zhou, Baohang and Qian, Xinying and Song, Kehui and Cai, Xiangrui},
title = {Contrast then Memorize: Semantic Neighbor Retrieval-Enhanced Inductive Multimodal Knowledge Graph Completion},
year = {2024},
booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {102–111},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```
## Acknowledgement
Our code is modified based on [SimKGC](https://github.com/intfloat/SimKGC). We would like to appreciate their open-sourced work!
