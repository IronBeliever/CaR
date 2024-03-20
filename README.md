<h1 align="center">Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation</h1>
<!-- Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation -->
<h4 align="center"> Yuan Ge, Yilun Liu, Chi Hu, Weibin Meng, Shimin Tao, Xiaofeng Zhao, Hongxia Ma, Li Zhang, Hao Yang, Tong Xiao</h4>

<p align="center">
    <img src="alpacar.png" width="20%"> <br>
    Our Model "AlpaCaR" is pronounced as "/ˈælpəˈkɑːr/". The logo is generated by <a href="https://chat.openai.com">DALL·E 3</a>.
</p>

## News💡
- [2024.02] We release our 📄<a href="https://arxiv.org/abs/2402.18191">paper</a>. If you have any questions about our project, please send email to geyuanqaq@gmail.com

## Quick Installation ⚙️
```bash
conda create --name car python=3.8
conda activate car
pip install poetry
poetry install
```

## Usage 🛠

### Ranking

> Download IQS or Comet model from Huggingface <a href="https://huggingface.co/GyQAQ/Instruction-quality-scoring">Link</a>, and save it under */CaR/Ranking/lightning_logs/*.

Default setting
```bash
python Ranking/split_IQS.py --batch_size=128
```

Using other instruction file
```bash
python Ranking/split_IQS.py --input='XX.json'
```
> 'XX.json' needs to be in the format of 'alpaca_data.json'.


### Clustering

Default setting
```bash
python Clustering/cluster.py
```

Using other instruction file with score
```bash
python Clustering/cluster.py --input='XX.json'
```
> 'XX.json' needs to be in the format of './data/ranking_IQS_data.json'.

## Training of Ranking Model 📜

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train --cfg configs/models/{your_model_config}.yaml
```

Specific yaml parameters of IQS 
```bash
instruction_metric:
  class_path: comet.models.InstructionMetric
  init_args:
    nr_frozen_epochs: 0.3
    keep_embeddings_frozen: True
    optimizer: AdamW
    encoder_learning_rate: 1.0e-06
    learning_rate: 1.5e-05
    layerwise_decay: 0.95
    encoder_model: XLM-RoBERTa
    pretrained_model: xlm-roberta-large
    pool: avg
    layer: mix
    layer_transformation: sparsemax
    layer_norm: False
    loss: mse
    dropout: 0.1
    batch_size: 8
    train_data: 
      - data/APE_score_train.csv
    validation_data: 
      - data/APE_score_valid.csv
    hidden_sizes:
      - 2048
      - 1024
    activations: Tanh
      
trainer: ../trainer.yaml
early_stopping: ../early_stopping.yaml
model_checkpoint: ../model_checkpoint.yaml
```
> Training data format of IQS can be found under */CaR/Ranking/data/expert-revised*, and Comet under */CaR/Ranking/data/expert-revised-comet*.

## Citation 
If you find our paper useful, please consider citing:
```bibtex

```
