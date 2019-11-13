# Duplicate Questions Identifier

## 简介

Quora 重复问题鉴定。

数据集介绍：https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs

数据集下载：http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv

## 运行

1. 修改`config.py`；
2. 安装 TensorFlow 2.0 某一版本（CPU/GPU、alpha/beta/nightly）；
3. `pip install -r requirements.txt`；
4. `python main.py`；
5. 如需使用 TensorBoard，`tensorboard --logdir log`。
