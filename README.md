## A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction

This repository contains the source code and dataset for the paper: **A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction**. [Zhepei Wei](https://weizhepei.com/), [Jianlin Su](https://spaces.ac.cn/), [Yue Wang](https://ils.unc.edu/~wangyue/), Yuan Tian and [Yi Chang](http://yichang-cs.com/). To appear in ACL 2020. [[arxiv]](https://arxiv.org/abs/1909.03227)

**Update 2020-04-06**: We've just updated the arxiv preprint and it is scheduled to be announced at **Tue, 7 Apr 2020 00:00:00 GMT**. Before that, you can find our latest paper [here](https://weizhepei.com/files/HBT.pdf). 

## Overview

At the core of the proposed HBT framework is the fresh perspective that instead of treating relations as discrete labels on entity pairs, we actually model the relations as functions that map subjects to objects. More precisely, instead of learning relation classifiers f(s,o) -> r, we learn relation-specific taggers f_{r}(s) -> o, each of which recognizes the possible object(s) of a given subject under a specific relation. Under this framework, relational triple extraction is a two-step process: first we identify all possible subjects in a sentence; then for each subject, we apply relation-specific taggers to simultaneously identify all possible relations and the corresponding objects.

![overview](https://weizhepei.com/images/HBT_overview.png)


## Requirements

This repo was tested on Python 3.7 and Keras 2.2.4. The main requirements are:

- tqdm
- codecs
- keras-bert
- tensorflow-gpu == 1.13.1

## Datasets

- [NYT](https://github.com/weizhepei/HBT/tree/master/data/NYT)
- [WebNLG](https://github.com/weizhepei/HBT/tree/master/data/WebNLG)
- [ACE04](https://github.com/weizhepei/HBT/tree/master/data/ACE04)
- [NYT10-HRL](https://github.com/weizhepei/HBT/tree/master/data/NYT10-HRL)
- [NYT11-HRL](https://github.com/weizhepei/HBT/tree/master/data/NYT11-HRL)
- [Wiki-KBP](https://github.com/weizhepei/HBT/tree/master/data/Wiki-KBP)

## Usage

1. **Get pre-trained BERT model for Keras**

   Download Google's pre-trained BERT model **[(`BERT-Base, Cased`)](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**. Then decompress it under `pretrained_bert_models/`. More pre-trained models are available [here](https://github.com/google-research/bert#pre-trained-models).

2. **Build dataset in the form of triples**

   Take the NYT dataset for example: 

   a) Switch to the corresponding directory and download the dataset 

   ```shell
   cd HBT/data/NYT/raw_NYT
   ```

   b) Follow the [instructions]((https://github.com/weizhepei/HBT/tree/master/data/NYT/raw_NYT)) at the same directory, and just run

   ```shell
   python generate.py
   ```

   c) Finally, build dataset in the form of triples

   ```shell
   cd HBT/data/NYT
   python build_data.py
   ```

   This will convert the raw numerical dataset into a proper format for our model and generate `train.json`, `test.json` and `val.json`(if not provided in the raw dataset, it will randomly sample 5% or 10% data from the `train.json` or `test.json` to create `val.json` as in line with previous works). Then split the test dataset by type and num for in-depth analysis on different scenarios of overlapping triples.

3. **Specify the experimental settings**

   By default, we use the following settings in [run.py](https://github.com/weizhepei/HBT/blob/master/run.py):

   ```json
   {
       "bert_model": "cased_L-12_H-768_A-12",
       "max_len": 100,
       "learning_rate": 1e-5,
       "batch_size": 6,
       "epoch_num": 100,
   }
   ```

4. **Train and select the model**

   Specify the running mode and dataset at the command line

   ```shell
   python run.py ---train=True --dataset=NYT
   ```

   The model weights that lead to the best performance on validation set will be stored in `saved_weights/DATASET/`.

5. **Evaluate on the test set**

   Specify the test dataset at the command line

   ```shell
   python run.py --dataset=NYT
   ```

   The extracted result will be saved in `results/DATASET/` with the following format:

   ```json
   {
       "text": "Tim Brooke-Taylor was the star of Bananaman , an STV series first aired on 10/03/1983 and created by Steve Bright .",
       "triple_list_gold": [
           {
               "subject": "Bananaman",
               "relation": "starring",
               "object": "Tim Brooke-Taylor"
           },
           {
               "subject": "Bananaman",
               "relation": "creator",
               "object": "Steve Bright"
           }
       ],
       "triple_list_pred": [
           {
               "subject": "Bananaman",
               "relation": "starring",
               "object": "Tim Brooke-Taylor"
           },
           {
               "subject": "Bananaman",
               "relation": "creator",
               "object": "Steve Bright"
           }
       ],
       "new": [],
       "lack": []
   }
   ```
