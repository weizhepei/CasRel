We follow the same 5-fold crossvalidation setting as adopted in previous works (Li and Ji, 2014; Miwa and Bansal, 2016; Li et al., 2019) and use the [code](https://github.com/tticoin/LSTM-ER) released by (Miwa and Bansal, 2016) to preprocess the raw XML-style data for fair comparison.

# Requirements

* python3
* perl
* nltk (for stanford pos tagger)
* java (for stanford tools)
* zsh
* task datasets (see below)

# Links to tasks/data sets

* ACE 2004 (https://catalog.ldc.upenn.edu/LDC2005T09)
* ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)
* SemEval 2010 Task 8 (https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)

Please note that ACE corpora are not free.

# Usage

## download Stanford Core NLP & POS tagger

```
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

## copy and convert each corpus 

Please set the environment variables for the directories, or directly put the directories in the following commands beforehand.

### ACE 2004

```
cp -r ${ACE2004_DIR}/*/english ace2004/
cd ace2004
zsh run.zsh
cd ..
```

### ACE 2005

```
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
zsh run.zsh
cd ..
```

### SemEval 2010 Task 8

```
cp ${SEMEVAL_TRAIN_DIR}/TRAIN_FILE.TXT semeval-2010/
cp ${SEMEVAL_TEST_DIR}/TEST_FILE.txt semeval-2010/
cd semeval-2010/
zsh run.zsh
cd ..
```
