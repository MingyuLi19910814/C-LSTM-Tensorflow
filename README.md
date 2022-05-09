# Introduction

This repository implements the [C-LSTM](https://arxiv.org/pdf/1511.08630v2.pdf)
for text classification(SST-2) in Tensorflow2 and tested on binary classification and
5-class classification (SST-5) on [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) dataset.

# Contribution
1. Implement with Tensorflow2 and Python3
2. Use pretrained Google-News-300 weights on the Embedding layer
3. Achieved performance very close to the original paper on SST dataset

# Install
```commandline
conda env create -f environment.yml
conda activate tensorflow
```

# Train

## Binary Classification
```commandline
python train.py --dataset SST-2\
                --batch_size 48\
                --num_epochs 80\
                --num_class 2\
                --max_len 48\
                --learning_rate 1e-4
```

## 5-class classification
```commandline
python train.py --dataset SST-5\
                --batch_size 48\
                --num_epochs 80\
                --num_class 5\
                --max_len 48\
                --learning_rate 1e-4
```

# Result

| Dataset | Accuracy of this implementation | Accuracy of original paper |  
|---------|---------------------------------|---------------------------|
| SST-2   | 86.5%                           | 87.8%                           |
| SST-5   | 48.6%                           | 49.2%                           |

# Acknowledgement
* The SST-2 dataset is from [clairett](https://github.com/clairett/pytorch-sentiment-classification).  
* The SST-5 dataset is from [prrao87](https://github.com/prrao87/fine-grained-sentiment)

