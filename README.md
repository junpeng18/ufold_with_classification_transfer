# UFOLD with Classification Transfer Learning
## Structure
```
├───data
│   ├───classifier_data: Rfam data
│   ├───family_data
│   ├───TestSetA
│   ├───TestSetB
│   ├───TrainSetA
│   └───TrainSetB
├───models: save pre-trained model
├───ufold
├───utils
│   ├───download_RNA_family_data.py: download Rfam data and return index dataframe
│   ├───path_utils.py: define working dir
├───Network.py: define necessary neural network (torch.nn.Module)
├───Trainer.py: define Trainer used to train different models
├───preprocess_prob_feat.py: pre-generate the 17-th feature of the image and save to speed up model training
├───*_train_simple.py: simplified version of model training
├───ufold_test_simple.py: simplified version of model testing
```

## Run
1. install necessary packages
- pytorch
- tensorboard
- munch
- numpy
- pandas
- matplotlib
- pathos
- Bio
- tqdm
- p_tqdm
- ...

2. save data in './data' folder
3. run preprocess_prob_feat.py to preprocess rna sequence data
4. run *_train_simple.py as you wish
5. run ufold_test_simple.py for testing