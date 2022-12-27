# Songs_Year_Classification

## SVM, KNN, ANN for songs-years classification

Final project for IO407-1 @ SJTU.

### requirements

```
torch
tensorboard
sklearn
numpy
joblib
```

### data

Download YearPredictionMSD dataset from https://www.kaggle.com/datasets/ryanholbrook/dl-course-data?select=songs.csv

Make ./data in your root path and put songs.csv in ./data.

```
data
├── songs.csv               <- manual put here
├── data_dict_2000+200.pkl  <- these pkl will be generated
├── data_dict_1000+100.pkl
└── ...
```

The dataset script is in lib/dataset/Songs.py,
everytime using this dataset with different train_size+test_size,
a pkl cache file will generate in ./data.

You can specify the size of the data set adding the following instructions when start running:

```
--train_size xxxx --test_size xxx
```

if you want to use the whole dataset just add(only ANN support):

```
-bd
```

The default setting is the best performance on 2000+200, you can set hyperparameters in terminal.

### SVM

#### run:

```shell
python train_SVM_songs.py -c 13.0 -kt rbf -t cat
```

Models and relevant information will be saved into ./exp/SVM

### KNN

#### run:

```shell
python train_KNN_songs.py -k 45 -t cat
```

Models and relevant information will be saved into ./exp/KNN

### ANN

#### baseline:

```shell
python train_ANN_songs.py -m baseline -t avg
```

#### Songs:

```shell
python train_ANN_songs.py -m Songs -e 300 -b 200 -lr 0.001 -ds 50 -dg 0.5
```

Models and relevant information will be saved into ./exp/ANN

### Results

All results has been saved in ./results/SVM_results.csv ./results/KNN_results.csv and ./results/ANN_results.csv,
you can delete the folder to run your own results or run directly(it will continue writing your results to the original
file).

### Utils

#### clean dummy exp

This scripts can clean all the dummy exp whose results not saved in .csv

```shell
python lib/utils/clean_dummy_exp.py
```

#### tensorboard(only for ANN)

Tensorboard is available in this project to supervise training in real time
or review the training process.