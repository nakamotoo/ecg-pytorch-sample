This is a PyTorch implementation for training deep learning models for 12-lead ECGs (2D-CNN, 1D-CNN, Transformer)

## Step1: Placing the raw data.

Place the raw csv files of ECG under `data/ecg`．

## Step2: Generate FileList.csv

Next, we should split the data into train/valid/test dataset and generate a FileList．<br>
Place the csv for annotated labels under `data/Labels`，<br>
If you run `data/FileList_Maker.ipynb `, a file named FileList_{outcome}.csv will be generated under `data/FileLists`

## Step3: Training
Run `./run.sh` to train the models!<br>

## Else:
You can add various models to `ecg/models` <br>
`train.py` : training code for CNN models. <br>
`train_tf.py` : training code for Transformer models (with saving the attention maps at the same time). <br>



## 生データの配置

まずは生のecgデータを配置する必要があります．
`data/ecg`配下にcsvファイルをすべて入れます．

## FileList.csv の作成

続いて， train/valid/test の FileList を作成します．`data/Labels`にアノテーションしたcsvファイルを配置し，
` data/FileList_Maker.ipynb ` を実行すると，`data/FileLists` 配下に FileList_{outcome}.csv が保存されます

## 学習`
学習ようのpythonコードは，CNNモデルはrun.py, Transfomerモデルはrun_tf.pyです．run.sh, run_tf.shはそれぞれを実行するシェルスクリプトで，該当するパラメータを指定して，`./run.sh`で学習が開始します．
