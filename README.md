## 生データの配置

まずは生のecgデータを配置する必要があります．
`data/ecg`配下にcsvファイルをすべて入れます．

## FileList.csv の作成

続いて， train/valid/test の FileList を作成します．`data/Labels`にアノテーションしたcsvファイルを配置し，
` data/FileList_Maker.ipynb ` を実行すると，`data/FileLists` 配下に FileList_{outcome}.csv が保存されます

## 学習`
学習ようのpythonコードは，CNNモデルはrun.py, Transfomerモデルはrun_tf.pyです．run.sh, run_tf.shはそれぞれを実行するシェルスクリプトで，該当するパラメータを指定して，`./run.sh`で学習が開始します．
