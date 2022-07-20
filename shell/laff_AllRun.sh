#!/bin/sh
source activate
conda activate hf-torch

devices=(1)
# *************************萌萌哒******************************
# 实验超参
rootpath="$HOME/VisualSearch"
trainCollections=("msrvtt10ktrain" "msvdtrain" "msrvtt1kAtrain" "tgiftrain" "vatex_train")
valCollections=("msrvtt10kval" "msvdval" "msrvtt1kAval" "tgifval" "vatex_val1k5")
val_set='no'
testCollections=("msrvtt10ktest" "msvdtest" "msrvtt1kAtest" "tgiftest" "vatex_test1k5")

config='laff'
batch_size=64
overwrite=1


random_seeds=(2)  # 初始化随机数种子
#parm_adjust_configs=(1)


parm_adjust_configs=(0_12_0_12_0_0_1)

echo ${parm_adjust_configs[*]}

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_${config}.txt"


for index in 0 1 2 3 4
do

  trainCollection=${trainCollections[${index}]}
  valCollection=${valCollections[${index}]}
  val_set='no'
  testCollection=${testCollections[${index}]}
  bash ./retrieval_task.sh --rootpath $rootpath --trainCollection $trainCollection  --valCollection $valCollection \
  --val_set $val_set --testCollection $testCollection \
  --config $config  --batch_size $batch_size  --overwrite $overwrite \
  --devices "${devices[*]}" --Nproc 1 --parm_adjust_configs "${parm_adjust_configs[*]}" \
  --model_prefix_ $model_prefix_ --result_file $result_file --random_seeds "${random_seeds[*]}"

done
