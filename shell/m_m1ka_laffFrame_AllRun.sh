#!/bin/sh
source activate
conda activate hf-torch

devices=(0)
# *************************萌萌哒******************************
# 实验超参
rootpath="$HOME/hf_code/VisualSearch"
trainCollections=("msrvtt10ktrain" "msrvtt1kAtrain")
valCollections=("msrvtt10kval" "msrvtt1kAval")
val_set='no'
testCollections=("msrvtt10ktest" "msrvtt1kAtest")

config='FrameLaff_NoFrameFc_StrongCLIP_adjust'
batch_size=64
overwrite=0


random_seeds=(2)  # 初始化随机数种子
#parm_adjust_configs=(1)


parm_adjust_configs=(0_7_1_12_0_12_0)

echo ${parm_adjust_configs[*]}

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_${config}.txt"


for index in 0 1
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
