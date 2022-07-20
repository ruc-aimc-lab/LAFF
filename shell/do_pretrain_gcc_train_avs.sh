#!/bin/sh

source activate
conda activate hf-torch


# *************************萌萌哒******************************
# 并行参数
Nproc=1  # 最大同时运行数目
devices=(0)  # 占用多个gpu: (0 1 2 3)
# 实验超参
rootpath="$HOME/hf_code/VisualSearch"

trainCollection1="gcc11train"  # 预训练数据集
valCollection1="gcc11val"

trainCollection2="tgif-msrvtt10k"  # 训练数据集
valCollection2="tv2016train"  # 验证数据集
val_set='no'
config='sea_avs_adjustVisTxt'
batch_size=256
overwrite=0

random_seeds=(2)  # 初始化随机数种子
#parm_adjust_configs=(1)


parm_adjust_configs=()
for item1 in 9
do
    for item2 in 8
    do
        for item3 in 0
        do
            for item4 in 9
        do
            for item5 in 0
        do
            for item6 in 1
        do
#            for item7 in 512 1024
            for item7 in 512
            do
        parm_adjust_configs[${#parm_adjust_configs[@]}]=${item1}_${item2}_${item3}_${item4}
            done
        done
        done
        done

    done
    done
done

echo ${parm_adjust_configs[*]}

#model_prefix_="runs_cased_lr0.05_bertFrozen_"
#model_prefix_="runs_uncased_lr0.05_"
model_prefix_="runs1_"
result_file="result_log/result_${model_prefix_}_${config}.txt"

# GCC pretrain
trainCollection=${trainCollection1}
valCollection=${valCollection1}
only_train=1

bash ./avs_task.sh --rootpath $rootpath --trainCollection $trainCollection  --valCollection $valCollection \
--val_set $val_set \
--config $config  --batch_size $batch_size  --overwrite $overwrite \
--devices "${devices[*]}" --Nproc $Nproc --parm_adjust_configs "${parm_adjust_configs[*]}" \
--model_prefix_ $model_prefix_ --result_file $result_file --only_train ${only_train}


# msrvtt-tgif train
trainCollection=${trainCollection2}
valCollection=${valCollection2}

only_train=0

if [[ $val_set = 'no' ]]
        then
            pretrained_file_path=$rootpath/$trainCollection1/w2vvpp_train/$valCollection1/$config
        else
            pretrained_file_path=$rootpath/$trainCollection1/w2vvpp_train/$valCollection1/$val_set/$config
        fi

val_set='setA'
bash ./avs_task.sh --rootpath $rootpath --trainCollection $trainCollection  --valCollection $valCollection \
--val_set $val_set \
--config $config  --batch_size $batch_size  --overwrite $overwrite \
--devices "${devices[*]}" --Nproc $Nproc --parm_adjust_configs "${parm_adjust_configs[*]}" \
--model_prefix_ $model_prefix_ --result_file $result_file --only_train ${only_train} \
--pretrained_file_path ${pretrained_file_path}

