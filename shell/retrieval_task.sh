#!/bin/sh

<<COMMENT
    这个脚本是把训练，生成测试文件，生成 avs检索 结果三部分结合起来，自动进行。
    注意：
        1. 更改模型在 config 文件中更改。
        2. random_seed 是传入 do_train.py 文件的一个参数，如果没有使用可以修改或者删除。
        3. result_file 是存储结果的文件名。
        3. parm_adjust_configs 是传入 config 文件的一个参数，如果没有使用可以修改或者删除。
COMMENT


# *************************萌萌哒******************************
# 并行参数
Nproc=1    # 可同时运行的最大作业数
devices=()
# shellcheck disable=SC2113
function PushQue {    # 将PID压入队列
	Que="$Que $1"
	Nrun=$(($Nrun+1))
}
function GenQue {     # 更新队列
	OldQue=$Que
	Que=""; Nrun=0
	for PID in $OldQue; do
		if [[ -d /proc/$PID ]]; then
			PushQue $PID
		fi
	done
}
function ChkQue {     # 检查队列
	OldQue=$Que
	for PID in $OldQue; do
		if [[ ! -d /proc/$PID ]] ; then
			GenQue; break
		fi
	done
}
function paralle {
    PID=$!
	PushQue $PID
	while [[ $Nrun -ge ${Nproc} ]]; do
		ChkQue
		sleep 1
	done
}


# *************************萌萌哒******************************
path_shell=`pwd`
cd ..
path_w2vvpp=`pwd`

rootpath="$HOME/hf_code/VisualSearch"
trainCollection=""
valCollection=""
val_set=''  # setA
trainCollection2='None'
testCollection=''
txt_feature_task2='no'
config=''
batch_size=128
workers=2
overwrite=0
pretrained_file_path='None'  # 默认没有 pretrained_file
num_epochs=80

random_seeds=(2)  # 初始化随机数种子
save_mean_last=0

#parm_adjust_configs=()

model_prefix_="runs_"
result_file="$path_w2vvpp/result_log/result_${model_prefix_}_${config}.txt"

# 读取输入的参数
GETOPT_ARGS=$(getopt -o l: -al rootpath:,trainCollection:,valCollection:,val_set:,\
testCollection:,config:,batch_size:,overwrite:,devices:,Nproc:,random_seeds:,parm_adjust_configs:,\
model_prefix_:,result_file:,trainCollection2:,save_mean_last:,num_epochs: -- "$@")  # , 后一定不要有空格
eval set -- "$GETOPT_ARGS"
#获取参数
while [ -n "$1" ]
do
        case "$1" in
                --rootpath) rootpath=$2; shift 2;;
                --trainCollection) trainCollection=$2; shift 2;;
                --valCollection) valCollection=$2; shift 2;;
                --val_set) val_set=$2; shift 2;;
                --testCollection) testCollection=$2; shift 2;;
                --config) config=$2; shift 2;;
                --batch_size) batch_size=$(($2)); shift 2;;
                --overwrite) overwrite=$2; shift 2;;
                --devices) devices_temp=$2; shift 2;;
                --Nproc) Nproc=$(($2)); shift 2;;
                --parm_adjust_configs) parm_adjust_configs=$2; shift 2;;
                --random_seeds) random_seeds=$2; shift 2;;
                --pretrained_file_path) pretrained_file_path=$2; shift 2;;
                --model_prefix_) model_prefix_=$2; shift 2;;
                --result_file) result_file=$2; shift 2;;
                 --trainCollection2) trainCollection2=$2; shift 2;;
                 --save_mean_last) save_mean_last=$2; shift 2;;
                 --num_epochs) num_epochs=$(($2)); shift 2;;
                --) break ;;
                *) echo $1,$2; break ;;
        esac
done


echo "result_file：$result_file, devices: ${devices[*]} , config: $config , parm_adjust_configs: ${parm_adjust_configs},
 Nproc: ${Nproc} "

for each in ${devices_temp[*]}
do
    devices[${#devices[@]}]=$each
done

#exit 0


# ****************************************
# 训练
# shellcheck disable=SC2039
device_index=0
for random_seed in ${random_seeds[*]}
do
    for parm_adjust_config in ${parm_adjust_configs[*]}
    do
        model_prefix="${model_prefix_}${parm_adjust_config}_seed_${random_seed}"

        device=${devices[device_index]}
        let device_index="($device_index + 1) % ${#devices[*]}"

        if [[ $Nproc -gt 1 ]]; then
            echo "$trainCollection $valCollection --rootpath $rootpath --config $config --val_set $val_set --model_prefix $model_prefix"

            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --pretrained_file_path $pretrained_file_path\
                --random_seed $random_seed --trainCollection2 $trainCollection2 \
                --save_mean_last ${save_mean_last} --num_epochs ${num_epochs} \
                & paralle
        else
            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --pretrained_file_path $pretrained_file_path\
                --random_seed $random_seed  --trainCollection2 $trainCollection2 \
                --save_mean_last ${save_mean_last} --num_epochs ${num_epochs}
        fi

    done
done
wait


## ******************萌萌哒分割线***********************
# 测试与输出结果
Nproc=1
overwrite=1
batch_size=64
if [[ $trainCollection2 != 'None' ]]
then
    trainCollection="${trainCollection}_${trainCollection2}"
fi
for random_seed in ${random_seeds[*]}
do
    for parm_adjust_config in ${parm_adjust_configs[*]}
    do
        model_prefix="${model_prefix_}${parm_adjust_config}_seed_${random_seed}"
        if [[ $val_set = 'no' ]]
        then
            model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/$model_prefix/model_best.pth.tar
            sim_name=$trainCollection/$valCollection/$config
        else
            model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$val_set/$config/$model_prefix/model_best.pth.tar
            sim_name=$trainCollection/$valCollection/$val_set/$config
        fi

        query_sets=$testCollection.caption.txt

        device=${devices[device_index]}
        let device_index="($device_index + 1) % ${#devices[*]}"
        if [[ $Nproc -gt 1 ]]; then
            python do_predictor.py $testCollection $model_path $sim_name \
                --query_sets $query_sets \
                --rootpath $rootpath  --overwrite $overwrite --device $device \
                --batch_size $batch_size --predict_result_file $result_file --num_workers $workers \
                & paralle
        else
            python do_predictor.py $testCollection $model_path $sim_name \
                --query_sets $query_sets \
                --rootpath $rootpath  --overwrite $overwrite --device $device \
                --batch_size $batch_size --predict_result_file $result_file --num_workers $workers
        fi

    done
done
wait

cd $path_shell