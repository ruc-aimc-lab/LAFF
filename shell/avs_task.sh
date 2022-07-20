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
testCollection=''
txt_feature_task2='no'
config=''
batch_size=128
workers=2
overwrite=0
pretrained_file_path='None'  # 默认没有 pretrained_file
only_train=0
save_mean_last=0
trainCollection2='None'

random_seeds=(2)  # 初始化随机数种子


#parm_adjust_configs=()

model_prefix_="runs_"
result_file="$path_w2vvpp/result_log/result_${model_prefix_}_${config}.txt"

# 读取输入的参数
GETOPT_ARGS=$(getopt -o l: -al rootpath:,trainCollection:,valCollection:,val_set:,\
testCollection:,config:,batch_size:,overwrite:,devices:,Nproc:,random_seeds:,\
pretrained_file_path:,parm_adjust_configs:,\
model_prefix_:,result_file:,save_mean_last:,workers:,trainCollection2:,only_train: -- "$@")  # , 后一定不要有空格
eval set -- "$GETOPT_ARGS"
#获取参数
while [ -n "$1" ]
do
        case "$1" in
                --rootpath) rootpath=$2; shift 2;;
                --trainCollection) trainCollection=$2; shift 2;;
                --trainCollection2) trainCollection2=$2; shift 2;;
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
                --only_train) only_train=$(($2)); shift 2;;
                --save_mean_last) save_mean_last=$(($2)); shift 2;;
                --workers) workers=$(($2)); shift 2;;
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

        if [[ ${pretrained_file_path} != 'None' ]]; then
            pretrained_file_path_model="${pretrained_file_path}/${model_prefix}/model_best.pth.tar"
        else
            pretrained_file_path_model='None'
        fi

#        echo "let us train ${trainCollection2}"
        device=${devices[device_index]}
        let device_index="($device_index + 1) % ${#devices[*]}"


        if [[ $Nproc -gt 1 ]]; then
            echo "$trainCollection $valCollection --rootpath $rootpath --config $config --val_set $val_set --model_prefix $model_prefix"

            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --pretrained_file_path ${pretrained_file_path_model} \
                --random_seed $random_seed --save_mean_last ${save_mean_last} --trainCollection2 ${trainCollection2} \
                & paralle
        else
            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --pretrained_file_path ${pretrained_file_path_model} \
                --random_seed $random_seed --save_mean_last ${save_mean_last} --trainCollection2 ${trainCollection2}
        fi

    done
done

if [[ ${only_train} = 1 ]]
then
    exit 0
fi
wait

## ******************萌萌哒分割线***********************
# 测试与输出结果
overwrite=1
echo "overwrite" ${overwrite}
batch_size=1024
# create result_file directory
result_file_path=${path_w2vvpp}/${result_file}
mkdir -p ${result_file_path%/*}  # 创建文件目录以及父目录


model_best_names=('model_best.pth.tar' 'mean_last10.pth.tar')
for model_best_name in ${model_best_names[*]}
do
  for random_seed in ${random_seeds[*]}
  do
      for parm_adjust_config in ${parm_adjust_configs[*]}
      do
#          overwrite=0
          model_prefix="${model_prefix_}${parm_adjust_config}_seed_${random_seed}"
          if [[ $val_set = 'no' ]]
          then
              model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/$model_prefix/${model_best_name}
              sim_name=$trainCollection/$valCollection/$config/$model_prefix/${model_best_name}
          else
              model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$val_set/$config/$model_prefix/${model_best_name}
              sim_name=$trainCollection/$valCollection/$val_set/$config/$model_prefix/${model_best_name}
          fi

          # tv16,tv17,tv18
          query_sets=tv16.avs.txt,tv17.avs.txt,tv18.avs.txt
          testCollection=iacc.3
          python do_predictor.py $testCollection $model_path $sim_name \
              --query_sets $query_sets \
              --rootpath $rootpath  --overwrite $overwrite --device $device \
              --batch_size $batch_size --predict_result_file ${result_file_path} --num_workers $workers
          # tv2019 2020 2021
          testCollection=v3c1
          query_sets=tv19.avs.txt,tv20.avs.txt,tv21.avs.txt,trecvid.progress.avs.txt
          python do_predictor.py $testCollection $model_path $sim_name \
              --query_sets $query_sets \
              --rootpath $rootpath  --overwrite $overwrite --device $device \
              --batch_size $batch_size --predict_result_file ${result_file_path} --num_workers $workers

          ## ******************萌萌哒分割线***********************
          # 评估表现
          overwrite=1
          cd tv_avs_eval

          # shellcheck disable=SC2046
          echo -e "\n" $(date "+%Y-%m-%d %H:%M:%S") "\t\c" | tee -a "${result_file_path}"  # 输出到文件
          echo -e  ${model_path} ${parm_adjust_config//_/"\t"} "\t\c" | tee -a "${result_file_path}"

          # tv16, 17, 18
          for topic_set in "tv16" 'tv17' 'tv18'
          do
              test_collection="iacc.3"
              score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
              echo $score_file

              python txt2xml.py $test_collection $score_file --edition $topic_set --priority 1 \
                  --etime 1.0 --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite

              python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection \
              --edition $topic_set \
              --overwrite $overwrite | tee -a "${result_file_path}"

#              rm -rf $rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
          done


          # tv19
          for topic_set in 'tv19' 'tv20' 'tv21' 'trecvid.progress'
          do
              test_collection="v3c1"
              score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
              echo $score_file

              python txt2xml.py $test_collection $score_file --edition $topic_set --priority 1 \
                  --etime 1.0 --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite

              python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection \
              --edition $topic_set \
              --overwrite $overwrite | tee -a "${result_file_path}"


  #            rm -rf $rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
          done

          cd ..
      done
  done
done
