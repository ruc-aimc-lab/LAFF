rootpath=$HOME/VisualSearch
overwrite=0

#./do_eval.sh iacc.3 tv16 tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0
#./do_eval.sh iacc.3 tv17 tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0
#./do_eval.sh iacc.3 tv18 tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 testCollection topic_set sim_name"
    exit
fi

test_collection=$1
topic_set=$2
sim_name=$3

score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
echo $score_file

bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite

