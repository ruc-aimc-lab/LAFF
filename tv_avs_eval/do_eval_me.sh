cd tv_avs_eval
conda activate py2-torch1-2
rootpath=$HOME/VisualSearch
overwrite=1

# train w2vvpp on tgif-msrvtt10k based on config "w2vvpp_resnext101-resnet152_subspace"
trainCollection="tgif-msrvtt10k"
valCollection="tv2016train"
val_set="setA"
config="w2vvpp_resnext101-resnet152_subspace"

sim_name="$trainCollection/$valCollection/$val_set/$config"

# tv16, 17, 18
for topic_set in "tv16" 'tv17' 'tv18'
do
    test_collection="iacc.3"
    score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
    echo $score_file

    bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
    python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite
done
# tv19
topic_set="tv19"
test_collection="v3c1"
score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
echo $score_file

bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite

# -------------------------
# 改成 bow+w2v
cd tv_avs_eval
trainCollection="tgif-msrvtt10k"
valCollection="tv2016train"
val_set="setA"
config="w2vvpp_res101_152_bow_w2v"

sim_name="$trainCollection/$valCollection/$val_set/$config"

# tv16, 17, 18
for topic_set in "tv16" 'tv17' 'tv18'
do
    test_collection="iacc.3"
    score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
    echo $score_file

    bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
    python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite
done
# tv19
topic_set="tv19"
test_collection="v3c1"
score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
echo $score_file

bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite

cd ..


# -------------------------
# 公共空间维度由2048改为4096
cd tv_avs_eval
trainCollection="tgif-msrvtt10k"
valCollection="tv2016train"
val_set="setA"
config="w2vvpp_res101_152_me"

sim_name="$trainCollection/$valCollection/$val_set/$config"

# tv16, 17, 18
for topic_set in "tv16" 'tv17' 'tv18'
do
    test_collection="iacc.3"
    score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
    echo $score_file

    bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
    python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite
done
# tv19
topic_set="tv19"
test_collection="v3c1"
score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
echo $score_file

bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite

cd ..
