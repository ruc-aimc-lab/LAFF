
# 给出 train_collection， caption_name，folder_name 即可生成相应 vocab 文件
cd ..
rootpath="$HOME/hf_code/VisualSearch"
#train_collection='msrvtt10k_object_labels'  # 数据集
train_collection='coco'  # 数据集
#train_collection='msrvtt10k_queryexpansion_train'  # 数据集
caption_name="${train_collection}.caption.txt"
folder_name="vocab"
#train_collection='gcc/gcc11train'
#sub_name='union_vg_unidet_898_Select15'
#caption_name="msrvtt10k.caption.${sub_name}.txt"  # caption 文件名
#folder_name="vocab_${sub_name}"  # vocab 文件夹名

overwrite=1

for encoding in bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding \
    --rootpath $rootpath --overwrite $overwrite --folder_name $folder_name --caption_name $caption_name
done

cd shell

# 创建软连接
#ln -s ${rootpath}/msrvtt10k_object_labels/TextData/${caption_name} ${rootpath}/msrvtt10ktrain/TextData/msrvtt10ktrain.caption.${sub_name}.txt
#ln -s ${rootpath}/msrvtt10k_object_labels/TextData/${folder_name} ${rootpath}/msrvtt10ktrain/TextData/${folder_name}
#
#ln -s ${rootpath}/msrvtt10k_object_labels/TextData/${caption_name} ${rootpath}/msrvtt10kval/TextData/msrvtt10kval.caption.${sub_name}.txt


# gcc
#ln -s /data2/hf/VisualSearch/gcc11train/TextData/gcc11train.caption.vg_ConfidenceRank15.txt