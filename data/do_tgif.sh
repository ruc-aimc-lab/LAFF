data_path=${HOME}/VisualSearch

if [ ! -d "$data_path" ]; then
    mkdir $data_path
fi

unzip -q msrvtt.zip -d $data_path
base='tgif'

for split in "${base}train" "${base}val" "${base}test"
do
    mkdir -p ${data_path}/${split}
    ln -s ${data_path}/${base}/FeatureData/ ${data_path}/${split}/FeatureData
    ln -s ${data_path}/${base}/FrameFeatureData/ ${data_path}/${split}/FeatureData/frame
    ln -s ${data_path}/${base}/TextData/ ${data_path}/${split}/TextData
    ln -s ${data_path}/${base}/VideoSets/ ${data_path}/${split}/VideoSets
done
