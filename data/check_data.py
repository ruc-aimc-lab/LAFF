import os
import argparse


ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')


def parse_args():
    parser = argparse.ArgumentParser('check data')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('dataset', type=str,
                        help='test dataset')
    args = parser.parse_args()
    return args


opt = parse_args()

dataset_dict = {
    "mv-test3k": ["msrvtt10k"+each for each in ["train", "val", "test"]],
    "mv-test1k": ["msrvtt1kA"+each for each in ["train", "val", "test"]],
    "msvd": ["msvd"+each for each in ["train", "val", "test"]],
    "tgif": ["tgif"+each for each in ["train", "val", "test"]],
    "vatex": ["vatex_train", "vatex_val1k5", "vatex_test1k5"]
}

root_path = opt.rootpath


for dataset in dataset_dict[opt.dataset]:
    # check FeatureData
    FeatureDataPath = os.path.join(root_path, dataset, "FeatureData")
    for each in ["X3D_L", "HowTo100M_TimeSformer_divST_96x4_224","mean_irCSN_152_ig65m_from_scratch", "clip_finetune_8frame_uniform_1103"]:
        resume_file = os.path.join(FeatureDataPath, each)
        if not os.path.exists(resume_file):
            raise Exception ("%s not exist" % (resume_file))

    resume_file = os.path.join(FeatureDataPath, "frame", "clip_finetune_8frame_uniform_1103")
    if not os.path.exists(resume_file):
        raise Exception ("%s not exist" % (resume_file))

    # check TextFeatureData
    resume_file = os.path.join(root_path, "TextFeatureData", "clip_finetune_8frame_uniform_1103")
    if not os.path.exists(resume_file):
        raise Exception ("%s not exist" % (resume_file))