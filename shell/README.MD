
# Scripts 

Before running any script, please run [check_data.py](check_data.py) to make sure all the [data](../data) needed per dataset is in place. 

```bash
python check_data --dataset mv-test3k
python check_data --dataset mv-test1k
python check_data --dataset msvd
python check_data --dataset vatex
python check_data --dataset tgif
```

## Training from scratch

| Method | Dataset |  Script | Estimated runtime (hrs) on a 2080Ti GPU card |
| :-----| :-----| :-----| -----:|
| LAFF-ml | MV-test3k | [do_laffml_mvtest3k.sh](do_laffml_mvtest3k.sh) |7h|
| LAFF | MV-test3k | [do_laff_mvtest3k.sh](do_laff_mvtest3k.sh) |5h|



## Inference with trained models

download the pretrained model for LAFF-ml(MV-test3k). 

BaiduYun link: https://pan.baidu.com/s/1iTrBQSs1up1rX_QL4qSDdw?pwd=ajfn 

GoogleDrive link: https://drive.google.com/drive/folders/1A2f3qtBsNd1cbOkwOke0NpjAuItyM6K9?usp=sharing

```shell
python predictor.py --device 1 msrvtt1kAtest FrameLaff_NoFrameFc_StrongCLIP_adjust-model_best.pth.tar msrvtt10ktrain/msrvtt10kval/laff --rootpath ~/VisualSearch --batch_size 512 --query_sets msrvtt1kAtest.caption.txt --overwrite 1
```

