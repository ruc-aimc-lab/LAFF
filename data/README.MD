# Data

## Organization

We use `msrvtt10ktrain` (the training set of `MV-test3k`) to explain how we organize various data including videoids, captions, frame-level/video-level features, text features, *etc*. For a specific 2D CNN feature, say `clip_finetune_8frame_uniform_1103`, its frame-level feature is stored at `msrvtt10ktrain/FeatureData/frame/clip_finetune_8frame_uniform_1103`, while the video-level feature, obtained by mean pooling over frames, is stored at a higher level at `msrvtt10ktrain/FeatureData/clip_finetune_8frame_uniform_1103`.

```
msrvtt10ktrain/
    FeatureData/
        X3D_L
        HowTo100M_TimeSformer_divST_96x4_224
        mean_irCSN_152_ig65m_from_scratch
        clip_finetune_8frame_uniform_1103
        frame/
            clip_finetune_8frame_uniform_1103
    TextFeatureData/
        clip_finetune_8frame_uniform_1103
    TextData/
        msrvtt10ktrain.caption.txt
    VideoSets/
        msrvtt10ktrain.txt
```

## Downloads

| Dataset | File size | Download link | Setup |
| :-----| ----: | :---- | :---- |
| MSR-VTT (MV-test3k) | 1.85 GB | https://pan.baidu.com/s/1ltPXtO7VJl7rDGFBCS5jZA?pwd=hdrn | [do_msrvtt.sh](do_msrvtt.sh) |
| MV-test1k | - | - | [do_mvtest1k.sh](do_mvtest1k.sh) |
| MSVD | 246.4 MB | https://pan.baidu.com/s/1ycrWqs-pGa4TJUUrOPWTAQ?pwd=qf2z | [do_msvd.sh](do_msvd.sh) |
| TGIF | 3.7 GB | https://pan.baidu.com/s/15b4NHOemWZl8KMSgKZDSWw?pwd=ppuj  [vocab_tgif.zip](vocab_tgif.zip) | [do_tgif.sh](do_tgif.sh) |
| VATEX | 3.0 GB | https://pan.baidu.com/s/1d39e3ABLbM4rfvvdVl5VVQ?pwd=n31h | [do_vatex.sh](do_vatex.sh) |

## Features

We extract visual/text features by publicly available deep models.

| Visual feature | Model | Code | Paper |
| :-----| :-----|  :-----| :-----|
| X3D_L | X3D | https://github.com/facebookresearch/SlowFast | Feichtenhofer *et al.* X3D: Expanding Architectures for Efficient Video Recognition, CVPR 2020。 |
| HowTo100M_TimeSformer_divST_96x4_224 |TF|https://github.com/facebookresearch/TimeSformer|Bertasius *et al.* Is Space-Time Attention All You Need for Video Understanding?  ICML 2021。|
| mean_irCSN_152_ig65m_from_scratch |irCSN|https://github.com/microsoft/computervision-recipes|Ghadiyaram et al. Large-scale weakly-supervised pre-training for video action recognition, CVPR 2019.|
| clip_finetune_8frame_uniform_1103 |CLIP|https://github.com/openai/CLIP|Radford et al. Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.|

| Text feature | Model | Code | Paper |
| :-----| :-----|  :-----| :-----|
| clip_finetune_8frame_uniform_1103 |CLIP|https://github.com/openai/CLIP|Radford et al. Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.|

