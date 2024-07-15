# Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning
This is the official implementation of Multistage Contrastive Learning (MCL) proposed in *Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning* on ECCV 2024

## Set up
To get started, create and activate a Conda environment using the provided `environment.yml` file:
```
conda env create -f environment.yml
conda activate fs
```

## Evaluation on MMVP

Here we provide the weights of the CLIP ViT models obtained by MCL as described in Section 5.5 of our main paper.
For evaluating the MCL fine-tuned Vision Transformer (ViT) version of CLIP models on MMVP, refer to `MMVP.ipynb`. The fine-tuned models are available on [OSF](https://osf.io/b79rw/?view_only=c8a6f78611ef432389df7810ed540785). Please download the weights and `tar -xzvf` to the `./weights` directory. 

## Fine-Tuning CLIP with MCL

Begin by downloading the CC12M dataset using [img2dataset](https://github.com/rom1504/img2dataset). Note that we currently do not support the webdataset format. Please download the dataset in the standard image and text file format (`--output_format files`)

To initialize the MCL process, perform inference using a vanilla model and conduct clustering to generate pseudo labels for Stage 1 training:
```
python cluster.py --img-file-path <path to CC12M> --modelarch ViT-L-14 --pretrained openai --stage 0 --num-clusters 10
```
This command saves the cluster centroids, labels, and pseudo labels of Stage 0 into the `./save` directory.

Subsequently, the pseudo labels obtained can be utilized for Stage 1 tuning:
```
torchrun --nproc_per_node 8 main.py \
    --train-data <path to CC12M> \
    --batch-size 1000 \
    --precision amp \
    --workers 16 \
    --MCL-label-path './save/ViT-L-14_0_pseudo_labels.pt' \
    --epochs 20 \
    --pretrained openai \
    --model ViT-L-14 \
    --force-quick-gelu \
    --report-to tensorboard \
    --zeroshot-frequency 1 \
    --dataset-type files \
    --ddp-static-graph \
    --gather-with-grad \
    --lock-text \
    --lr-scheduler const \
    --warmup 600 \
    --lock-image \
    --lock-image-unlocked-groups 6
```
The fine-tuned model will be saved in `./log`. You can then iteratively perform clustering to generate pseudo labels for Stage 2 tuning.

## Adapt MCL to other project

MCL is model-agnostic, and can easily be adapted to other contrastive learning models (e.g., variations of CLIP and SimCLR). Without changing the model's code, you only need to add a **clustering process** (e.g., K-means) to generate the pseudo labels, and add the **feature-aware negative sampling** to sample the negative samples according to the pseudo labels in your training loop. The feature-aware negative sampling mechanism is detailed in ./my_training/data.py at line 587. 

## Acknowledgement
- [Open_clip](https://github.com/mlfoundations/open_clip) The codebase we built upon.
- [MMVP](https://github.com/tsb0601/MMVP) We use MMVP for evaluation.