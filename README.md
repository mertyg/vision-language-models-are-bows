# When and why vision-language models behave like bags-of-words, and what to do about it? (ICLR 2023 Oral)

[![ICLR2023 Paper](https://img.shields.io/badge/paper-ICLR2023-brightgreen)](https://openreview.net/forum?id=KRLUvxh8uaX)  [![Medium Blog Post](https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg)](https://towardsdatascience.com/your-vision-language-model-might-be-a-bag-of-words-30b1beaef7f8) [![Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1Rmn8CYXRFg4eC458vkBHwAdVKgS03e5D?usp=sharing)


Experiments and data for the paper "When and why vision-language models behave like bags-of-words, and what to do about it?". <br>
This paper got an Oral (notable-top-5%) at ICLR 2023! You can find our camera-ready version [here](https://openreview.net/forum?id=KRLUvxh8uaX).


**Imporant Note**: Thank you for your interest. I apologize for the delay in releasing the code and the camera-ready version, I will do my best to make up for the missing bits as soon as possible. I am currently in Turkey after the devastating [Turkey-Syria earthquake](https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake). Not only me, but also tens of thousands of people lost their families and homes. Please consider [donating](https://ahbap.org/), and at the very least please ask your friends with connections to the regions how they are doing. 

Below we give details about how to easily use our dataset and models, and reproduce our experiments.

# ARO Benchmark
## Visual Genome Relation & Attribution Datasets
It's very easy to use VG-Relation and VG-Attribution datasets. Here's an example:
```python
import clip
from dataset_zoo import VG_Relation, VG_Attribution

model, image_preprocess = clip.load("ViT-B/32", device="cuda")

root_dir="/path/to/aro/datasets"
# Setting download=True will download the dataset to `root_dir` if it's not already there. 
# For VG-R and VG-A, this is a 1GB zip file that is a subset of GQA.

vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
vga_dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=root_dir)

# Do anything with the dataset. Each item will look like this : 
# item = {"image_options": [image], "caption_options": [false_caption, true_caption]}
```

## COCO-Order and Flickr30k-Order Datasets
These datasets require the COCO and Flickr30k retrieval datasets. We provided the interface to download COCO (e.g. set `download=True` in the constructor), however, for Flickr30k, you need to sign up and download it yourself. You can find the Flickr30k retrieval dataset [here](https://forms.illinois.edu/sec/229675).

```python
from dataset_zoo import COCO_Order, Flickr30k_Order

coco_order_dataset = COCO_Order(image_preprocess=preprocess, download=True, root_dir=root_dir) 
flickr_order_dataset = Flickr30k_Order(image_preprocess=preprocess, root_dir=root_dir)
```


# Quick reproducibility
See the notebook in `notebooks/` for a quick way to reproduce some of the results in the paper. We provide a notebook to reproduce the VG-Relation and VG-Attribution datasets [here](notebooks/Replicate%20ARO!%20VG-Relation%2C%20VG-Attribution.ipynb).

## Models
We experiment with a bunch of models here, and let us know if you have any other you would like to add here. You can find BLIP, CLIP, Flava, and XVLM. Please see `model_zoo/` folder for more details. This work is heavily inspired from, and would not be possible without the awesome repos for [BLIP](https://github.com/salesforce/BLIP), [CLIP](https://github.com/openai/CLIP), [Flava](https://huggingface.co/docs/transformers/model_doc/flava), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [XVLM](https://github.com/zengyan-97/X-VLM). A huge, huge thanks to them for open-sourcing their models / implementations! Here's a summary of what we have now: 

Model Name | Model File in this Repo | Repo |
--- | --- | --- |
BLIP | [BLIP implementation](model_zoo/blip_models.py) | https://github.com/salesforce/BLIP |
CLIP | [CLIP implementation](model_zoo/clip_models.py) | https://github.com/openai/CLIP |
Flava | [Flava implementation](model_zoo/flava.py) | https://huggingface.co/facebook/flava-full |
XVLM | [XVLM implementation](model_zoo/xvlm_models.py) | https://github.com/zengyan-97/X-VLM |
NegCLIP | NegCLIP was trained with a fork of the `open_clip` repo. Find the ckpt info [here](model_zoo/__init__.py#L66)| https://github.com/vinid/open_clip |
COCA & CLIP on LAION | We added the usage of the other models in the open_clip repo.| https://github.com/mlfoundations/open_clip |


## ARO Results


## Order-Perturbed Retrieval Results


## NegCLIP Training
We trained the NegCLIP with a fork of the `open_clip` repo. You can find the fork [here](https://github.com/vinid/open_clip). Our modifications are super minor and you will find an detailed description of the main edits [here](https://github.com/vinid/neg_clip#negclip-implementation).

We plan to add support for the distributed setting in the future. However, we trained the model using a single GPU (which is quite a bit of a limitation). Here's the command to reproduce results:
```base
CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --train-data="./mscoco_with_negatives_training.csv" \
    --batch-size=256 \
    --epochs=5 \
    --name="negclip_256_1e-6" \
    --lr=1e-6 \
    --val-data="./mscoco_with_negatives_valid.csv"  \
    --logs="./logs/negCLIP/" \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 14 \
    --warmup 50
```
Note here that `batch_size=256` would result in a matrix of size `512x1024` with negatives.


# Citation
If you use this code or data, please consider citing our paper:

```
@inproceedings{
  yuksekgonul2023when,
  title={When and why Vision-Language Models behave like  Bags-of-Words, and what to do about it?},
  author={Mert Yuksekgonul and Federico Bianchi and Pratyusha   Kalluri and Dan Jurafsky and James Zou},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=KRLUvxh8uaX}
}
```


## TODO
<details>
<summary> Current TODO List.</summary>

| Name | Description | Status |
| --- | --- | --- |
| Add support for distributed training | We trained NegCLIP with a single GPU, and we plan to add support for distributed training in the future. | :white_check_mark: |
| Add negative generation | How to generate negatives for negclip. This could also be on the forked repo. | :white_check_mark: |

</details>

# Contact 
Please let us know if you have further questions or comments. You can reach out to me at `merty@stanford.edu`. 
