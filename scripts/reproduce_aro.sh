model=openai-clip:ViT-B/32 # Choose the model you want to test

for dataset in VG_Relation VG_Attribution COCO_Order Flickr30k_order
do
    python3 main_aro.py --dataset=$dataset --model-name=$model --device=cuda
done
