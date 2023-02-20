model=openai-clip:ViT-B/32 # Choose the model you want to test

# Deterministic Experiments
for dataset in COCO_Retrieval Flickr30k_Retrieval
do
    python3 main_retrieval.py --dataset=$dataset --model-name=$model --device=cuda
done

# Stochastic Experiments
# The randomness is over the patch/text perturbations, which are different permutations of input patches/words.
seed=0
for dataset in COCO_Retrieval Flickr30k_Retrieval
do
    for text_perturb in shuffle_nouns_and_adj shuffle_allbut_nouns_and_adj shuffle_trigrams shuffle_within_trigrams shuffle_all_words
    do
        python3 main_retrieval.py --dataset=$dataset --model-name=$model --device=cuda --seed=$seed --text-perturb-fn=$text_perturb

    done
done


for dataset in COCO_Retrieval Flickr30k_Retrieval
do
    for image_perturb in shuffle_rows_4 shuffle_patches_9 shuffle_cols_4
    do
        python3 main_retrieval.py --dataset=$dataset --model-name=$model --device=cuda --seed=$seed --image-perturb-fn=$image_perturb
    done
done

