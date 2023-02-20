import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from dataset_zoo.perturbations import get_text_perturb_fn, get_image_perturb_fn
from misc import seed_all, _default_collate, save_scores


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--dataset", default="COCO_Retrieval", type=str, choices=["COCO_Retrieval", "Flickr30k_Retrieval"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--text-perturb-fn", default=None, type=str, help="Perturbation function to apply to the text.")
    parser.add_argument("--image-perturb-fn", default=None, type=str, help="Perturbation function to apply to the images.")
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()

    
def main(args):
    seed_all(args.seed)
    
    model, image_preprocess = get_model(args.model_name, args.device)
    text_perturb_fn = get_text_perturb_fn(args.text_perturb_fn)
    image_perturb_fn = get_image_perturb_fn(args.image_perturb_fn)
    
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=args.download)
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    if text_perturb_fn is not None:
        loader.dataset.text = [text_perturb_fn(t) for t in loader.dataset.text]
    
    scores = model.get_retrieval_scores_dataset(loader)
    result_records = dataset.evaluate_scores(scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Text Perturbation Strategy": args.text_perturb_fn,
                       "Seed": args.seed, "Image Perturbation Strategy": args.image_perturb_fn})
    
    df = pd.DataFrame(result_records)
    output_file = os.path.join(args.output_dir, f"{args.dataset}.csv")
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)
                
    if args.save_scores:
        save_scores(scores, args)

    
if __name__ == "__main__":
    args = config()
    main(args)