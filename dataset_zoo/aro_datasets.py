import os
import json
import subprocess

import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url

from .perturbations import TextShuffler
from .constants import ARO_ROOT, COCO_ROOT, FLICKR_ROOT
from .retrieval import pre_caption


class VG_Relation(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=ARO_ROOT, download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-R dataset.
        download: Whether to download the dataset if it does not exist.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])
        
        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        return result_records



class VG_Attribution(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=ARO_ROOT, download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-A dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Attribution could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", "--output", annotation_file])

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
        
        # Set of attributes in each test case
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies",  "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })
        return result_records




class COCO_Order(Dataset):
    def __init__(self, image_preprocess=None, root_dir=COCO_ROOT, max_words=30, split="test",
                 image_perturb_fn=None, download=False):  
        """
        COCO Order Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: not used; for compatibility.
        download: Whether to download the dataset if it does not exist.
        """
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]

        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        download_url(urls[split],root_dir)
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_root = root_dir
        
        self.test_cases = []
        
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                    
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.image_root, test_case["image"])       
         
        image = Image.open(image_path).convert('RGB')    
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
        
        item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        return item
    
    def download(self):
        import subprocess
        os.makedirs(self.root_dir, exist_ok=True)
        #subprocess.call(["wget", "http://images.cocodataset.org/zips/train2014.zip"], cwd=self.root_dir)
        #subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)
        
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Precision@1": np.mean(correct_mask)}]
        return records


class Flickr30k_Order(Dataset):
    def __init__(self, image_preprocess, split, root_dir=FLICKR_ROOT, max_words=30,
                 *args, **kwargs):  
        """
        image_preprocess: image preprocessing function
        split: 'val' or 'test'
        root_dir: The directory of the flickr30k images. This should contain the `flickr30k-images` directory that \
            contains all the images. 
        """
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        if not os.path.exists(root_dir):
            print("Directory for Flickr30k could not be found!")
            flickr_url = "https://forms.illinois.edu/sec/229675"
            raise RuntimeError(f"You need to manually sign up and download the dataset from {flickr_url} and place it in the `root_dir`.")
        
        download_url(urls[split],root_dir)
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir
        
        self.test_cases = []
        
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.root_dir, test_case["image"])        
        image = Image.open(image_path).convert('RGB')    
        
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
            
        item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        return item
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        result_records = [{"Precision@1": np.mean(correct_mask)}]
        return result_records


def get_visual_genome_relation(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False):
    return VG_Relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download)


def get_visual_genome_attribution(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False):
    return VG_Attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
                   image_perturb_fn=image_perturb_fn, download=download)

def get_coco_order(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False, root_dir=COCO_ROOT, split="test"):
    return COCO_Order(root_dir=root_dir, split=split, image_preprocess=image_preprocess, image_perturb_fn=image_perturb_fn, max_words=max_words, 
                            download=download)

def get_flickr30k_order(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False, root_dir=FLICKR_ROOT, split="test"):
    return Flickr30k_Order(root_dir=root_dir, split=split, image_preprocess=image_preprocess, image_perturb_fn=image_perturb_fn, max_words=max_words, 
                            download=download)

