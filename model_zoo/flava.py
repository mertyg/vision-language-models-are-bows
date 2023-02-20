import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor


class FlavaWrapper:
    def __init__(self, root_dir, device):
        self.model = FlavaForPreTraining.from_pretrained("facebook/flava-full", cache_dir=root_dir).eval().to(device)
        self.feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full", cache_dir=root_dir)
        self.tokenizer = BertTokenizer.from_pretrained("facebook/flava-full", cache_dir=root_dir)
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full", cache_dir=root_dir)
        self.device = device
    
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = self.tokenizer(text=text, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
            text_feats = self.model.flava.get_text_features(**text_input).cpu().numpy()[:, 0, :]
            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)          
            text_embeds.append(text_feats)   
            
        return np.concatenate(text_embeds, axis=0)
    

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.flava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
            image_embeds.append(image_feats)

        return np.concatenate(image_embeds, axis=0)
    
    
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores
    
    
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        for batch in tqdm(joint_loader):
            
            batch_scores = []
            for i_option in batch["image_options"]:
                im_scores = []
                for c_option in batch["caption_options"]:
                    inputs = self.processor(text=list(c_option), images=list(i_option), return_tensors="pt", padding="max_length", max_length=77, return_codebook_pixels=True, return_image_mask=True).to(self.device)
                    inputs["input_ids_masked"] = inputs["input_ids"].detach().clone() 
                    inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
                    outputs = self.model(**inputs)
                    score = nn.functional.softmax(outputs.itm_logits, dim=-1)[:, 1:2].cpu().numpy()
                    im_scores.append(np.expand_dims(score, -1))
                batch_scores.append(np.concatenate(im_scores, axis=-1))
                
            batch_scores = np.concatenate(batch_scores, axis=1)
            scores.append(batch_scores)
        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores
