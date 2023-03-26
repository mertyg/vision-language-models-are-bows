import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as nnf
from torchvision import transforms as T

# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations

def get_text_perturb_fn(text_perturb_fn):
    if text_perturb_fn == "shuffle_nouns_and_adj":
        return shuffle_nouns_and_adj
    elif text_perturb_fn == "shuffle_allbut_nouns_and_adj":
        return shuffle_allbut_nouns_and_adj
    elif text_perturb_fn == "shuffle_within_trigrams":
        return shuffle_within_trigrams
    elif text_perturb_fn == "shuffle_all_words":
        return shuffle_all_words
    elif text_perturb_fn == "shuffle_trigrams":
        return shuffle_trigrams
    elif text_perturb_fn is None:
        return None
    else:
        print("Unknown text perturbation function: {}, returning None".format(text_perturb_fn))
        return None
    
    
def get_image_perturb_fn(image_perturb_fn):
    if image_perturb_fn == "shuffle_rows_4":
        return partial(shuffle_rows, n_rows=4)
    elif image_perturb_fn == "shuffle_patches_9":
        return partial(shuffle_patches, n_ratio=3)
    elif image_perturb_fn == "shuffle_cols_4":
        return partial(shuffle_columns, n_cols=4)
    elif image_perturb_fn is None:
        return None
    else:
        print("Unknown image perturbation function: {}, returning None".format(image_perturb_fn))
        return None
    


class TextShuffler:

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def shuffle_nouns_and_adj(self, ex):

        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']]
        ## Finding adjectives
        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        ## Shuffle the nouns of the text
        text[noun_idx] = np.random.permutation(text[noun_idx])
        ## Shuffle the adjectives of the text
        text[adjective_idx] = np.random.permutation(text[adjective_idx])

        return " ".join(text)

    def shuffle_all_words(self, ex):
        return " ".join(np.random.permutation(ex.split(" ")))


    def shuffle_allbut_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
        ## Finding adjectives

        else_idx = np.ones(text.shape[0])
        else_idx[noun_adj_idx] = 0

        else_idx = else_idx.astype(bool)
        ## Shuffle everything that are nouns or adjectives
        text[else_idx] = np.random.permutation(text[else_idx])
        return " ".join(text)


    def get_trigrams(self, sentence):
        # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
        trigrams = []
        trigram = []
        for i in range(len(sentence)):
            trigram.append(sentence[i])
            if i % 3 == 2:
                trigrams.append(trigram[:])
                trigram = []
        if trigram:
            trigrams.append(trigram)
        return trigrams

    def trigram_shuffle(self, sentence):
        trigrams = self.get_trigrams(sentence)
        for trigram in trigrams:
            random.shuffle(trigram)
        return " ".join([" ".join(trigram) for trigram in trigrams])


    def shuffle_within_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        shuffled_ex = self.trigram_shuffle(tokens)
        return shuffled_ex


    def shuffle_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        trigrams = self.get_trigrams(tokens)
        random.shuffle(trigrams)
        shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
        return shuffled_ex


def _handle_image_4shuffle(x):
    return_image = False
    if not isinstance(x, torch.Tensor):
        # print(f"x is not a tensor: {type(x)}. Trying to handle but fix this or I'll annoy you with this log")
        t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
        t = t.permute(0, 3, 1, 2)
        return_image = True
        return t, return_image
    if len(x.shape) != 4:
        #print("You did not send a tensor of shape NxCxWxH. Unsqueezing not but fix this or I'll annoy you with this log")
        return x.unsqueeze(dim=0), return_image
    else:
        # Good boi
        return x, return_image
        

def shuffle_rows(x, n_rows=7):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-2]//n_rows
    u = nnf.unfold(x, kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def shuffle_columns(x, n_cols=7):
    """
    Shuffle the columns of the image tensor where we'll have n_cols columns.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-1]//n_cols
    u = nnf.unfold(x, kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image



def shuffle_patches(x, n_ratio=4):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size_x = x.shape[-2]//n_ratio
    patch_size_y = x.shape[-1]//n_ratio
    u = nnf.unfold(x, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image