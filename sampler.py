import random
from torch.utils.data.sampler import Sampler


class PositiveSampler(Sampler):
    """Sample each class equally:
    benign, malignant, benign&malignant and negative
    """
    def __init__(self, data_list):
        self.data_list = data_list
        self.generate_label_dict()
    
    def generate_label_dict(self):
        print("Generate label dict")
        labels_indices = {
            "benign": [],
            "malignant": [],
            "benign_and_malignant": [],
            "negative": []
        }

        for index, (acn, d) in enumerate(self.data_list.items()):
            label = d['label']
            is_benign = (label[0] or label[2])
            is_malignant = (label[1] or label[3])
            if is_benign and is_malignant:
                labels_indices["benign_and_malignant"].append(index)
            elif is_benign:
                labels_indices["benign"].append(index)
            elif is_malignant:
                labels_indices["malignant"].append(index)
            else:
                labels_indices["negative"].append(index)
        
        self.labels_indices = labels_indices

        # Number of samples in the least populated category
        # (usually this is malignant)
        self.min_n = min([len(d) for d in self.labels_indices.values()])
    
    def __iter__(self):
        selected_indices = {}
        for cat, indices in self.labels_indices.items():
            random.shuffle(indices)
            selected_indices[cat] = indices[:self.min_n]
        
        sampled_indices = []
        for indices in selected_indices.values():
            sampled_indices.extend(indices)
        random.shuffle(sampled_indices)

        self.sampled_indices = sampled_indices
        return iter(sampled_indices)
    
    def __len__(self):
        return len(self.labels_indices.keys()) * self.min_n


class BPESampler(Sampler):
    """Sample equally studies based on BPE
    
    Assuming that there are 5 BPE categories: (1, 2, 3, 4, unknown)
    they will be sampled equally: we will check which category has
    the least samples and we will reduce the number of samples in
    other categories to match least common category.

    :param data_list: Data list - dictionary
    """

    def __init__(self, data_list):
        self.data_list = data_list
        self.bpe_indices = None
        self.min_n = None

        self.generate_bpe_dict()

    def generate_bpe_dict(self):
        bpe_indices = {}

        for index, (acn, d) in enumerate(self.data_list.items()):
            bpe_cat = d['bpe']
            if bpe_cat in ['NOT FOUND', 'TOO MANY', 'unknown']:
                bpe_cat = 'unknown'
            if bpe_cat not in bpe_indices:
                bpe_indices[bpe_cat] = []
            bpe_indices[bpe_cat].append(index)
        
        self.bpe_indices = bpe_indices

        # Number of samples in the least populated category
        self.min_n = min([len(d) for d in self.bpe_indices.values()])
    
    def __iter__(self):
        selected_indices = {}
        for bpe_cat, indices in self.bpe_indices.items():
            random.shuffle(indices)
            selected_indices[bpe_cat] = indices[:self.min_n]

        # New data list
        sampled_indices = []
        for indices in selected_indices.values():
            sampled_indices.extend(indices)
        random.shuffle(sampled_indices)

        self.sampled_indices = sampled_indices
        return iter(sampled_indices)
    
    def __len__(self):
        return len(self.bpe_indices.keys()) * self.min_n
