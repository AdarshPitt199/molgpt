import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re
import math

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen or 0
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, prop = self.data[idx], self.prop[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = None
        if self.sca is not None:
            scaffold = self.sca[idx].strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2,}|[0-9])"
        regex = re.compile(pattern)

        smiles_tokens = regex.findall(smiles)
        if len(smiles_tokens) > self.max_len:
            smiles_tokens = smiles_tokens[:self.max_len]
        elif len(smiles_tokens) < self.max_len:
            smiles_tokens = smiles_tokens + ['<'] * (self.max_len - len(smiles_tokens))

        if scaffold is not None and self.scaf_max_len > 0:
            scaffold_tokens = regex.findall(scaffold)
            if len(scaffold_tokens) > self.scaf_max_len:
                scaffold_tokens = scaffold_tokens[:self.scaf_max_len]
            elif len(scaffold_tokens) < self.scaf_max_len:
                scaffold_tokens = scaffold_tokens + ['<'] * (self.scaf_max_len - len(scaffold_tokens))
        else:
            scaffold_tokens = []

        dix =  [self.stoi[s] for s in smiles_tokens]
        sca_dix = [self.stoi[s] for s in scaffold_tokens]

        if self.scaf_max_len > 0:
            sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        else:
            sca_tensor = torch.zeros(1, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        if isinstance(prop, (list, np.ndarray)):
            prop = torch.tensor(prop, dtype=torch.float)
        else:
            prop = torch.tensor([prop], dtype=torch.float)
        return x, y, prop, sca_tensor
