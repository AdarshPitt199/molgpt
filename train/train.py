import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--data_path', type=str, default=None,
                        help="path to dataset csv (overrides data_name)", required=False)
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help="column name for SMILES", required=False)
    parser.add_argument('--scaffold_col', type=str, default='scaffold_smiles',
                        help="column name for scaffold SMILES", required=False)
    parser.add_argument('--split_col', type=str, default=None,
                        help="column name for split/source", required=False)
    parser.add_argument('--train_value', type=str, default='train',
                        help="value for training split", required=False)
    parser.add_argument('--val_value', type=str, default='val',
                        help="value for validation split", required=False)
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help="validation fraction if no split column", required=False)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--num_workers', type=int, default=0,
                        help="num workers for dataloader", required=False)

    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(project="lig_gpt", name=args.run_name)

    data_path = args.data_path if args.data_path is not None else 'datasets/' + args.data_name + '.csv'
    data = pd.read_csv(data_path)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    smiles_col = args.smiles_col.lower()
    scaffold_col = args.scaffold_col.lower() if args.scaffold_col else None
    split_col = args.split_col.lower() if args.split_col else None
    train_value = args.train_value.lower() if args.train_value else None
    val_value = args.val_value.lower() if args.val_value else None

    props = [p.lower() for p in args.props] if args.props else []
    num_props = args.num_props if args.num_props > 0 else len(props)

    required_cols = [smiles_col]
    if args.scaffold and scaffold_col:
        required_cols.append(scaffold_col)

    missing_required = [c for c in required_cols if c not in data.columns]
    if missing_required:
        raise ValueError(f"Required columns not found: {missing_required}")

    data = data.dropna(subset=required_cols).reset_index(drop=True)
    data = data[data[smiles_col].astype(str).str.strip() != ""].reset_index(drop=True)

    if num_props > 0:
        missing_props = [p for p in props if p not in data.columns]
        if missing_props:
            raise ValueError(f"Property columns not found: {missing_props}")
        for p in props:
            data[p] = pd.to_numeric(data[p], errors='coerce')
            if data[p].notna().any():
                data[p] = data[p].fillna(data[p].mean())
            else:
                data[p] = data[p].fillna(0.0)

    if split_col and split_col in data.columns:
        split_series = data[split_col].astype(str).str.lower()
        train_data = data[split_series == train_value].reset_index(drop=True)
        val_data = data[split_series == val_value].reset_index(drop=True)
        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("split_col values did not match train/val splits")
    elif 'split' in data.columns:
        if 'moses' in args.data_name:
            train_data = data[data['split'] == 'train'].reset_index(drop=True)
            val_data = data[data['split'] == 'test'].reset_index(drop=True)
        else:
            train_data = data[data['split'] == 'train'].reset_index(drop=True)
            val_data = data[data['split'] == 'val'].reset_index(drop=True)
    elif 'source' in data.columns:
        train_data = data[data['source'] == 'train'].reset_index(drop=True)
        val_data = data[data['source'] == 'val'].reset_index(drop=True)
    else:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(data))
        val_size = max(1, int(len(data) * args.val_frac))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        train_data = data.iloc[train_idx].reset_index(drop=True)
        val_data = data.iloc[val_idx].reset_index(drop=True)

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data[smiles_col]
    vsmiles = val_data[smiles_col]

    # prop = train_data[['qed']]
    # vprop = val_data[['qed']]

    if num_props > 0:
        prop = train_data[props].values.tolist()
        vprop = val_data[props].values.tolist()
    else:
        prop = [0.0] * len(smiles)
        vprop = [0.0] * len(vsmiles)

    if args.scaffold:
        if scaffold_col is None or scaffold_col not in data.columns:
            raise ValueError("Scaffold column not found; provide --scaffold_col or disable --scaffold")
        scaffold = train_data[scaffold_col]
        vscaffold = val_data[scaffold_col]
    else:
        scaffold = [''] * len(smiles)
        vscaffold = [''] * len(vsmiles)

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2,}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    if args.scaffold:
        lens = [len(regex.findall(i.strip()))
                for i in (list(scaffold.values) + list(vscaffold.values))]
        scaffold_max_len = max(lens)
        print('Scaffold max len: ', scaffold_max_len)
    else:
        scaffold_max_len = 0

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    scaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold] 
    vscaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in vscaffold]

    whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    if '<' not in whole_string:
        whole_string.append('<')

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen= scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen= scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)

    print('Vocab size: ', train_dataset.vocab_size)
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=args.num_workers, ckpt_path=f'../cond_gpt/weights/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)

    df.to_csv(f'{args.run_name}.csv', index=False)
