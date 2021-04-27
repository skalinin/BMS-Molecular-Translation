import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager, Process
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import random

from bms.utils import noisy_smile

RDLogger.DisableLog('rdApp.*')

tqdm.pandas()

N_SAMPLES = 1000000
NUM_PROCESS = 8


def rnd_smiles_generator(n_samples, tokens):
    smiles = []
    for i in range(n_samples):
        found = False
        while not found:
            rnd_smile = random.choices(tokens, k=random.randint(5, 15))
            rnd_smile = ''.join(rnd_smile)
            try:
                mol = Chem.MolFromSmiles(rnd_smile)
                d = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
                d.DrawMolecule(mol)
            except:
                mol = None
            if mol is not None:
                smiles.append(rnd_smile)
                found = True
    return smiles


def worker(process_num, num_smiles, tokens, return_dict):
    smiles = rnd_smiles_generator(num_smiles, tokens)
    return_dict[process_num] = smiles


def preprocess_data():
    save_path = '/workdir/data/Synth_Smiles_by_BMS_tokens/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer = torch.load('/workdir/data/processed/tokenizer.pth')
    tokens = list(tokenizer.token2idx.keys())
    tokens.remove('<sos>')
    tokens.remove('<eos>')

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    num_smiles = int(N_SAMPLES / NUM_PROCESS)
    for i in range(NUM_PROCESS):
        p = Process(target=worker, args=(i, num_smiles, tokens, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    smiles = []
    for i in range(NUM_PROCESS):
        for smile in return_dict.values()[i]:
            smiles.append(smile)

    data_csv = pd.DataFrame(smiles, columns=['Smile'])
    data_csv['index_col'] = data_csv.index
    data_csv['image_path'] = data_csv.progress_apply(
        lambda x: os.path.join(save_path, f"train_{x['index_col']}.png"), axis=1)

    with Pool(NUM_PROCESS) as p:
        p.starmap(
            noisy_smile,
            zip(data_csv['Smile'].values,
                data_csv['image_path'].values)
        )

    data_csv.to_csv('/workdir/data/Synth_Smiles_by_BMS_tokens/train_processed.csv')


if __name__ == '__main__':
    preprocess_data()
