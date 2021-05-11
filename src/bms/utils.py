import torch
import numpy as np
import os
import cv2
from multiprocessing import Pool
import math
from rdkit import Chem
from rdkit.Chem import Draw


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def sec2min(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class FilesLimitControl:
    """Delete files from the disk if there are more files than the set limit.

    Args:
        max_weights_to_save (int, optional): The number of files that will be
            stored on the disk at the same time. Default is 3.
    """
    def __init__(self, max_weights_to_save=3):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)
        if len(self.saved_weights_paths) > self.max_weights_to_save:
            old_weights_path = self.saved_weights_paths.pop(0)
            if os.path.exists(old_weights_path):
                os.remove(old_weights_path)
                print(f"Weigths removed '{old_weights_path}'")


def load_pretrain_model(weights_path, model, device):
    """Load all pretrain model or as many layers as possible.
    """
    old_model = torch.load(weights_path, device)
    new_dict = model.state_dict()
    old_dict = old_model
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                print('Weights {} were not loaded'.format(key))
        else:
            print('Weights {} were not loaded'.format(key))
    return new_dict


def get_file_path(image_id, main_folder='train'):
    return "/workdir/data/bms-molecular-translation/{}/{}/{}/{}/{}.png".format(
        main_folder, image_id[0], image_id[1], image_id[2], image_id
    )


# Source https://gist.github.com/lucaswiman/1e877a164a69f78694f845eab45c381a
def sp_noise(image):
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < .00015] = black
    image[probs > .85] = white
    return image


# Source https://www.kaggle.com/tuckerarrants/inchi-allowed-external-data
# https://www.kaggle.com/stainsby/improved-synthetic-data-for-bms-competition-v3
def noisy_smile(smile, save_path, add_noise=True, crop_and_pad=True):
    """Create synth smile mol images and augment it."""
    mol = Chem.MolFromSmiles(smile)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
    Chem.rdDepictor.SetPreferCoordGen(True)
    d.drawOptions().maxFontSize = 14
    d.drawOptions().multipleBondOffset = np.random.uniform(0.05, 0.2)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().bondLineWidth = 1
    d.drawOptions().additionalAtomLabelPadding = np.random.uniform(0, .2)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    img = d.GetDrawingText()
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # crop
    crop_rows = img[~np.all(img == 255, axis=1), :]
    img = crop_rows[:, ~np.all(crop_rows == 255, axis=0)]
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT,
                             value=255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # noise
    img = sp_noise(img)
    cv2.imwrite(save_path, img)


def generate_synth_images(smiles, img_paths, num_process):
    with Pool(num_process) as p:
        p.starmap(noisy_smile, zip(smiles, img_paths))
