import os
import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from utils.function import SquarePad, ColorReverse


class CharacterDataset(TensorDataset):
    def __init__(self, data_root, reference_count: int = 1):
        self.script_root = os.path.join(data_root, 'script')
        self.template_root = os.path.join(data_root, 'template')
        self.reference_count = reference_count

        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        for writer in tqdm(writer_list, desc='loading dataset'):
            character_list = glob.glob('{}/*.*g'.format(writer))
            self.script_list.append(character_list)

        self.remap_list = []
        for writer in range(len(self.script_list)):
            for character in range(len(self.script_list[writer])):
                self.remap_list.append((writer, character))

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            ColorReverse(),
            SquarePad(),
            transforms.Resize((128, 128), antialias=True),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        writer, character = self.remap_list[index]
        reference_path = np.random.choice(self.script_list[writer], self.reference_count, replace=True)
        script_path = self.script_list[writer][character]
        template_path = os.path.join(self.template_root, os.path.basename(script_path))

        reference_image = torch.concat([self.transforms(Image.open(path)) for path in reference_path])
        template_image = self.transforms(Image.open(template_path))
        script_image = self.transforms(Image.open(script_path))
        return reference_image, template_image, script_image

    def __len__(self):
        return len(self.remap_list)
