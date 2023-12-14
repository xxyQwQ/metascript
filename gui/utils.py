import hashlib
import datetime
import os
import numpy as np
import logging
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('..')
from utils.function import SquarePad, ColorReverse, RecoverNormalize, SciptTyper
from model.generator import SynthesisGenerator

logging.basicConfig(
    level=logging.INFO,
    filename='./gui/gui.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def new_path():
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    hash_value = hashlib.sha256(current_time_str.encode()).hexdigest()
    try:
        os.makedirs(f".cache/{hash_value}")
    except Exception as e:
        logging.error(f"Error creating directory: {e}")
        return None
    return os.path.abspath(f".cache/{hash_value}")

def load_referance(reference_root, index_path='./assets/reference/index.pkl'):
    index = pickle.load(open(index_path, 'rb'))
    for writer in range(len(index)):
        for character in range(len(index[writer])):
            index[writer][character] = os.path.join(reference_root, index[writer][character])
    return index

input_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    ColorReverse(),
    SquarePad(),
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize((0.5,), (0.5,))
])
output_transform = transforms.Compose([
    RecoverNormalize(),
    transforms.Resize((64, 64), antialias=True),
    ColorReverse(),
    transforms.ToPILImage()
])
align_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64), antialias=True),
    ])

class MetaScript:
    def __init__(self, ref_path, model_path) -> None:
        logging.info('Start initing model.')
        self.generator_model = SynthesisGenerator(reference_count=4).cuda()
        self.generator_model.eval()
        self.generator_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
        self.template_list = None
        self.reference_list = load_referance(ref_path)
        with open('./assets/dictionary/character.pkl', 'rb') as file:
            self.character_map = pickle.load(file)
        self.character_remap = {value: key for key, value in self.character_map.items()}
        with open('./assets/dictionary/punctuation.pkl', 'rb') as file:
            self.punctuation_map = pickle.load(file)
        self.punctuation_remap = {value: key for key, value in self.punctuation_map.items()}
    
    def process_reference(self, image_list, path):
        logging.info('Start processing reference.')
        reference_image = [np.array(align_transform(image)) for image in image_list]
        reference_image = np.concatenate(reference_image, axis=1)
        Image.fromarray(reference_image).save(os.path.join(path, 'reference.png'))
        logging.info('Reference image saved to {}'.format(os.path.join(path, 'reference.png')))
        reference = [input_transform(image) for image in image_list]
        reference = torch.cat(reference, dim=0).unsqueeze(0).cuda()
        return reference
    
    def get_random_reference(self):
        logging.info('No upload reference. Start fetching reference.')
        writer = np.random.randint(len(self.reference_list))
        character = np.random.randint(len(self.reference_list[writer]), size=4)
        file_list = [self.reference_list[writer][i] for i in character]
        image_list = [np.array(Image.open(name)) for name in file_list]
        return image_list

    
    def generate(self, target_text, reference, size, width, path):
        script_typer = SciptTyper(size, width)
        logging.info('start generating script')
        for word in tqdm(target_text, desc='generating script'):
            if word in self.character_remap.keys():
                image = Image.open(os.path.join('./assets/character', '{}.png'.format(self.character_remap[word])))
                template = input_transform(image).unsqueeze(0).cuda()
                with torch.no_grad():
                    result, _, _ = self.generator_model(reference, template)
                result = output_transform(result.squeeze(0).detach().cpu())
                script_typer.insert_word(result, type='character')
            elif word in self.punctuation_remap.keys():
                image = Image.open(os.path.join('./assets/punctuation', '{}.png'.format(self.punctuation_remap[word])))
                result = align_transform(image)
                script_typer.insert_word(result, type='punctuation')
            else:
                logging.error('word {} is not supported'.format(word))
                yield False, word
            yield True, script_typer.plot_result_gui()
        result_image = script_typer.plot_result()
        result_image.save(os.path.join(path, 'result.png'))

