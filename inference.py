import os
import sys
import glob
import pickle

import hydra
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from utils.logger import Logger
from utils.function import SquarePad, ColorReverse, RecoverNormalize
from model.generator import SynthesisGenerator


def fetch_template(target_text):
    with open('./assets/dictionary/dictionary.pkl', 'rb') as file:
        dictionary = pickle.load(file)
    remap = {value: key for key, value in dictionary.items()}
    template_list = []
    for character in tqdm(target_text, desc='fetching template'):
        name = remap.get(character, None)
        if name is None:
            raise ValueError('character {} is not supported'.format(character))
        image = Image.open(os.path.join('./assets/template', '{}.png'.format(name)))
        template_list.append(image)
    return template_list


def fetch_reference(reference_path):
    reference_list = []
    file_list = glob.glob('{}/*'.format(reference_path))
    for file in tqdm(file_list, desc='fetching reference'):
        image = Image.open(file)
        reference_list.append(image)
    return reference_list


def display_images(image_list, cols, overlap=0.1):
    max_width = max(image.width for image in image_list)
    max_height = max(image.height for image in image_list)

    total_rows = len(image_list) // cols + (len(image_list) % cols > 0)

    overlap_pixels_x = int(max_width * overlap)

    big_image = Image.new('RGBA', (cols * (max_width - overlap_pixels_x), total_rows * (max_height)), (255, 255, 255, 0))

    for i in range(total_rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(image_list) and image_list[index] is not None:
                image = image_list[index].convert('RGBA')

                # 计算偏移量，考虑重叠
                offset_x = j * (max_width - overlap_pixels_x)
                offset_y = i * (max_height)

                big_image.paste(image, (offset_x, offset_y), mask=image)

    plt.imshow(big_image)
    plt.axis('off')
    plt.savefig("./result.png")


@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    reference_path = str(config.parameter.reference_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    reference_count = int(config.parameter.reference_count)
    target_text = str(config.parameter.target_text)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'inference.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # create transform
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
        transforms.Resize((128, 128), antialias=True),
        ColorReverse(),
        transforms.ToPILImage()
    ])

    # fetch data
    template_list = fetch_template(target_text)
    reference_list = fetch_reference(reference_path)
    while len(reference_list) < reference_count:
        reference_list.extend(reference_list)
    reference_list = reference_list[:reference_count]

    # generate script
    result_list = []
    reference = [input_transform(image) for image in reference_list]
    reference = torch.cat(reference, dim=0).unsqueeze(0).to(device)
    for image in tqdm(template_list, desc='generating script'):
        template = input_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            result, _, _ = generator_model(reference, template)
        result = output_transform(result.squeeze(0).detach().cpu())
        result_list.append(result)

    display_images(result_list, 10)


if __name__ == '__main__':
    main()
