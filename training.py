import os
import sys
import time

import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.dataset import CharacterDataset
from utils.function import hinge_loss, plot_sample
from model.generator import SynthesisGenerator
from model.discriminator import MultiscaleDiscriminator


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    # load configuration
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    learning_rate = float(config.parameter.learning_rate)
    reference_count = int(config.parameter.reference_count)
    weight_adversarial = float(config.parameter.loss_function.weight_adversarial)
    weight_structure = float(config.parameter.loss_function.weight_structure)
    weight_style = float(config.parameter.loss_function.weight_style)
    weight_reconstruction = float(config.parameter.loss_function.weight_reconstruction)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.train()

    discriminator_model = MultiscaleDiscriminator().to(device)
    discriminator_model.train()

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)

    # load dataset
    dataset = CharacterDataset(dataset_path, reference_count=reference_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    print('image number: {}\n'.format(len(dataset)))

    # start training
    current_iteration = 0
    current_time = time.time()

    while current_iteration < num_iterations:
        for reference_image, template_image, script_image in dataloader:
            current_iteration += 1

            reference_image, template_image, script_image = reference_image.to(device), template_image.to(device), script_image.to(device)

            # generator
            generator_optimizer.zero_grad()

            result_image, template_structure, reference_style = generator_model(reference_image, template_image)

            prediction_result = discriminator_model(result_image)
            loss_adversarial = 0
            for prediction in prediction_result:
                loss_adversarial += hinge_loss(prediction, positive=True)

            result_structure = generator_model.structure(result_image)
            loss_structure = 0
            for i in range(len(result_structure)):
                loss_structure += 0.5 * torch.mean(torch.square(template_structure[i] - result_structure[i]))

            result_style = generator_model.style(result_image.repeat_interleave(reference_count, dim=1))
            loss_style = 0
            for i in range(len(result_style)):
                loss_style += 0.5 * torch.mean(torch.square(reference_style[i] - result_style[i]))

            loss_reconstruction = 0.5 * torch.mean(torch.square(script_image - result_image))

            loss_generator = weight_adversarial * loss_adversarial + weight_structure * loss_structure + weight_style * loss_style + weight_reconstruction * loss_reconstruction
            loss_generator.backward()
            generator_optimizer.step()

            # discriminator
            discriminator_optimizer.zero_grad()

            prediction_fake = discriminator_model(result_image.detach())
            loss_fake = 0
            for prediction in prediction_fake:
                loss_fake += hinge_loss(prediction, False)

            prediction_real = discriminator_model(script_image)
            loss_true = 0
            for prediction in prediction_real:
                loss_true += hinge_loss(prediction, True)

            loss_discriminator = 0.5 * (loss_true + loss_fake)
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # report
            if current_iteration % report_interval == 0:
                last_time = current_time
                current_time = time.time()
                iteration_time = (current_time - last_time) / report_interval

                print('iteration {} / {}:'.format(current_iteration, num_iterations))
                print('time: {:.6f} seconds per iteration'.format(iteration_time))
                print('discriminator loss: {:.6f}, generator loss: {:.6f}'.format(loss_discriminator.item(), loss_generator.item()))
                print('adversarial loss: {:.6f}, structure loss: {:.6f}, style loss: {:.6f}, reconstruction loss: {:.6f}\n'.format(loss_adversarial.item(), loss_structure.item(), loss_style.item(), loss_reconstruction.item()))

            # save
            if current_iteration % save_interval == 0:
                save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))
                os.makedirs(save_path, exist_ok=True)

                image_path = os.path.join(save_path, 'sample.png')
                generator_path = os.path.join(save_path, 'generator.pth')
                discriminator_path = os.path.join(save_path, 'discriminator.pth')

                image = plot_sample(reference_image, template_image, script_image, result_image)[0]
                Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
                torch.save(generator_model.state_dict(), generator_path)
                torch.save(discriminator_model.state_dict(), discriminator_path)

                print('save sample image in: {}'.format(image_path))
                print('save generator model in: {}'.format(generator_path))
                print('save discriminator model in: {}\n'.format(discriminator_path))

            if current_iteration >= num_iterations:
                break


if __name__ == '__main__':
    main()
