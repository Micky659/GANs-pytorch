import argparse
import time

import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.constants import *
import utils.utils as utils


def train_gan(training_config):
    writer = SummaryWriter()
    device = torch.device("cpu")

    # Download MNIST dataset in the directory data
    mnist_data_loader = utils.get_mnist_data_loader(training_config['batch_size'])

    discriminator_net, generator_net = utils.get_gan(device, GANType.CLASSIC.name)
    discriminator_opt, generator_opt = utils.get_optimizers(discriminator_net, generator_net)

    adversarial_loss = nn.BCELoss()
    real_image_gt = torch.ones((training_config['batch_size'], 1), device=device)
    fake_image_gt = torch.zeros((training_config['batch_size'], 1), device=device)

    ref_batch_size = 16
    ref_noise_batch = utils.get_gaussian_latent_batch(ref_batch_size, device)
    discriminator_loss_values = []
    generator_loss_values = []
    img_cnt = 0

    ts = time.time()

    utils.print_training_info_to_console(training_config)
    for epoch in range(training_config['num_epochs']):
        for batch_idx, (real_images, _) in enumerate(mnist_data_loader):
            real_images = real_images.to(device)

            # Train discriminator
            discriminator_opt.zero_grad()

            real_discriminator_loss = adversarial_loss(discriminator_net(real_images), real_image_gt)

            fake_images = generator_net(utils.get_gaussian_latent_batch(training_config['batch_size'], device))
            fake_images_predictions = discriminator_net(fake_images.detach())
            fake_discriminator_loss = adversarial_loss(fake_images_predictions, fake_image_gt)

            discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            discriminator_loss.backward()
            discriminator_opt.step()

            # Train generator
            generator_opt.zero_grad()

            generated_images_prediction = discriminator_net(
                generator_net(utils.get_gaussian_latent_batch(training_config['batch_size'], device)))

            generator_loss = adversarial_loss(generated_images_prediction, real_image_gt)

            generator_loss.backward()
            generator_opt.step()

            # Logging and checkpoint creation
            generator_loss_values.append(generator_loss.item())
            discriminator_loss_values.append(discriminator_loss.item())

            if training_config['enable_tensorboard']:
                writer.add_scalars('Losses/g-and-d', {'g': generator_loss.item(), 'd': discriminator_loss.item()},
                                   len(mnist_data_loader) * epoch + batch_idx + 1)

                if training_config['debug_imagery_log_freq'] is not None and batch_idx % training_config[
                    'debug_imagery_log_freq'] == 0:
                    with torch.no_grad():
                        log_generated_images = generator_net(ref_noise_batch)
                        log_generated_images_resized = nn.Upsample(scale_factor=2, mode='nearest')(log_generated_images)
                        intermediate_imagery_grid = make_grid(log_generated_images_resized,
                                                              nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                        writer.add_image('intermediate generated imagery', intermediate_imagery_grid,
                                         len(mnist_data_loader) * epoch + batch_idx + 1)

            if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                print(
                    f'GAN training: time elapsed = {(time.time() - ts):.2f} [s] | epoch={epoch + 1} | batch= [{batch_idx + 1}/{len(mnist_data_loader)}]')

            # Save intermediate generator images
            if training_config['debug_imagery_log_freq'] is not None and batch_idx % training_config[
                'debug_imagery_log_freq'] == 0:
                with torch.no_grad():
                    log_generated_images = generator_net(ref_noise_batch)
                    log_generated_images_resized = nn.Upsample(scale_factor=2, mode='nearest')(log_generated_images)
                    save_image(log_generated_images_resized,
                               os.path.join(training_config['debug_path'], f'{str(img_cnt).zfill(6)}.jpg'),
                               nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                    img_cnt += 1

            # Save generator checkpoint
            if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config[
                'checkpoint_freq'] == 0 and batch_idx == 0:
                ckpt_model_name = f"Classic_ckpt_epoch_{epoch + 1}_batch_{batch_idx + 1}.pth"
                torch.save(utils.get_training_state(generator_net, GANType.CLASSIC.name),
                           os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

    torch.save(utils.get_training_state(generator_net, GANType.CLASSIC.name),
               os.path.join(BINARIES_PATH, utils.get_available_binary_name()))


if __name__ == "__main__":
    debug_path = os.path.join(DATA_DIR_PATH, 'debug_imagery')
    os.makedirs(debug_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, help="height of content and style images", default=100)
    parser.add_argument("--batch_size", type=int, help="height of content and style images", default=128)

    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging (D and G loss)",
                        default=True)
    parser.add_argument("--debug_imagery_log_freq", type=int, help="log generator images during training (batch) freq",
                        default=100)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=5)
    args = parser.parse_args()

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['debug_path'] = debug_path

    train_gan(config)

# Created by Akshat Kothari (April, 2021)
