import re

import git
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

from .constants import *
from models.definitions.classic_gan import DiscriminatorNet, GeneratorNet

# Return a new file name for image to save
def get_available_file_name(input_dir):
    def valid_frame_name(str):
        pattern = re.compile(r'[0-9]{6}\.jpg')
        return re.fullmatch(pattern, str) is not None

    valid_frames = list(filter(valid_frame_name, os.listdir(input_dir)))
    if len(valid_frames) > 0:
        last_img_name = sorted(valid_frames)[-1]
        new_prefix = int(last_img_name.split('.')[0]) + 1  # increment by 1
        return f'{str(new_prefix).zfill(6)}.jpg'
    else:
        return '000000.jpg'

# Return a new file name for checkpoint to save
def get_available_binary_name(gan_type_enum=GANType.CLASSIC):
    def valid_binary_name(binary_name):
        pattern = re.compile(rf'{gan_type_enum.name}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    prefix = gan_type_enum.name
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def get_gan_data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    return transform


def get_mnist_dataset():
    # This will download the MNIST the first time it is called
    return datasets.MNIST(root=DATA_DIR_PATH, train=True, download=True, transform=get_gan_data_transform())


def get_mnist_data_loader(batch_size):
    mnist_dataset = get_mnist_dataset()
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return mnist_data_loader


def get_gaussian_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)

# Loads up the neural net according to the type of gan selected(Right now there is only one available)
def get_gan(device, gan_type_name):
    assert gan_type_name in [gan_type.name for gan_type in GANType], f'Unknown GAN type = {gan_type_name}.'

    if gan_type_name == GANType.CLASSIC.name:
        d_net = DiscriminatorNet().train().to(device)
        g_net = GeneratorNet().train().to(device)
    else:
        raise Exception(f'GAN type {gan_type_name} not yet supported.')

    return d_net, g_net


def get_optimizers(d_net, g_net):
    d_opt = Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return d_opt, g_opt


def get_training_state(generator_net, gan_type_name):
    training_state = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "state_dict": generator_net.state_dict(),
        "gan_type": gan_type_name
    }
    return training_state


def print_training_info_to_console(training_config):
    print(f'Starting the GAN training.')
    print('*' * 80)
    print(f'Settings: num_epochs={training_config["num_epochs"]}, batch_size={training_config["batch_size"]}')
    print('*' * 80)

    if training_config["console_log_freq"]:
        print(f'Logging to console every {training_config["console_log_freq"]} batches.')
    else:
        print(f'Console logging disabled. Set console_log_freq if you want to use it.')

    print('')

    if training_config["debug_imagery_log_freq"]:
        print(f'Saving intermediate generator images to {training_config["debug_path"]} every {training_config["debug_imagery_log_freq"]} batches.')
    else:
        print(f'Generator intermediate image saving disabled. Set debug_imagery_log_freq you want to use it')

    print('')

    if training_config["checkpoint_freq"]:
        print(f'Saving checkpoint models to {CHECKPOINTS_PATH} every {training_config["checkpoint_freq"]} epochs.')
    else:
        print(f'Checkpoint models saving disabled. Set checkpoint_freq you want to use it')

    print('')

    if training_config['enable_tensorboard']:
        print('Tensorboard enabled. Logging generator and discriminator losses.')
        print('Run "tensorboard --logdir=runs" from your Anaconda (with conda env activated)')
        print('Open http://localhost:6006/ in your browser and you\'re ready to use tensorboard!')
    else:
        print('Tensorboard logging disabled.')
    print('*' * 80)
