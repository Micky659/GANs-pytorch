import os
import enum


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)


LATENT_SPACE_DIM = 100
MNIST_IMG_SIZE = 28
MNIST_NUM_CLASSES = 10


# For other GAN implementations
class GANType(enum.Enum):
    CLASSIC = 0
