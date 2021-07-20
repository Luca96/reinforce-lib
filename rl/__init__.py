import os

from rl import utils
from rl import augmentations
from rl import environments
from rl import layers
from rl import parameters
from rl import presets

from rl.v2 import agents
from rl.v2 import memories
from rl.v2 import networks


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
