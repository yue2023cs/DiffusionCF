# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/10/2023
# ============================================================================

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import init
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional
from linear_attention_transformer import LinearAttentionTransformer
from scipy.stats import norm
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl                                                                                          # mush install this lib, since self.global_step is a inner objective of it
from abc import abstractmethod, ABC
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from scipy import signal
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.neighbors import KNeighborsTimeSeries
import matplotlib.ticker as ticker
from datetime import datetime

import datetime
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import matplotlib as plt
import math
from scipy.stats import genextreme
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse
