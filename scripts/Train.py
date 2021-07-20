import transformers as b
import torch as t
import numpy as np
import os
from collections import defaultdict
from pymo.parsers import BVHParser
import torch as t
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline

class Train():
    def __init__(self,sample_n):
        self.sample_n = sample_n
