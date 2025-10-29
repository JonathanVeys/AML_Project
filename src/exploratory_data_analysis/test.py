import numpy as np
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
train_data = np.load(ROOT / 'data/species_test.npz', allow_pickle=True)



for key in train_data.keys():
    print(key)
