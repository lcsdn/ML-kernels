from os import listdir, path
import pandas as pd
import numpy as np

def load_data(path_data: str, *filenames) -> dict:
    if not filenames:
        filenames = [filename for filename in listdir(path_data) if filename[-4:] == ".csv"]
    data = {}
    for filename in filenames:
        path_file = path.join(path_data, filename)
        data[filename] = np.loadtxt(path_file) if "mat" in filename else pd.read_csv(path_file)
    return data
