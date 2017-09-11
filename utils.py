from scipy.io import loadmat
from datetime import datetime
import os



def load_data(mat_path):
    d = loadmat(mat_path)

    return d["X"], d["Y"], d["X_val"],d["Y_val"]


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass
