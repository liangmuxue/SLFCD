import numpy as np
import matplotlib.pyplot as plt
from utils.vis import vis_data,visdom_data

def show_prob_map(map_filepath):
    map_data = np.load(map_filepath)
    plt.imshow(map_data)
    visdom_data(map_data,[])
    
if __name__ == '__main__':    
    map_filepath = "./results/infer_prop/80-CG23_15084_02.npy"
    show_prob_map(map_filepath)