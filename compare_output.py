import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
#gt_df = pd.read_csv('ground_truth.csv')['labels']
#output = pd.read_csv('test_result_new.csv')['labels'] 
#
#
#a = list(gt_df)


img_dir_ALL = "D:\\Bhaia_Works\\ALL\\C-NMC_training_data\\fold_0\\all\\UID_1_2_2_all.bmp"
img_dir_HEM = "D:\\Bhaia_Works\\ALL\\C-NMC_training_data\\fold_0\\hem\\UID_H6_1_2_hem.bmp"

from imageio import imread
from pyemd.EMD2d import EMD2D

emd2d = EMD2D()
ALL_img = imread(img_dir_ALL)
HEM_img = imread(img_dir_HEM)


IMFredALL = emd2d.emd(ALL_img[:,:,0], max_imf = -1)
IMFgreenALL = emd2d.emd(ALL_img[:,:,1], max_imf = -1)
IMFblueALL = emd2d.emd(ALL_img[:,:,2], max_imf = -1)

IMFredHEM = emd2d.emd(HEM_img[:,:,0], max_imf = -1)
IMFgreenHEM = emd2d.emd(HEM_img[:,:,1], max_imf = -1)
IMFblueHEM = emd2d.emd(HEM_img[:,:,2], max_imf = -1)