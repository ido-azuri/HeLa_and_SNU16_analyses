#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:49:46 2024

@author: ido
"""

# The Code that is supplied here is to analyse,
# DAPI and one channel of ecDNAs (DMs).
# It segments all channels and does analyses.
# Details are in manuscript and the following code.

# Import Libraries
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
import cv2

from fcmeans import FCM
from skimage.morphology import remove_small_objects

#%%

# Data Paths
data_path = '/PATH_TO_DATA_FOLDER/SNU16/'
exp_1_path = 'PATH_TO_EXPERIMENT/'

#%%

# Saving Paths
save_data_1 = '/PATH_TO_SAVE_DATA/'

# Creating directory
os.mkdir(save_data_1)

#%%

# Access all image files
files = glob.glob(data_path + exp_1_path + '*')

#%%

# Storing data lists
overlap = []
file_names = []
dm_pix = []
cc_n_passed_size_all = []
non_overlap_dms_pix = []

# Parameters for segmenting "BLUE" - DAPI
N_Blue_Clusters_B = 20
N_Clust_Thresh_B = 7

# Parameters for segmenting "RED" - DMs
Adapt_Window_Thresh_R = 9
Adapt_Const_Shift_R = -10

# Bluring parameter
B_SIZE = 3

# Function for segmentation and analyses
def segment_nuc(files, B_SIZE, N_Blue_Clusters_B, N_Clust_Thresh_B,
                Adapt_Window_Thresh_R, Adapt_Const_Shift_R, save_data_rep):
    '''

    Parameters
    ----------
    files : List
        List of files.
    B_SIZE : INT
        Bluring parameter.
    N_Blue_Clusters_B : INT
        Number of Clusters.
    N_Clust_Thresh_B : INT
        Threshold Cluster.
    Adapt_Window_Thresh_R : INT
        Adaptive Mean Parameter 1 (DMs).
    Adapt_Const_Shift_R : INT
        Adaptive Mean Parameter 2 (DMs).
    save_data_rep : String
        Path for saving data.

    Returns
    -------
    overlap : List
        Overlap of DMs with DAPI.
    file_names : List
        Contains names of smaples.
    dm_pix : List
        Number of pixels for DMs.
    non_overlap_dms_pix : TYPE
        The number of pixels of DMs that
        are completely not overlapped with DAPI.
    cc_n_passed_size_all : List
        Number of pixels of each DM that has at least partial
        overlap with DAPI.

    '''
    # Iterating over files and read images
    for file in files:
        img = tifffile.imread(file)
        blue = img[2] # "BLUE" channel (DAPI)
        red = img[1] # "RED" channel (DMs)

        # Segmenting DAPI with clustering
        blue = cv2.GaussianBlur(blue, (B_SIZE, B_SIZE), 0)

        H, W = blue.shape
        X = np.expand_dims(blue.flatten(), axis=1)
        fcm = FCM(n_clusters=N_Blue_Clusters_B)
        fcm.fit(X)
        labeled_X = fcm.predict(X)
        transformed_X = fcm.centers[labeled_X]
        transformed_X = transformed_X.reshape(H, W)

        blue_bin = transformed_X > np.unique(transformed_X)[N_Clust_Thresh_B]
        #---------------------------------------------------------------------

        # Segmenting DMs with adaptive mean thresholding
        red_ = (red / np.max(red)) * 255
        red_ = np.uint8(red_)

        red_ = cv2.adaptiveThreshold(red_,
                                     255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     Adapt_Window_Thresh_R,
                                     Adapt_Const_Shift_R)

        red2_ = remove_small_objects(red_ == 255, min_size=10, connectivity=1)
        red_bin = red2_

        # Analyses:
        # Calculating area of DMs that has at least partial
        # overlap with DAPI
        cc_number, cc_red = cv2.connectedComponents(np.uint8(red_bin))
        cc_n_passed = []
        cc_n_passed_size = []
        y_gc_passed = []
        x_gc_passed = []
        for cc_n in range(1, cc_number):
            cc_red_bin = cc_red == cc_n
            unique_val_overlap = np.unique(np.int8(cc_red_bin) + np.int8(blue_bin))
            if (len(unique_val_overlap) == 3) and (np.array_equal(unique_val_overlap, np.array([0, 1, 2]))):
                cc_n_passed.append(cc_n)
                cc_n_passed_size.append(np.sum(cc_red_bin))

                y_gc_coord, x_gc_coord = np.where(cc_red_bin > 0)
                y_gc_coord = np.average(y_gc_coord)
                x_gc_coord = np.average(x_gc_coord)
                y_gc_passed.append(y_gc_coord)
                x_gc_passed.append(x_gc_coord)

        # Number of pixels of DMs signals that are
        # completely not overlapped with DAPI
        non_overlap_dm = np.zeros_like(red_bin)
        for cc_n in range(1, cc_number):
            cc_red_bin = cc_red == cc_n
            unique_val_overlap_2 = np.unique(np.int8(cc_red_bin) + np.int8(blue_bin))
            if (len(unique_val_overlap_2) == 2) and (np.array_equal(unique_val_overlap_2, np.array([0, 1]))):
                non_overlap_dm += cc_red_bin

        non_overlap_dms_pix_ = np.sum(non_overlap_dm)
        non_overlap_dms_pix.append(non_overlap_dms_pix_)

        cc_n_passed_size_all.append(cc_n_passed_size)

        # Plotting
        f_name = file.split('/')[-1]

        fig = plt.figure(figsize=[4, 6], dpi=500)
        plt.suptitle(f_name)
        plt.subplot(3, 2, 1)
        plt.imshow(blue)
        plt.title('DAPI', fontsize=8)
        plt.axis('off')
        plt.subplot(3, 2, 2)
        plt.imshow(blue_bin)
        plt.title('DAPI-SEGMENTED', fontsize=8)
        plt.axis('off')
        plt.subplot(3, 2, 3)
        plt.imshow(red)
        plt.title('DMs', fontsize=8)
        plt.axis('off')
        plt.subplot(3, 2, 4)
        plt.imshow(red_bin)
        plt.title('DMs-SEGMENTED', fontsize=8)
        plt.axis('off')

        # Calculating overlap of signals ("RED" (DMs )with "BLUE" (DAPI))
        dm_pix_ = np.sum(red_bin)
        dm_pix.append(dm_pix_)

        overlap_ = np.sum(red_bin * blue_bin) / np.sum(red_bin)
        overlap.append(overlap_)

        file_names.append(f_name)

        green = np.zeros_like(blue)
        img_rgb = cv2.merge((red, green, blue))
        img_rgb = np.uint8((img_rgb / np.max(img_rgb)) * 255)

        green_bin = np.zeros_like(blue_bin)
        img_rgb_bin = cv2.merge((np.uint8(red_bin),
                                 np.uint8(green_bin),
                                 np.uint8(blue_bin)))

        img_rgb_bin *= 255
        img_rgb_bin = np.uint8(img_rgb_bin)

        plt.subplot(3, 2, 5)
        plt.imshow(img_rgb)
        plt.title('RAW IMAGE', fontsize=8)
        plt.axis('off')
        plt.subplot(3, 2, 6)
        plt.imshow(img_rgb_bin)
        plt.title('SEGMENTED \n OVERLAP = ' + str(np.round(overlap_, decimals=2)), fontsize=8)
        plt.axis('off')
        plt.savefig(save_data_rep + f_name.split('.')[0] + '.png')
        plt.close(fig)
    return overlap, file_names, dm_pix, non_overlap_dms_pix, cc_n_passed_size_all

#%%

# Calling the function segment_nuc() to do the calculations
overlap_1, file_names, dm_pix_1, non_overlap_dms_pix_1, passed_size = segment_nuc(files,
                                                                                  B_SIZE, N_Blue_Clusters_B, N_Clust_Thresh_B,
                                                                                  Adapt_Window_Thresh_R, Adapt_Const_Shift_R,
                                                                                  save_data_1)

#%%

# Writing all featrues to a file
df_overlap_1 = pd.DataFrame([file_names,
                             overlap_1,
                             dm_pix_1,
                             non_overlap_dms_pix_1,
                             passed_size]).T

df_overlap_1.columns = ['File_Name',
                        'Overlap',
                        'DMs_AREA_PIX',
                        'DMs_AREA_PIX_NOT_OVERLAP',
                        'DMs_SIZE_OVERLAP']

df_overlap_1.to_csv(save_data_1 + exp_1_path.split('/')[0] + '_overlap_1' + '.csv')

#%%

# Writing sizes of DMs that are at least with partial
# overlap with DAPI to a file
df_to_sizes = df_overlap_1.copy()
df_to_sizes = df_to_sizes['DMs_SIZE_OVERLAP']

lens_dms_lists  = []
for dms_list in df_to_sizes:
    lens_dms_lists.append(len(dms_list))

max_n_dms = np.max(lens_dms_lists)

df_sizes = np.zeros((len(lens_dms_lists), max_n_dms))

for idx_n, dms_list in enumerate(df_to_sizes):
    df_sizes[idx_n, :len(dms_list)] =  np.array(dms_list)

df_sizes = pd.DataFrame(df_sizes)
df_sizes = pd.concat([df_overlap_1['File_Name'], df_sizes], axis=1)
df_sizes.to_csv(save_data_1 + exp_1_path.split('/')[0] + 'dms_overlap_1_sizes' + '.csv')
