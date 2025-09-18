#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:49:46 2024

@author: ido
"""

# The Code that is supplied here is to analyse,
# DAPI and two channels of ecDNAs, edDNA-1, and ecDNA-2.
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
overlap_red = []
overlap_green = []
file_names = []
dm_pix_red = []
dm_pix_green = []

count_green_red_all = []
count_red_green_all = []

count_red_green_m_all = []
count_green_red_m_all = []

non_overlap_dms_pix_3 = []
non_overlap_dms_pix_4 = []

# Parameters for segmenting "BLUE" - DAPI
N_Blue_Clusters_B = 20
N_Clust_Thresh_B = 7

# Parameters for segmenting "RED" - ecDNA-1
Adapt_Window_Thresh_R = 9
Adapt_Const_Shift_R = -6

# Paramenters for segmenting "GREEN" - ecDNA-2
Adapt_Window_Thresh_G = 9
Adapt_Const_Shift_G = -4

# Bluring parameter
B_SIZE = 3

# Function for segmentation and analyses
def segment_nuc(files, B_SIZE, N_Blue_Clusters_B, N_Clust_Thresh_B,
                Adapt_Window_Thresh_R, Adapt_Const_Shift_R,
                Adapt_Window_Thresh_G, Adapt_Const_Shift_G, save_data_rep):
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
        Adaptive Mean Parameter 1 (ecDNA1).
    Adapt_Const_Shift_R : INT
        Adaptive Mean Parameter 2 (ecDNA1).
    Adapt_Window_Thresh_G : INT
        Adaptive Mean Parameter 1 (ecDNA2).
    Adapt_Const_Shift_G : INT
        Adaptive Mean Parameter 2 (ecDNA2).
    save_data_rep : String
        Path for saving data.

    Returns
    -------
    overlap_red : List
        Overlap of ecDNA-1 with DAPI.
    overlap_green : List
        Overlap of ecDNA-2 with DAPI.
    file_names : List
        List with all files.
    dm_pix_red : List
        Number of pixels for ecDNA-1.
    dm_pix_green : List
        Number of pixels for ecDNA-2.
    count_red_green_all : List
        For the ecDNA-1 that does not overlap with DAPI,
        the number of overlaps between ecDNA-1 and ecDNA-2 that
        does not overlap with DAPI
    count_red_green_m_all : List
        For the ecDNA-1 that does not overlap with DAPI,
        number of non-overlaps ecDNA-2 (not overlapped with DAPI) with ecDNA-1.
    count_green_red_all : List
    Notice: It is like count_red_green_all (Same results)
        For the ecDNA-2 that does not overlap with DAPI,
        the number of overlaps between ecDNA-2 and ecDNA-1 that
        does not overlap with DAPI.
    count_green_red_m_all : List
        For the ecDNA-2 that does not overlap with DAPI,
        number of non-overlaps ecDNA-1 (not overlapped with DAPI) with ecDNA-2.
    non_overlap_dms_pix_3 : List
        The number of pixels of ecDNA-1 that
        are completely not overlapped with DAPI.
    non_overlap_dms_pix_4 :
        The number of pixels of ecDNA-2 that
        are completely not overlapped with DAPI.

    '''

    # Iterating over files and read images
    for file in files:
        img = tifffile.imread(file)
        blue = img[2] # "BLUE" channel (DAPI)
        red = img[0] # "RED" channel (ecDNA-1)
        green = img[1] # "GREEN" channel (ecDNA-2)

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

        # Segmenting ecDNA-1 with adaptive mean thresholding
        red_ = (red / np.max(red)) * 255
        red_ = np.uint8(red_)

        red_ = cv2.adaptiveThreshold(red_,
                                     255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     Adapt_Window_Thresh_R,
                                     Adapt_Const_Shift_R)

        red2_ = remove_small_objects(red_ == 255,
                                     min_size=10,
                                     connectivity=1)
        red_bin = red2_
        #---------------------------------------------------------------------

        # Segmenting ecDNA-2 with adaptive mean thresholding
        green_ = (green / np.max(green)) * 255
        green_ = np.uint8(green_)

        green_ = cv2.adaptiveThreshold(green_,
                                       255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       Adapt_Window_Thresh_G,
                                       Adapt_Const_Shift_G)

        green2_ = remove_small_objects(green_ == 255,
                                       min_size=10,
                                       connectivity=1)
        green_bin = green2_
        #---------------------------------------------------------------------

        # Analyses 1:
        n_cc, img_cc = cv2.connectedComponents(np.uint8(red_bin))
        count_red_green_m = 0
        count_red_green = 0
        for n_cc_ in range(1,  n_cc):
            cc_bin = img_cc == n_cc_
            unique_val_overlap = np.unique(np.int8(cc_bin) + np.int8(blue_bin))
            # For each ecDNA-1 checking if it does not overlap with DAPI
            if (len(unique_val_overlap) == 2) and (np.array_equal(unique_val_overlap, np.array([0, 1]))):
                # Updating the number of ecDNA-1
                # that does not overlap with DAPI
                count_red_green_m += 1
                n_cc_g, img_cc_g = cv2.connectedComponents(np.uint8(green_bin))
                # For each ecDNA-1 that does not overlap with DAPI,
                # check for each ecDNA-2 if it does not overlap with DAPI
                for n_cc_g_ in range(1, n_cc_g):
                    cc_bin_g = img_cc_g == n_cc_g_
                    unique_val_overlap_green_blue = np.unique(np.int8(cc_bin_g) + np.int8(blue_bin))
                    if (len(unique_val_overlap_green_blue) == 2) and (np.array_equal(unique_val_overlap_green_blue, np.array([0, 1]))):
                        # If it does not overlap with DAPI,
                        # calculates if ecDNA-1 and ecDNA2 overlaps
                        unique_val_overlap_red_green = np.unique(np.int8(cc_bin_g) + np.int8(cc_bin))
                        if np.sum(unique_val_overlap_red_green == 2) == 1:
                            # Updating the number of ecDNA-1 and ecDNA-2 that overlaps
                            count_red_green += 1
        # For the ecDNA-1 that does not overlap with DAPI,
        # the number of overlaps between ecDNA-1 and ecDNA-2
        # that does not overlap with DAPI
        count_red_green_all.append(count_red_green)
        # For the ecDNA-1 that does not overlap with DAPI,
        # the number of non-overlaps of
        # ecDNA-2 (that does not overlap with DAPI) with ecDNA-1
        count_red_green_m_ = count_red_green_m - count_red_green
        count_red_green_m_all.append(count_red_green_m_)

        # Analyses 2: Same as Analyses 1,
        # but now for ecDNA-1 is ecDNA-2 and ecDNA-2 is ecDNA-1
        n_cc, img_cc = cv2.connectedComponents(np.uint8(green_bin))
        count_green_red_m = 0
        count_green_red = 0
        for n_cc_ in range(1,  n_cc):
            cc_bin = img_cc == n_cc_
            unique_val_overlap = np.unique(np.int8(cc_bin) + np.int8(blue_bin))
            if (len(unique_val_overlap) == 2) and (np.array_equal(unique_val_overlap, np.array([0, 1]))):
                count_green_red_m += 1
                print(count_green_red_m)
                n_cc_r, img_cc_r = cv2.connectedComponents(np.uint8(red_bin))
                for n_cc_r_ in range(1, n_cc_r):
                    cc_bin_r = img_cc_r == n_cc_r_
                    unique_val_overlap_red_blue = np.unique(np.int8(cc_bin_r) + np.int8(blue_bin))
                    if (len(unique_val_overlap_red_blue) == 2) and (np.array_equal(unique_val_overlap_red_blue, np.array([0, 1]))):
                        unique_val_overlap_green_red = np.unique(np.int8(cc_bin_r) + np.int8(cc_bin))
                        if np.sum(unique_val_overlap_green_red == 2) == 1:
                            count_green_red += 1
        # For the ecDNA-2 that does not overlap with DAPI,
        # the number of overlaps between ecDNA-2 and ecDNA-1
        # that does not overlap with DAPI
        # Notice, it should be the same result as in count_red_green_all
        count_green_red_all.append(count_green_red)
        # For the ecDNA-2 that does not overlap with DAPI,
        # the number of non-overlaps of ecDNA-1
        # (that does not overlap with DAPI) with ecDNA-2
        count_green_red_m_ = count_green_red_m - count_green_red
        count_green_red_m_all.append(count_green_red_m_)

        # Analyses 3: The number of pixels of ecDNA-1
        # that are completely not overlapped with DAPI
        cc_number_3, cc_red_3 = cv2.connectedComponents(np.uint8(red_bin))
        non_overlap_dm_3 = np.zeros_like(red_bin)
        for cc_n_3 in range(1, cc_number_3):
            cc_red_bin_3 = cc_red_3 == cc_n_3
            unique_val_overlap_3 = np.unique(np.int8(cc_red_bin_3) + np.int8(blue_bin))
            if (len(unique_val_overlap_3) == 2) and (np.array_equal(unique_val_overlap_3, np.array([0, 1]))):
                non_overlap_dm_3 += cc_red_bin_3

        non_overlap_dms_pix_3_ = np.sum(non_overlap_dm_3)
        non_overlap_dms_pix_3.append(non_overlap_dms_pix_3_)

        # Analyses 4: The number of pixels of ecDNA-2
        # that are completely not overlapped with DAPI
        cc_number_4, cc_green_4 = cv2.connectedComponents(np.uint8(green_bin))
        non_overlap_dm_4 = np.zeros_like(green_bin)
        for cc_n_4 in range(1, cc_number_4):
            cc_green_bin_4 = cc_green_4 == cc_n_4
            unique_val_overlap_4 = np.unique(np.int8(cc_green_bin_4) + np.int8(blue_bin))
            if (len(unique_val_overlap_4) == 2) and (np.array_equal(unique_val_overlap_4, np.array([0, 1]))):
                non_overlap_dm_4 += cc_green_bin_4

        non_overlap_dms_pix_4_ = np.sum(non_overlap_dm_4)
        non_overlap_dms_pix_4.append(non_overlap_dms_pix_4_)

        # Plotting
        f_name = file.split('/')[-1]

        fig = plt.figure(figsize=[4, 10], dpi=500)
        plt.suptitle(f_name)
        plt.subplot(4, 2, 1)
        plt.imshow(blue)
        plt.title('DAPI', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 2)
        plt.imshow(blue_bin)
        plt.title('DAPI-SEGMENTED', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 3)
        plt.imshow(red)
        plt.title('DMs', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 4)
        plt.imshow(red_bin)
        plt.title('DMs-SEGMENTED', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 5)
        plt.imshow(green)
        plt.title('DMs Green', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 6)
        plt.imshow(green_bin)
        plt.title('DMs Green-SEGMENTED', fontsize=8)
        plt.axis('off')

        # Calculating overlap of signals ("RED" and "GREEN" with "BLUE")
        dm_pix_ = np.sum(red_bin)
        dm_pix_red.append(dm_pix_)

        dm_pix_ = np.sum(green_bin)
        dm_pix_green.append(dm_pix_)

        overlap_ = np.sum(red_bin * blue_bin) / np.sum(red_bin)
        overlap_red.append(overlap_)
        overlap_red_ = overlap_

        overlap_ = np.sum(green_bin * blue_bin) / np.sum(green_bin)
        overlap_green.append(overlap_)
        overlap_green_ = overlap_

        file_names.append(f_name)

        img_rgb = cv2.merge((red, green, blue))
        img_rgb = np.uint8((img_rgb / np.max(img_rgb)) * 255)

        img_rgb_bin = cv2.merge((np.uint8(red_bin),
                                 np.uint8(green_bin),
                                 np.uint8(blue_bin)))

        img_rgb_bin *= 255
        img_rgb_bin = np.uint8(img_rgb_bin)

        # More sub-plots
        plt.subplot(4, 2, 7)
        plt.imshow(img_rgb)
        plt.title('RAW IMAGE', fontsize=8)
        plt.axis('off')
        plt.subplot(4, 2, 8)
        plt.imshow(img_rgb_bin)
        plt.title('SEGMENTED \n OVERLAP = ' + str(np.round(overlap_red_, decimals=2)) + '\n' + str(np.round(overlap_green_, decimals=2)), fontsize=8)
        plt.axis('off')
        plt.savefig(save_data_rep + f_name.split('.')[0] + '.png')
        plt.close(fig)
    return overlap_red, overlap_green, file_names, dm_pix_red, dm_pix_green, count_red_green_all, np.array(count_red_green_all) > 0, count_red_green_m_all, count_green_red_all, np.array(count_green_red_all) > 0, count_green_red_m_all, non_overlap_dms_pix_3, non_overlap_dms_pix_4

#%%

# Calling the function segment_nuc() to do the calculations
overlap_1, overlap_2, file_names, dm_pix_1, dm_pix_2, count_rg, count_rg_bool, count_rg_m, count_gr, count_gr_bool, count_gr_m, non_overlap_dm_red, non_overlap_dm_green = segment_nuc(files,
                                                                                                                                                                                       B_SIZE, N_Blue_Clusters_B, N_Clust_Thresh_B,
                                                                                                                                                                                       Adapt_Window_Thresh_R, Adapt_Const_Shift_R,
                                                                                                                                                                                       Adapt_Window_Thresh_G, Adapt_Const_Shift_G,
                                                                                                                                                                                       save_data_1)

#%%

# Writing all featrues to a file
df_overlap_gr = pd.DataFrame([file_names,
                              overlap_1, overlap_2,
                              dm_pix_1, dm_pix_2,
                              count_rg, count_rg_bool, count_rg_m,
                              count_gr, count_gr_bool, count_gr_m,
                              non_overlap_dm_red, non_overlap_dm_green]).T

df_overlap_gr.columns = ['File_Name', 'Overlap_Red', 'Overlap_Green',
                         'DMs_AREA_PIX_RED', 'DMs_AREA_PIX_GREEN',
                         'Close_to_Red', 'Close_to_Red_Bool', 'Close_to_Red_M',
                         'Close_to_Green', 'Close_to_Green_Bool', 'Close_to_Green_M',
                         'DMs_AREA_PIX_NOT_OVERLAP_RED', 'DMs_AREA_PIX_NOT_OVERLAP_GREEN']

df_overlap_gr.to_csv(save_data_1 + exp_1_path.split('/')[0] + '_overlap_gr' + '.csv')
