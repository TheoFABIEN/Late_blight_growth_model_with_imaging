import glob
import shutil
import os
import cv2
import skimage as ski 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from natsort import natsorted, ns
#%%

dataset = glob.glob(
        '/home/theo/datasets/Dossiers_Mildiou_Maj2023/*'
)
#%%

def get_visible_lesion(image_path):
    """
    Returns segmented visible image with only 2 classes, according to visible segmented image path 
    """

    image = ski.io.imread(image_path)
    image[image == 100] = 0
    image[image != 0] = 1
    return image

#%%

def infected_area_fluo(image_path):
    """
    Returns binary image giving infected pixels according to fluo image path
    """

    fluo_img = ski.io.imread(image_path)
    mask = np.array(fluo_img)
    mask[mask != 0] = 1
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel)

    fluo_threshold = np.copy(fluo_img)
    fluo_threshold[(fluo_threshold < .6) & (fluo_threshold != 0)] = 2
    fluo_threshold[fluo_threshold != 2] = 0
    fluo_threshold[mask == 0] = 0
    fluo_threshold[fluo_threshold != 0] = 1 

    return fluo_threshold


#%%

for leaf in dataset:

    
    #First create folder to save matrices for each sequence
    save_folder = '/home/theo/Bureau/suivis_infection_necrose/' + leaf.rsplit('/')[-1] 
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)


    segmentations_fluo = [
            infected_area_fluo(image_path)
            for image_path in natsorted(glob.glob(leaf + '/Recalibrated_CPD/*'), alg = ns.IGNORECASE)
            if 'FvFm' in image_path
    ]


    if len(segmentations_fluo) < 4:
        continue

    for method in ['RF', 'Unet', "XGB"]:

        segmentations_visible = [
                get_visible_lesion(image_path)
                for image_path in natsorted(glob.glob(leaf + '/Segment_' + method + '/*'), alg = ns.IGNORECASE)
                if 'Seg_' in image_path or 'Pred_class' in image_path
        ]



        matrix_day_infected = np.full(
                (
                    segmentations_fluo[0].shape[0], 
                    segmentations_fluo[0].shape[1],
                ),
                "No_infection",
        )

        matrix_day_dead = np.full(
                (
                    segmentations_fluo[0].shape[0], 
                    segmentations_fluo[0].shape[1],
                ),
                "No_infection",
        )

        for day_value, day_char in enumerate(['J2', 'J3', 'J4', 'J5']):

            current_vis_lesion = segmentations_visible[day_value]
            current_necrosis = segmentations_fluo[day_value]

  

            matrix_day_infected[np.logical_and(
                matrix_day_infected == "No_infection",
                current_vis_lesion == 1
            )] = day_char

            matrix_day_dead[np.logical_and(
                matrix_day_dead == "No_infection",
                current_necrosis == 1
            )] = day_char 

            matrix_day_dead[matrix_day_infected == 'No_infection'] = 'No_infection'
            matrix_day_infected[matrix_day_dead == 'J2'] = 'Dead_at_J2'

        np.save(save_folder + '/' + method + '_day_infected.npy', matrix_day_infected)
        np.save(save_folder + '/' + method + '_day_dead.npy', matrix_day_dead)
#%%

