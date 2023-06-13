import numpy as np 
import skimage as ski
import matplotlib.pyplot as plt 
import glob
from natsort import natsorted
import cv2

folder = "/home/theo/Bureau/Model_results/Example_simulation/*"
files = glob.glob(folder)
files = natsorted(files)

time_biotroph = 1.7 #estimated time for an infected pixel to get necrotic


def binarize(image):
    new_image = np.copy(image)
    new_image[image >= .5] = 1
    new_image[image < .5] = 0
    return new_image

lesion_sequence = [
        np.load(img)
        for img in files
]

infected_pixels_sequence = [
        binarize(img)
        for img in lesion_sequence
]


mat_time_infected  = np.zeros((lesion_sequence[0].shape[0], lesion_sequence[0].shape[1]))

necrosis = []

for t, infected_image in enumerate(infected_pixels_sequence):

    infected_image = infected_image[:,:,0]
    mat_time_infected += infected_image
    infected_image[mat_time_infected >= 22] = 2
    necrosis.append(infected_image)

for img in necrosis:
    img[img == 1] = 0

# Change original sequence to add necrosis:
for i, lesion in enumerate(lesion_sequence):
    lesion[necrosis[i] == 2] = 2

    
# Plot images

# we need to get images of the leaf if we want to show its contour:
contour_fold = glob.glob(
    '/home/theo/datasets/Dossiers_Mildiou_Maj2023/1_R1_Bintje_ccf_P1/Recalibrated_CPD/*'
)
contour_fold = natsorted(
    [
        file for file in contour_fold
        if 'Bin_' in file
    ]
)

dim = np.zeros((lesion_sequence[0].shape[0], lesion_sequence[0].shape[1]))


for i, contour in [(0, 0), (13, 1), (26, 2), (39, 3)]:

    pred = np.stack(
        (lesion_sequence[i][:,:,0], lesion_sequence[i][:,:,0], lesion_sequence[i][:,:,0]), 
        axis = 2
    )
    pred[lesion_sequence[i][:,:,0] == 2]= [75, 0, 13]
    leaf_contour = ski.io.imread(contour_fold[contour])
    pred[leaf_contour == 255] = 255
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(pred), plt.show()
