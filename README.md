# Computer-vision-for-modelling-growth-of-Phytophota-infestans

This work is still in progress.

This repository contains the code I used to segment lesions and model the growth of Phytophtora infestans on leaves of Solanum tuberosum (potatoe), using both RGB images and Fv/Fm intensity images. This was done during my internship at INRAE (Rennes, France), during my second year of Master's degree.

***Image registration***: Coherent Point Drift method adapted to this dataset. 

***Segmentation***: the goal was to compare the performances of different segmentation approaches for RGB images. Other methods are yet to be tried.

***Contour detection***: find contours of the lesions after supervised segmentation for RGB images and unsupervised segmentation for Fv/Fm images. Build a table containing areas of detected contours for each leaf, according to segmentation modality, day, Phytophtora strain...and export it for further statistical analysis using R (soon avaliable on the repo).

***Growth model***: Packages used by the model work on a Python 2.7 virtual environment. All the dependancies are listed in the requirements.txt file. The model itself will be uploaded soon.

Demo of the model:

![test_animation](https://user-images.githubusercontent.com/73390220/230041534-f4bf48a0-25e8-4fb0-8d34-4ac7cd05a82e.gif)
