# Computer vision and modelling to study lesions caused by ***Phytophota infestans***

This work is still in progress.

This repository contains the code I used to segment lesions and model the growth of *Phytophtora infestans* on leaves of *Solanum tuberosum* (potato), using both RGB images and Fv/Fm intensity images. This was done during my internship at INRAE (Rennes, France), during my second year of Master's degree.

***Image registration***: Coherent Point Drift method adapted to this dataset. 

***Segmentation***: the goal was to compare the performances of different segmentation approaches for RGB images. Other methods are yet to be tried.

***Contour detection***: find contours of the lesions after supervised segmentation for RGB images and unsupervised segmentation for Fv/Fm images. Build a table containing areas of detected contours for each leaf, according to segmentation modality, day, Phytophtora strain...and export it for further statistical analysis using R (soon avaliable on the repo).

***Growth model***: To make it work, a Python 2.7 virtual environment is needed. All the dependancies are listed in the requirements.txt file. To install these, make shure you enabled the conda-forge channel. The model itself will be uploaded soon.

Demo of the model:

![test_animation](https://user-images.githubusercontent.com/73390220/230373101-42f8cbc4-fa3d-436a-8abb-55f4ffcebd0e.gif)
