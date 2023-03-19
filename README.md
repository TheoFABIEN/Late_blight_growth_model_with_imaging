# Computer-vision-for-modelling-growth-of-Phytophota-infestans

This work is still in progress.

This repository contains the code I used to segment lesions and model the growth of Phytophtora infestans on leaves of Solanum tuberosum (potatoe), using both RGB images and Fv/Fm intensity images. This was done during my internship at INRAE (Rennes, France), during my second year of Master's degree.

***Image registration***: Coherent Point Drift method adapted to this dataset. The function itself is avaliable, and files detailing its implementation on the whole dataset will be soon provided.

***Segmentation***: the goal was to compare the performances of different segmentation approaches for RGB images. Other methods are yet to be tried.

***Contour detection***: find contours of the lesions after supervised segmentation for RGB images and unsupervised segmentation for Fv/Fm images. Build a table containing areas of detected contours for each leaf, according to segmentation modality, day, Phytophtora strain...and export it for further statistical analysis using R (soon avaliable on the repo).

Growth model: soon.
