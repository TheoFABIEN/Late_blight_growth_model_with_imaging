import skimage as ski 
from skimage import morphology
import skimage.feature
import numpy as np
import cv2
from pycpd import RigidRegistration
from functools import partial
import time
import scipy.spatial.distance as ssd

def CPD(img1_o,img2_o,Multimodal=False,Multistart=False,Sym=False,Param=False,border=0, artifacts = True):
    """Coherent Point Drift method for point registration, adapted to the Phytophtora dataset

    Args:
        img1_o (array): image taken as reference for the registration
        img2_o (array): image that is modified depending on img1_o
        Multimodal (bool, optional): True if the first image is RGB and the second one is fv/fm. Defaults to False.
        Multistart (bool, optional): True allows trying multiple starts for the algorithm. Defaults to False.
        Sym (bool, optional): True if the images are mirrored. Defaults to False.
        Param (bool, optional): True allows to also return the transformation matrix and position. Defaults to False.
        border (int, optional): Size of borders to avoid bad cropping. Defaults to 0.
        artifacts (bool, optional): True removes some artifacts that can distort the result. Defaults to True.
    """

    t0 = time.time()

    img1 = np.copy(img1_o)
    img2 = np.copy(img2_o)

    # Transform img2 if needed
    if Multimodal:

        if artifacts:
            arimg2 = np.array(img2)
            arimg2[arimg2 > 40] = 255
            arimg2[arimg2 <= 40] = 0
            arimg2 = morphology.remove_small_holes(arimg2.astype(int), 5000)    
            arimg2 = morphology.remove_small_objects(arimg2.astype(bool), 5000)   
            img2[arimg2 == False] = 0

        #Zoom on the leaf in image 2:
        loc = np.where(img2 > 0)                  # Coordinates of the leaf 
        if len(loc[0]) == 0 or len(loc[1]) == 0:
          print('Error ! No pixel is detected in this image !')
          return None
        width = np.array([min(loc[0]), img2.shape[0]-max(loc[0]),
                          min(loc[1]), img2.shape[1]-max(loc[1])])
        for w in width:     # Correct eventual negative values
            w = max(w, 0)

        img2_crop = ski.util.crop(
            img2,
            ((width[0],width[1]),(width[2],width[3])),
            copy = True
        )
        scale = np.max(img1.shape[0:2]) / np.max(img2_crop.shape[0:2])
        
        img2 = cv2.resize(
            img2_crop,
            (0,0),
            fx = scale,
            fy = scale
        )

    # Extract masks
    img1_m = ski.color.rgb2gray(img1) < 1 
    img2_m = img2 > 0 if Multimodal else ski.color.rgb2gray(img2) <1 

    # Extract contours with Canny filter
    img1_c = ski.feature.canny(img1_m)       # Contour 
    img1_p = np.transpose(np.where(img1_c))  # Location of the contour points
    img2_c = ski.feature.canny(img2_m)      
    img2_p = np.transpose(np.where(img2_c))  

    # Callback function
    def trace(iteration, error,X,Y): # Function to see the progress of the registration
        print('Iteration: {:d}\nError: {:06.4f}'.format(iteration, error))
    callback = partial(trace) 

    background = 0 if Multimodal else [255, 255, 255]
    Multistart = True if Sym else Multistart 

    if Multistart:
       h = 1 if Multimodal else 255     
       starts = [img2.astype('uint8'),
        (ski.transform.rotate(img2,90,resize=True)*h).astype('uint8'), 
       (ski.transform.rotate(img2,180,resize=True)*h).astype('uint8'),
       (ski.transform.rotate(img2,270,resize=True)*h).astype('uint8')]

       y1, x1, z1 = img1_o.shape
       img1 = 255 * np.ones((border*2+y1, border*2+x1, z1), dtype = np.uint8)
       img1[border:border+y1, border:border+x1] = img1_o
       img1_m = ski.color.rgb2gray(img1) <1 
       img1_c = ski.feature.canny(img1_m) 
       img1_p = np.transpose(np.array(np.where(img1_c)))  

       similarity = []    # List of similarity index
       Tm = []            # List of transformation matrices
       new = []           # List of transformed images
 
       for i, s in enumerate(starts, 1) :

           s_c = ski.feature.canny(s > 0) if Multimodal else ski.feature.canny(ski.color.rgb2gray(s) < 1)
           s_p = np.transpose(np.where(s_c))  

           # Point registration
           print('Start ' + str(i))
           reg = RigidRegistration(**{ 'X': img1_p, 'Y':s_p,'max_iterations':200, 'tolerance':0.0001})
           reg.register(callback)
           # Affine transformation with OpenCV
           T = np.float32(np.c_[reg.R, [reg.t[1],reg.t[0]]]*reg.s) # Transformation matrix
           s_new = cv2.warpAffine(
               s, 
               T, 
               dsize = (img1.shape[1],img1.shape[0]), 
               borderValue = background
           )
            
           if Multimodal:
               s_c = ski.feature.canny(s_new > 0) # Extract contour with Canny filter
           else :
               s_c = ski.feature.canny(ski.color.rgb2gray(s_new) < 1)

           s_p = np.transpose(np.where(s_c)) 
           dst = np.mean(ssd.cdist(img1_p,s_p,'seuclidean'))
            
           similarity.append(-dst), Tm.append((reg.R,reg.t,reg.s)), new.append(s_new)

       print('Elapsed time: ' + str(time.time()-t0) + ' s')

       if Param:
          return (new[similarity.index(max(similarity))], # image with the minimal error
                  Tm[similarity.index(max(similarity))], 
                  starts[similarity.index(max(similarity))], similarity.index(max(similarity)))
       
       return new[similarity.index(max(similarity))] 

   # If Multistart is not used:
    reg = RigidRegistration(**{ 'X': img1_p, 'Y':img2_p,'max_iterations':200, 'tolerance':0.0001})
    reg.register(callback)
    T = np.float32(np.c_[reg.R,[reg.t[1],reg.t[0]]]*reg.s)
    img2_new = cv2.warpAffine(
        img2, 
        T, 
        dsize = (img1.shape[1],img1.shape[0]), 
        borderValue = background
    )

    print('Elapsed time: ' + str(time.time()-t0) + ' s')
    if Param:
        return(img2_new, T)
    return img2_new


