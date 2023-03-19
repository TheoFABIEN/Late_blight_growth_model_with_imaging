
import skimage as ski 
import os
os.chdir('')         #directory containing python scripts for image registration
from Coherent_point_drift import CPD
from Redresser import redresser       #preliminary processing for better results
import time
import numpy as np
import cv2
import shutil
from PIL import Image

def CPD_series_data(source, Multimodal = False, Multistart = True, Sym = False, Days = 4, border = 0):
   """Applies the Coherent Point Drift function to the whole dataset 
   (including all fluorescence modalities : fo, fv, fm, fv/fm).

   Args:
       source (string): path to the folder containing the dataset.
       Multimodal (bool, optional): True for registration between RGB and fv/fm images. Defaults to False.
       Multistart (bool, optional): True allows trying multiple starts for the algorithm. Defaults to True.
       Sym (bool, optional): True if the images are mirrored. Defaults to False.
       Days (int, optional): number of dpi on which we have pictures. Defaults to 4.
       border (int, optional): Size of borders to avoid bad cropping. Defaults to 0.
   """

   folders = ski.io.imread_collection(source + '/*')

   for folder in folders.files:
         
         if os.path.exists(folder + '/Recalibrated_CPD'):   
               shutil.rmtree(folder + '/Recalibrated_CPD')

         Images = ski.io.imread_collection(folder + '/*').files
         Visible = []
         Fo = []
         Fm = []
         Fv = []
         FvFm = []
         for im in Images:
               if '_Fo_' in im:
                  Fo.append(im)
               elif '_Fm_' in im:
                  Fm.append(im)
               elif '_FvFm_' in im:
                  FvFm.append(im)
               elif '_Fv_' in im:
                  Fv.append(im)
               else:
                  Visible.append(im)
         Visible = sorted(Visible, key = str.casefold)
         Fo = sorted(Fo, key = str.casefold)
         Fm = sorted(Fm, key = str.casefold)
         Fv = sorted(Fv, key = str.casefold)
         FvFm = sorted(FvFm, key = str.casefold)

         os.mkdir(folder + '/Recalibrated_CPD') 

         D0 = Image.open(Visible[0])                                                    
         resol = 0.3
         D0 = D0.resize( [int(resol * s) for s in D0.size] )
         D0 = np.array(D0)

         y, x, z = D0.shape
         ref = 255 * np.ones((border*2+y, border*2+x, z), dtype=np.uint8)
         ref[border:border+y, border:border+x] = D0
         ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + Visible[0].rsplit('/', 1)[-1], D0)
         f = open(str(folder) + '/Recalibrated_CPD/error.txt','a+')
         if len(Visible) != Days :
               print('Warning ! There are ' + str(abs(Days-len(Visible))) + 
                     ' missing files in the Visible folder ! \n' + 
                     'Leaf concerned : ' + str(folder))
               f.write('Warning ! There are ' + str(abs(Days-len(Visible))) + 
                       ' missing files in the Visible folder ! \n')
         
         # Point registration for RGB images:
         for file in Visible[1:len(Visible)]:
               img = Image.open(file) 
               img = img.resize([int(.4 * s) for s in img.size])
               img = np.array(img)
               img_new = CPD(D0,img,
                             Multimodal = False, 
                             Multistart=False, 
                             Sym = Sym, 
                             border = border)
               ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + file.rsplit('/', 1)[-1], img_new)
         Recalibrated = ski.io.imread_collection(folder + '/Recalibrated_CPD/' + '*.tif').files
         R = len(Recalibrated)

         # Point registration for fv/fm images:
         for i, recal in enumerate(Recalibrated):

            Vis = ski.io.imread(recal)
            Flu = Image.open(Fo[i])

            # pre-registration using the idea that the leaf should be longer than wider, if in the right position:
            adjusted = redresser(Flu)   
            Flu = np.array(adjusted[0])
            cpd = CPD(Vis, Flu, 
                      Multimodal = True, 
                      Multistart = Multistart, 
                      Sym = Sym, 
                      border = border, 
                      Param = True)
            Fo_new = cpd[0]

            ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + Fo[i].rsplit('/', 1)[-1], Fo_new)

            #Matrix computed on Fo will be used as reference for other modalities:
            Tm = cpd[1]
            T = np.float32(np.c_[Tm[0], [Tm[1][1], Tm[1][0]]]*Tm[2])

            Fv_temp = Image.open(Fv[i])
            Fv_temp = Fv_temp.rotate(adjusted[1], expand = True)
            if adjusted[2]:
               Fv_temp = Fv_temp.rotate(180, expand = True)
            Fv_temp = np.array(Fv_temp)
            loc = np.where(Fv_temp > 0)
            width = np.array([min(loc[0]),Fv_temp.shape[0]-max(loc[0]),
                          min(loc[1]),Fv_temp.shape[1]-max(loc[1])])
            width[width < 0] = 0
            Fv_crop = ski.util.crop(Fv_temp,((width[0],width[1]),
                                        (width[2],width[3])),copy=True)
            scale = np.max(Vis.shape[0:2]) / np.max(Fv_crop.shape[0:2])
            Fv_temp = cv2.resize(Fv_crop,(0,0),fx=scale,fy=scale)

            Fm_temp = Image.open(Fm[i])
            Fm_temp = Fm_temp.rotate(adjusted[1], expand = True)
            if adjusted[2]:
               Fm_temp = Fm_temp.rotate(180, expand = True)
            Fm_temp = np.array(Fm_temp)
            loc = np.where(Fm_temp > 0)
            width = np.array([min(loc[0]),Fm_temp.shape[0]-max(loc[0]),
                          min(loc[1]),Fm_temp.shape[1]-max(loc[1])])
            width[width < 0] = 0
            Fm_crop = ski.util.crop(Fm_temp,((width[0],width[1]),
                                        (width[2],width[3])),copy=True)
            scale = np.max(Vis.shape[0:2]) / np.max(Fm_crop.shape[0:2])
            Fm_temp = cv2.resize(Fm_crop,(0,0),fx=scale,fy=scale)

            FvFm_temp = Image.open(FvFm[i])
            FvFm_temp = FvFm_temp.rotate(adjusted[1], expand = True)
            if adjusted[2]:
               FvFm_temp = FvFm_temp.rotate(180, expand = True)
            FvFm_temp = np.array(FvFm_temp)
            loc = np.where(FvFm_temp > 0)
            width = np.array([min(loc[0]),FvFm_temp.shape[0]-max(loc[0]),
                          min(loc[1]),FvFm_temp.shape[1]-max(loc[1])])
            width[width < 0] = 0
            FvFm_crop = ski.util.crop(FvFm_temp, ((width[0],width[1]),
                                        (width[2],width[3])), copy = True)
            scale = np.max(Vis.shape[0:2]) / np.max(FvFm_crop.shape[0:2])
            FvFm_temp = cv2.resize(FvFm_crop, (0,0), fx = scale, fy = scale)
            
            if cpd[3] == 1:
               Fv_temp = ski.transform.rotate(Fv_temp, 90, resize = True)
               Fm_temp = ski.transform.rotate(Fm_temp, 90, resize = True)
               FvFm_temp = ski.transform.rotate(FvFm_temp, 90, resize = True)
            elif cpd[3] == 2:
               Fv_temp = ski.transform.rotate(Fv_temp, 180, resize = True)
               Fm_temp = ski.transform.rotate(Fm_temp, 180, resize = True)
               FvFm_temp = ski.transform.rotate(FvFm_temp, 180, resize = True)
            elif cpd[3] == 3:
               Fv_temp = ski.transform.rotate(Fv_temp, 270, resize = True)
               Fm_temp = ski.transform.rotate(Fm_temp, 270, resize = True)
               FvFm_temp = ski.transform.rotate(FvFm_temp, 270, resize = True)

            Fv_new = cv2.warpAffine(Fv_temp, 
                                    T, 
                                    dsize = (Vis.shape[1], Vis.shape[0]), 
                                    borderValue = 0)
            Fm_new = cv2.warpAffine(Fm_temp, 
                                    T, 
                                    dsize = (Vis.shape[1], Vis.shape[0]), 
                                    borderValue = 0)
            FvFm_new = cv2.warpAffine(FvFm_temp, 
                                      T, 
                                      dsize = (Vis.shape[1], Vis.shape[0]), 
                                      borderValue = 0)
            
            ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + Fv[i].rsplit('/', 1)[-1], Fv_new)
            ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + Fm[i].rsplit('/', 1)[-1], Fm_new)
            ski.io.imsave(folder + '/Recalibrated_CPD' + '/' + FvFm[i].rsplit('/', 1)[-1], FvFm_new)

            
   f.close() # Close the error file
   nline = len(open(str(folder) + '/Recalibrated_CPD/error.txt','r').read()) 
   if nline == 0 :
      os.remove(folder + '/Recalibrated_CPD/error.txt')


source = '/home/theo/Bureau/DATA/Fluo_Visible_Mildiou_Maj2023/Dossiers_Mildiou_Maj2023'

CPD_series_data(source, Multimodal = True, Sym = False, Days = 4, border = 0)