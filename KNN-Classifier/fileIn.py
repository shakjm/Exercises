# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:22:53 2018

@author: Kelvin Shak
"""

import cv2

class SimplePreprocessor:
    def __init__(self,width,height,inter= cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self,image):
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
        

import numpy as np
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors = None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self,imagePaths, verbose =-1):
        data = []
        labels = []
        
        for (i, imagePath) in enumerate(imagePaths):
            label = imagePath.split(os.path.sep)[-1]
            image = cv2.imread(label)
            label = label.split('.')[0]
            
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            data.append(image)
            labels.append(label)
            
            if verbose > 0 and i > 0 and (i+1)%verbose == 0:
                print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))
            
        return(np.array(data),np.array(labels))
    
    
            