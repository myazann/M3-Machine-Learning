#!/usr/bin/env python
# coding: utf-8
# %%
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Parts we added
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import random

from numpy import linalg as LA
from sklearn.preprocessing import power_transform, StandardScaler, normalize
from sklearn import svm
import time


# %%
def create_dense_kp(img_shape, step_div_size=50, num_sizes=1):
    
    keypoints = []
    init_step_size_x = max(img_shape[1] // step_div_size, 8)
    init_step_size_y = max(img_shape[0] // step_div_size, 8)
    
    for i in range(1, num_sizes+1):
        current_step_size_x = init_step_size_x * i
        current_step_size_y = init_step_size_y * i
        kp_size = (current_step_size_x + current_step_size_y) // 2
        
        keypoints += [cv2.KeyPoint(x, y, kp_size) for y in range(0, img_shape[0], current_step_size_y) 
                                                  for x in range(0, img_shape[1], current_step_size_x)]
    return keypoints


def get_descriptors(feat_num=250, step_div_size=10, num_sizes=1, mode="train"):

    descriptors = []
    label_per_descriptor = []
    
    if mode == "train":        
        img_filenames = pickle.load(open('train_images_filenames.dat','rb'))
        img_filenames = ['..' + n[15:] for n in img_filenames]
        lbl_filenames = pickle.load(open('train_labels.dat','rb'))
        
    else:
        img_filenames = pickle.load(open('test_images_filenames.dat','rb'))
        img_filenames = ['..' + n[15:] for n in img_filenames]
        lbl_filenames = pickle.load(open('test_labels.dat','rb'))
        
        
    Detector = cv2.SIFT_create(feat_num)

    for filename,labels in zip(img_filenames, lbl_filenames):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        
        kpt = create_dense_kp(gray.shape, step_div_size=step_div_size, num_sizes=num_sizes)                              
        _, des = Detector.compute(gray, kpt)
            
        descriptors.append(des.astype(int))
        label_per_descriptor.append(labels)
    
    return descriptors, label_per_descriptor


def get_codebook(descriptors, k=128):
    
    D = np.vstack(descriptors)
    codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k*20, compute_labels=False,
                               reassignment_ratio=10**-4, random_state=42)
    codebook.fit(D)
    
    return codebook


def get_visual_words(codebook, descriptors, k=128):
    
    descriptors = np.array(descriptors)
    descriptors = descriptors.reshape(len(descriptors), -1, descriptors[0].shape[1])
    visual_words=np.zeros((descriptors.shape[0], k),dtype=np.float32)
    
    for i in range(descriptors.shape[0]):
        words=codebook.predict(descriptors[i])
        visual_words[i,:]=np.bincount(words, minlength=k)
        
    return visual_words


# %%
def get_spatial_pyramids(pyramid_level=1, step_div_size=10, num_sizes=1, mode="train"):
    
        
    descriptors = []
    label_per_descriptor = []
    
    if mode == "train":        
        img_filenames = pickle.load(open('train_images_filenames.dat','rb'))
        img_filenames = ['..' + n[15:] for n in img_filenames]
        lbl_filenames = pickle.load(open('train_labels.dat','rb'))
        
    else:
        img_filenames = pickle.load(open('test_images_filenames.dat','rb'))
        img_filenames = ['..' + n[15:] for n in img_filenames]
        lbl_filenames = pickle.load(open('test_labels.dat','rb'))
        
    Detector = cv2.SIFT_create()

    for filename,labels in zip(img_filenames, lbl_filenames):
        
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kpt = create_dense_kp(gray.shape, step_div_size=step_div_size, num_sizes=num_sizes)                              
        _, des = Detector.compute(gray, kpt)
        
        pyramid_descriptors = [des]
        
        for l in range(1, pyramid_level+1):
            level_factor = 2*l
            cell_h = int(gray.shape[0]/level_factor)
            cell_w = int(gray.shape[1]/level_factor)

            dense_kp_cell = create_dense_kp([cell_h,cell_w], step_div_size=step_div_size, num_sizes=num_sizes)

            for f_h in range(level_factor):
                shift_h = f_h*cell_h
                for f_w in range(level_factor):
                    shift_w = f_w*cell_w
                    cell = img[shift_h:shift_h+cell_h, shift_w:shift_w+cell_w]
                    _,des = Detector.compute(cell, dense_kp_cell)
                    pyramid_descriptors.append(des)
            
        descriptors.append(pyramid_descriptors)
        label_per_descriptor.append(labels)
    
    return descriptors, label_per_descriptor


def norm_spatial_descs(descriptors, norm_type="l2", power_norm_method='yeo-johnson'):
    
    D = np.vstack(np.array([np.vstack(img) for img in descriptors]))
        
    if norm_type == "l2":
        norm_descriptors = normalize(D)
    elif norm_type == "l1":
        norm_descriptors = normalize(D, norm="l1")
    elif norm_type == "power":
        norm_descriptors = power_transform(D, method=power_norm_method)
        
    return norm_descriptors.reshape(len(descriptors), -1, descriptors[0][0].shape[0], descriptors[0][0].shape[1])


def get_spatial_visual_words(codebook, pyramid_descriptors, k):
    
    visual_words_pyramid=np.zeros((len(pyramid_descriptors), k*len(pyramid_descriptors[0])),dtype=np.float32)
    
    for i in range(len(pyramid_descriptors)):
        
        pyramid_descriptor = pyramid_descriptors[i]
    
        visual_words=np.zeros(k*len(pyramid_descriptor),dtype=np.float32)
        
        for d in range(len(pyramid_descriptor)):

            if pyramid_descriptor[d] is None:
                visual_words[d*k:d*k+k]=np.zeros(k)
            else:
                words=codebook.predict(pyramid_descriptor[d])
                visual_words[d*k:d*k+k]=np.bincount(words,minlength=k)
                
        visual_words_pyramid[i,:] = visual_words
        
    return visual_words_pyramid

