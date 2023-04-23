#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:59:54 2023

@author: avinash
"""

# !pip install deepface cmake dlib

from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import cv2

# img = numpy array(BGR) or base64 or img_path

def getRepresentations(img):
    obj = DeepFace.represent(
        img_path=img,
        model_name='Facenet',
        detector_backend='dlib',
        enforce_detection=False,
    )
    return obj


def getSuspectEmbedding(img):
    obj = getRepresentations(img)
    return obj[0]['embedding']


# find match

THREESHOLD_VALUE = 0.5  # facenet


def findCosineDistance(vector_1, vector_2):
    a = np.matmul(np.transpose(vector_1), vector_2)
    b = np.matmul(np.transpose(vector_1), vector_1)
    c = np.matmul(np.transpose(vector_2), vector_2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# suspects_embeddings = [{id, embedding}, ...]


def findSuspects(input_img, suspects_embeddings):
    input_representations = getRepresentations(input_img)
    found_suspects = []  # ids of matched suspects
    matched_reps = []  # ids of matched faces in input
    i = 0
    for rep in input_representations:
        for suspect in suspects_embeddings:
            distance = findCosineDistance(
                suspect['embedding'], rep['embedding'])
            if distance < THREESHOLD_VALUE:
                found_suspects.append(suspect['id'])
                matched_reps.append(i)
        i = i+1

    # show bounding box around found suspects

    # if input_img is a numpy array
    if(isinstance(input_img, np.ndarray)):
        img = input_img 
        
    # else convert to a numpy array
    else:
        img = plt.imread(input_img)

    for id in matched_reps:
        facial_area = input_representations[id]['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return {'found_suspects': found_suspects, 'modified_img': img}


# testing
suspects = ['/home/avinash/Desktop/mini_project/suspects/img_09.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_04.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_03.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_07.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_06.jpeg',
            '/home/avinash/Desktop/mini_project/suspects/img_01.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_08.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_02.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_05.jpeg', '/home/avinash/Desktop/mini_project/suspects/img_10.jpeg']

sus_embeddings = []
for sus_img in suspects:
    e = getSuspectEmbedding(sus_img)
    sus_embeddings.append({'id': sus_img, 'embedding': e})
    
input_img = '/home/avinash/Desktop/mini_project/test_img.jpeg'
input_img = cv2.imread(input_img) 
results = findSuspects(input_img, sus_embeddings)

plt.imshow(results['modified_img'])
plt.show()

results['found_suspects']

ts = getRepresentations(input_img)
len(ts)