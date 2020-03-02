# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:12:35 2020

@author: tripa
"""

# -*- coding: utf-8 -*-
"""
    Final_Year_Project.ipynb
"""

# Import useful libraries
import os
import sys
import cv2
import csv
import numpy as np
from pathlib import Path
from keras.models import model_from_json
from utils import prepare, binarizer

'''
    I. Data Preparation and Cleaning
'''

'''
The dataset used for this project is the
'Real Life Violence Situations Dataset' taken from Kaggle.
The dataset Contains 1000 Violence and 1000 non-violence videos
collected from youtube videos, violence videos in the dataset
contains many real street fights situations in several
environments and conditions.

Ref - https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset
Cite -
M. Soliman, M. Kamal, M. Nashed, Y. Mostafa, B. Chawky, D. Khattab,
“ Violence Recognition from Videos using Deep Learning Techniques”,
Proc. 9th International Conference on Intelligent Computing and Information
Systems (ICICIS'19), Cairo, pp. 79-84, 2019
'''

dataset_path = './Real Life Violence Dataset/'
sampledata_path = './Sample Data/'

'''
In the first step to data preperation, we create
a CSV file with video file name and annotation
referring to whether the video is violent (label=0) or
non-violent (label=1) category.
'''

print('Creating annotations file for the dataset.')
pathlist = Path(sampledata_path).glob('**/*.*')
with open('annotations.csv', 'w', newline='') as csvFile:

    writer = csv.writer(csvFile)
    for path in pathlist:

        base = os.path.basename(path)
        if base.startswith('V'):
            writer.writerow([base, 0])
        if base.startswith('NV'):
            writer.writerow([base, 1])

csvFile.close()
print('Annotations file created successfully.')


# %%

'''
    II. Loading pre-trained model
'''


'''
Before we start the training process, we load a pre-trained
model called Inception, developed by Google.
This model known as 'The Inception-v3' is trained on the
ImageNet Large Visual Recognition Challenge Dataset.
'''

print('\nReading Model JSON File')
model_json_file = open('./model.json', 'r')
model = model_json_file.read()
inception_model = model_from_json(model)
model_json_file.close()

# Load the model weights into the inception_model object
print('Loading the model weights')
inception_model.load_weights('./model.h5', 'r')
print('Model loaded successfully from disk.')
# inception_model.summary()


# %%

'''
    III. Extracting useful features from the pre-trained Inception v3 model
'''


'''
Extract features from image using the inception_model.
To do so, we capture snapshots from the video file and
run it through the inception_model to get the features.
'''

inception_features = []
inception_labels = []
with open('annotations.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    # Calculate the number of rows in CSV file and reset the file pointer to 0.
    number_of_files = len(list(csv_reader))
    csv_file.seek(0)
    file_no = 1

    for row in csv_reader:

        frame_no = 0
        ret = 1
        capture = cv2.VideoCapture(sampledata_path + row[0])
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_ex = total_frames // 15  # To extract every 15th frame
        frame_features = []
        l1 = []
        print(f'\rProcessing Video File {file_no}/{number_of_files}')
        while ret:
            ret, frame = capture.read()
            if(frame_no % frame_ex == 0):
                input_x = prepare(frame)
                feature = inception_model.predict(input_x)
                l1.append(feature.reshape(5*5*2048))
                if(len(l1) == 15):
                    break
            frame_no += 1
        frame_features = np.array(l1)
        feature = frame_features[:15,:].reshape(15, 5*5*2048)
        inception_features.append(feature)
        inception_labels.append(binarizer(int(row[1])))
        file_no += 1
        #sys.stdout.flush()

inception_features = np.array(inception_features)
inception_labels = np.array(inception_labels)

print(inception_features.shape)
print(inception_labels.shape)

np.save('inception_features.npy', inception_features)
np.save('inception_labels.npy', inception_labels)