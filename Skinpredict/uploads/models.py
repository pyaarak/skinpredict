from django.db import models
from PIL import Image
from PIL import FontFile
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile
import h5py
import joblib
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mahotas
import cv2
import os
import pyrebase

images_per_class = 80
fixed_size       = tuple((500, 450))
num_trees = 100
test_size = 0.10
seed      = 9
#train_path='\skindisease\skindisease\skin\dataset\train' 
h5_data = r'\skinpredict\Skinpredict\uploads\data.h5'
h5_labels= r'\skinpredict\Skinpredict\uploads\labels.h5'
bins       = 8

# Create your models here.
class skin(models.Model): 
    name = models.CharField(max_length=200) 
    images = models.ImageField(upload_to='images/')
    def _str_(self):
        return str(self.id)
    def save(self, *args, **kwargs):
    
       pil_img=Image.open(self.images)
    
       images =np.array(pil_img)
       global_features = []
       labels          = []
       train_labels = ['acne comedo','acne cystic','acne pustular','Eczema','Eczema sever','Herpes','Lentigo','Melasma','Rosacea']
       train_labels.sort()
       h5f_data  = h5py.File(h5_data, 'r')
       h5f_label = h5py.File(h5_labels, 'r')
       global_features_string = h5f_data['dataset_1']
       global_labels_string   = h5f_label['dataset_1']
       global_features = np.array(global_features_string)
       global_labels   = np.array(global_labels_string)
       h5f_data.close()
       h5f_label.close()
       (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=test_size,random_state=seed)
       clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
       clf.fit(trainDataGlobal, trainLabelsGlobal)
       #image = cv2.imread(image)
       images = cv2.resize(images, fixed_size)
       fv_hu_moments = fd_hu_moments(images)
       fv_haralick   = fd_haralick(images)
       fv_histogram  = fd_histogram(images)
       global_feature = np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
       prediction = clf.predict(global_feature.reshape(1, -1))[0]
       cv2.putText(images, train_labels[prediction], (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,40,400), 2)
       images=cv2.cvtColor(images, cv2.COLOR_BGR2RGB) 
       im_pil = Image.fromarray(images)
       buffer = BytesIO()
       im_pil.save(buffer, format='png')
       image_png=buffer.getvalue()
       self.images.save(str(self.images), ContentFile(image_png), save=False)
       super().save(*args, **kwargs)
def fd_hu_moments(images):
    images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(images)).flatten()
    return feature
def fd_haralick(images):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(images, mask=None):
    images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([images], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()