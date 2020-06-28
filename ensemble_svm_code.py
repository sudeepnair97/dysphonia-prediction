# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:27:08 2020

@author: Sudeep
"""

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
import pandas as pd
from firebase import Firebase
import parselmouth
from numpy import mean
import librosa
import os
import json
import time
#import wget

start = time.process_time();

config = {
        "apiKey" : "apiKey",
        "databaseURL":"databaseURL",
        "authDomain" : "voice-disorder-detection.firebaseapp.com",
        "storageBucket" : "voice-disorder-detection.appspot.com"
        }

firebase = Firebase(config)

storage = firebase.storage();

#soundURL = storage.child('sound.wav').getDownloadURL();

#wget.download(soundURL,'/home/ec2-user/soundDownload.wav')


soundData = storage.child("sound.wav").download("soundDownload.wav")
jsonData = storage.child('data.json').download('userData.json')

if(os.path.exists('soundDownload.wav')):
        sound = parselmouth.Sound('soundDownload.wav')
        with open('userData.json','r+') as f:
            data = json.load(f)
            f.truncate(0)

age = data['age']
gender = data['gender']
file = data['file']

def measurePitch(sound,f0min,f0max,unit,file):
    pitch = parselmouth.praat.call(sound,'To Pitch',0.0,f0min,f0max)
    f0 = parselmouth.praat.call(pitch , 'Get mean',0,0,unit)
    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    jitter = parselmouth.praat.call(pointProcess , "Get jitter (local)" , 0 , 0 , 0.001 , 0.02 , 1.3)
    shimmer =  parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    (numpySound , samplingRate) = librosa.load(file)
    mfccs = librosa.feature.mfcc(numpySound , samplingRate , n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfccs, order=1)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    return f0,jitter,shimmer,hnr,mfccs,mfcc_delta,mfcc_delta2

(f0,jitter,shimmer,hnr,mfccs,mfcc_delta,mfcc_delta2) = measurePitch(sound,75,500,'hertz','soundDownload.wav')
os.remove('soundDownload.wav')

mfcc_values = []
mfcc_delta_values = []
mfcc_delta2_values = []

for i in range(13):
    mfcc_values.append(mean(mfccs[i]))
    mfcc_delta_values.append(mean(mfcc_delta[i]))
    mfcc_delta2_values.append(mean(mfcc_delta2[i]))
    
app_input = {
        'age' : [age],
        'gender' : [gender],
        'f0' : [f0],
        'jitter' : [jitter],
        'shimmer' : [shimmer],
        'hnr' : [hnr],
        'mfcc-1' : [mfcc_values[0]],
        'mfcc-2' : [mfcc_values[1]],
        'mfcc-3' : [mfcc_values[2]],
        'mfcc-4' : [mfcc_values[3]],
        'mfcc-5' : [mfcc_values[4]],
        'mfcc-6' : [mfcc_values[5]],
        'mfcc-7' : [mfcc_values[6]],
        'mfcc-8' : [mfcc_values[7]],
        'mfcc-9' : [mfcc_values[8]],
        'mfcc-10' : [mfcc_values[9]],
        'mfcc-11' : [mfcc_values[10]],
        'mfcc-12' : [mfcc_values[11]],
        'mfcc-13' : [mfcc_values[12]],
        'mfcc1Del' : [mfcc_delta_values[0]],
        'mfcc2Del' : [mfcc_delta_values[1]],
        'mfcc3Del' : [mfcc_delta_values[2]],
        'mfcc4Del' : [mfcc_delta_values[3]],
        'mfcc5Del' : [mfcc_delta_values[4]],
        'mfcc6Del' : [mfcc_delta_values[5]],
        'mfcc7Del' : [mfcc_delta_values[6]],
        'mfcc8Del' : [mfcc_delta_values[7]],
        'mfcc9Del' : [mfcc_delta_values[8]],
        'mfcc10Del' : [mfcc_delta_values[9]],
        'mfcc11Del' : [mfcc_delta_values[10]],
        'mfcc12Del' : [mfcc_delta_values[11]],
        'mfcc13Del' : [mfcc_delta_values[12]],
        'mfcc1Del2' : [mfcc_delta2_values[0]],
        'mfcc2Del2' : [mfcc_delta2_values[1]],
        'mfcc3Del2' : [mfcc_delta2_values[2]],
        'mfcc4Del2' : [mfcc_delta2_values[3]],
        'mfcc5Del2' : [mfcc_delta2_values[4]],
        'mfcc6Del2' : [mfcc_delta2_values[5]],
        'mfcc7Del2' : [mfcc_delta2_values[6]],
        'mfcc8Del2' : [mfcc_delta2_values[7]],
        'mfcc9Del2' : [mfcc_delta2_values[8]],
        'mfcc10Del2' : [mfcc_delta2_values[9]],
        'mfcc11Del2' : [mfcc_delta2_values[10]],
        'mfcc12Del2' : [mfcc_delta2_values[11]],
        'mfcc13Del2' : [mfcc_delta2_values[12]],
        }

dataset = pd.read_csv("dataset/voice_features_final.csv")

dataset = dataset.drop("id",axis = 1)

#print(dataset)

features = dataset.drop("class",1)

target = dataset["class"]

X_train, X_test , y_train , y_test = train_test_split(features , target , test_size = 0.2)
app_input_dataFrame = pd.DataFrame(app_input)

model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='linear')
model3=svm.SVC(kernel='linear')
model4=svm.SVC(kernel='rbf',gamma='auto')
model5=svm.SVC(kernel='rbf',gamma='auto')
model6=svm.SVC(kernel='rbf',gamma='auto')
 
model = VotingClassifier(estimators=[('s1', model1), ('s2', model2),('s3',model3),('s4',model4),('s5',model5),('s6',model6)], voting='hard')
 
model.fit(X_train, y_train)

param_grid = {'C':[0.1,1,10,100,1000] , 'gamma':[1,0.1,0.01,0.001,0.0001]}


y_pred = model.predict(X_test)
app_pred = model.predict(app_input_dataFrame)

#class 0

#Accuracy: How often the classifier is correct
print("Accuracy:" , metrics.accuracy_score(y_test , y_pred))

#Precision: What percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_pred))

#Recall: What percentage of positive tuples were predicted correctly out of all the predictions
print("Recall:",metrics.recall_score(y_test,y_pred))
# =============================================================================
# 
# model = VotingClassifier(estimators,voting='hard')
# 
# model.fit(X_train,y_train)
# 
# y_pred = model.predict(X_test)
# 
# #Accuracy: How often the classifier is correct
# print("Accuracy:" , metrics.accuracy_score(y_test , y_pred))
# 
# #Precision: What percentage of positive tuples are labeled as such?
# print("Precision:",metrics.precision_score(y_test,y_pred))
# 
# #Recall: What percentage of positive tuples were predicted correctly out of all the predictions
# print("Recall:",metrics.recall_score(y_test,y_pred))
# =============================================================================

outputJSON = {
        'Class' : int(app_pred[0]),
        'Accuracy' : (float(metrics.accuracy_score(y_test , y_pred)))*100,
        'Precision' : (float(metrics.precision_score(y_test,y_pred)))*100,
        'Recall' : float(metrics.recall_score(y_test,y_pred))*100,
        'File' : file
        }

outputFile = open('outputData.json','w')


with open("outputData.json",'w',encoding = 'utf-8') as f:
   f.write(json.dumps(outputJSON))

storage.child('output.json').put('outputData.json')

print(time.process_time() - start);


