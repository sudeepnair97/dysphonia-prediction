# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:27:22 2020

@author: sudeepmn
"""
from firebase import Firebase
import json

while(1):
    try:
        config = {
                "apiKey" : "apiKey",
                "databaseURL":"databaseURL",
                "authDomain" : "voice-disorder-detection.firebaseapp.com",
                "storageBucket" : "voice-disorder-detection.appspot.com"
                }
        
        firebase = Firebase(config)
        
        storage = firebase.storage();
        
        jsonData = storage.child('data.json').download('userData.json')
        
        #url = storage.child('data.json').get_url(None);
        #wget.download(url,'/home/ubuntu/userData.json');
        
        with open('userData.json','r+') as f:
            data = json.load(f)
            f.truncate(0)
        
        #os.remove('userData.json');
        selection = data['select']
        
        
        if selection == 'SVM':
            exec(open('svm_code.py').read())
        elif selection == 'EnsembleSVM':
            exec(open('ensemble_svm_code.py').read())
    except:
        exec(open('automaticExec.py').read())