#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            xt = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from url")
        if not 'RTA_model.joblib' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            url = "https://github.com/omkarnigade21/RTA_deployment/releases/download/model/RTA_model.joblib"
            r = requests.get(url, allow_redirects=True)
            open(r"RTA_model.joblib", 'wb').write(r.content)
            del r
        with open(r"RTA_model.joblib", "rb") as m:
            xt = joblib.load(m)
    return xt

