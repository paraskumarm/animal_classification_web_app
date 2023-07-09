from flask import Flask,render_template,request
import os

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import make_pipeline

import os

import skimage
import skimage.color
import skimage.transform
import skimage.io
import skimage.feature
import pickle



app=Flask(__name__)

BASEPATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASEPATH,"static/uploads/")
MODEL_PATH=os.path.join(BASEPATH,'static/models/')

# loading models
model_sgd_path=os.path.join(MODEL_PATH,'image_classification_sgd.pickle')
scalar_path=os.path.join(MODEL_PATH,'dsa_scaler.pickle')

model_sgd=pickle.load(open(model_sgd_path,'rb'))
scalar=pickle.load(open(scalar_path,'rb'))

@app.route('/',methods=['GET','POST'])

def index():
    if(request.method=="POST"):
        upload_file=request.files['my_image']
        filename=upload_file.filename
        print("Uploaded File is ",filename)
        #extension of file
        #allow only .jpg,.jpeg,.png
        ext=filename.split('.')[-1]
        print("The extension of filename is ",ext)
      
        if(ext.lower() in ['png','jpeg','jpg']):
            path_save=os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            #send to ML model
            results=pipeline_model(path_save,scalar,model_sgd)
            print(results)
            return render_template('upload.html',fileupload=True,data=results,image_filename=filename,height=get_height(path_save))
        
        else:
            print(ext," file extension not allowed")

    return render_template('upload.html')
def pipeline_model(path,scaler_transformed,model_sgd):
    # read the image
    image = skimage.io.imread(path)
    # making transformation
    image_resized = skimage.transform.resize(image,(80,80))
    # rescaling
    rescaled_image = 255*image_resized
    image_transformed = rescaled_image.astype(np.uint8) # converting to 8 bit integer
    # graify
    gray = skimage.color.rgb2gray(image_transformed) # can use custom function as well
    # hog feature extraction
    hog_feature_vector = skimage.feature.hog(gray,
                                  orientations=10, pixels_per_cell=(8,8),cells_per_block=(3,3))
    
    #scaling
    scaled = scaler_transformed.transform(hog_feature_vector.reshape(1,-1))
    y_pred = model_sgd.predict(scaled)
    # confidence score for each class
    decision_value = model_sgd.decision_function(scaled)
    decision_value=decision_value.flatten()
    labels = model_sgd.classes_
    # probabilty 
    z = scipy.stats.zscore(decision_value)
    prob = scipy.special.softmax(z)
    top_5_prob_index = prob.argsort()[::-1][:5]
    top_5_prob_values = prob[top_5_prob_index]
    top_labels = labels[top_5_prob_index]
    #making dictionary
    top_dict = dict()
    for key,value in zip(top_labels,top_5_prob_values):
        top_dict[key]=np.round(value,2)
    
    return top_dict

def get_height(path):
    img=skimage.io.imread(path)
    h,w,_=img.shape
    aspect=h/w
    width=200
    height=aspect*width
    return height

if __name__=='__main__':
    app.run(debug=True)