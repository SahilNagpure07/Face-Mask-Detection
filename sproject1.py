from ast import Pass
from email.utils import getaddresses
from itertools import count
from operator import imod
from typing import Counter
from unittest import result
from xmlrpc.client import DateTime
from click import style
import matplotlib.pyplot as plt
from nbformat import write
import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import datetime
import time
import pymongo
import pandas as pd
import csv

header = st.container()
startcam = st.container()
stopcam = st.container()
detection = st.container()

client = pymongo.MongoClient("mongodb://localhost:27017/")

def get_data(face_id):
    db = client['FaceRecognition']
    collection = db['FaceData']
    item = collection.find_one({"face_id":face_id})
    # print("item:",item)
    db.FaceData.update_one( {"face_id":face_id}, {"$inc": {"count":1} } )
    

    print(f"Name:{item['name']}")
    with open('recface.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"{item['face_id']}",f"{item['name']}", f"{item['count']}"])

def detectface(test_image):
    search_model = load_model('model_3.h5')
    face = search_model.predict(test_image)[0]
    print("face:",face)
    faceindex = np.argmax(face)
    # print("faceindex:",faceindex) 
    faceindex = int(faceindex)
    get_data(faceindex)




selectbox = st.sidebar.selectbox(
    "NAVIGATE HERE",
    ["Home", "Dashboard"]
)

if selectbox == "Home":

    with header:
        st.title("FACE MASK DETECTION😷")

    with  startcam:
        startcam = st.button(label="START🎦")
        if startcam:
            into ={st.info("starting camera...")}
            # st.info("Starting camera...")
            out = st.image([])
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            del into
            # st.info("Detecting faces...stay still")
    def stcam():   
            cv2.destroyAllWindows()

    with stopcam:
        stopcam = st.button(label="STOP")
        if stopcam: stcam()

    with detection:	
        mymodel=load_model('model_1.h5')
        inc=0
        f=0
        t=2

        nomask, coun = st.columns(2)
        with nomask:
            st.markdown('**No Mask Detected:**')
            cou = st.markdown("0")
        with coun:
            st.markdown('**Detected Face Count:**')
            cou1 = st.markdown("0")

        while startcam == True:
            
            while cap.isOpened():
                _,img = cap.read()

                time.sleep(t)
                face=face_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)
                cou1.write(f"<h3 style='text-align: left; color:black'>{len(face)}</h3>", unsafe_allow_html=True)
                for(x,y,w,h) in face:
                    face_img = img[y:y+h, x:x+w]
                    cv2.imwrite('temp.jpg',face_img)
                    
                    test_image = image.load_img('temp.jpg',target_size=(128,128,3))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image,axis=0)
                    pred = mymodel.predict(test_image)[0][0]
                    # print(pred)
                    if pred==1:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                        cv2.putText(img,'No Mask',(x,y-5),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                        inc+=1
                        cou.write(f"<h3 style='text-align: left; color:black'>{inc}</h3>", unsafe_allow_html=True)

                        detectface(test_image)
                    else:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                        cv2.putText(img,'Mask',(x,y-5),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                    datet = datetime.datetime.now()
                    datef = datet.strftime("%d-%m-%Y %H:%M:%S")
                    cv2.putText(img,datef,(400,450),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)      
                #cv2.imshow('img',img)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                out.image(img)

                if cv2.waitKey(1)==ord('z'):
                    break
            # cap.release()
            # cv2.destroyAllWindows()

@st.cache
def load_data(nrows=None):
    chart_data = pd.read_csv("recface.csv",nrows=nrows)
    return chart_data

if selectbox =="Dashboard":
    st.title("DASHBOARD")
    st.subheader("Showing last 5 records.")
    chart_data = load_data()
    st.write(chart_data.tail())
    showdata = pd.DataFrame(chart_data['name'].value_counts())
    st.header("BAR GRAPH")
    st.bar_chart(showdata)
    showdata.hist()
