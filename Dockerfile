# Eye tracker
# 
# App to load frames from the images directory and produce 
# face embeddings in the embeddings directory that should be
# used to track the gaze direction of the subjects
# 
# Development started on 17/10/2020
# Developers Konstantinos Drakopoulos and Barbu Revencu

FROM tensorflow/tensorflow:2.3.1-gpu-jupyter

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN pip install mtcnn joblib

WORKDIR /tf/images
WORKDIR /tf/embeddings
WORKDIR /tf

COPY extract_faces.py .
# RUN python extract_faces.py
