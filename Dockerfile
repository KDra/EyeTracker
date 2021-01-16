# Eye tracker
# 
# App to load frames from the images directory and produce 
# face embeddings in the embeddings directory that should be
# used to track the gaze direction of the subjects
# 
# Development started on 17/10/2020
# Developers:
#   Konstantinos Drakopoulos    drakopuloskostas@gmail.com
#   Barbu Revencu

FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update
RUN apt install libgl1-mesa-glx -y
# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y
RUN pip install joblib pandas mediapipe

WORKDIR /tf/images
WORKDIR /tf/embeddings
WORKDIR /tf/videos
COPY videos/ .
WORKDIR /tf

COPY process_frames.py .
COPY classifier2.tflite .
CMD python process_frames.py videos/* -e 100
