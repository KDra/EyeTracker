import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

eye_landmarks = {7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155,
      157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 276, 282, 283, 285, 293, 295,
      296, 300, 334, 336, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390,
      398, 466}

get_pixel_coords = lambda proportion, shape: int(proportion*shape)

# For video input:
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
# drawing_spec = mp_drawing.drawingspec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture('videos/prePilot1Order7.mp4')
success = True
landmark_coords = []
frame_no = -1
while success:
  frame_no += 1
  success, image = cap.read()
  if not success:
    print("ignoring empty camera frame.")
    # if loading a video, use 'break' instead of 'continue'.
    break
  if not (frame_no%1000==0):
    continue

  # flip the image horizontally for a later selfie-view display, and convert
  # the bgr image to rgb.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # to improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = face_mesh.process(image)

  # draw the face mesh annotations on the image.
  # image.flags.writeable = true
  # image = cv2.cvtcolor(image, cv2.color_rgb2bgr)
  if results.multi_face_landmarks:
    lm = []
    for it, face_landmark in enumerate(results.multi_face_landmarks[0].landmark):
      if it in eye_landmarks:
        coords = [face_landmark.x, face_landmark.y, frame_no]
        landmark_coords.append(coords)
        lm.append(coords)
        with open('eye_coords.csv', 'a') as f:
          f.write(', '.join([str(c) for c in coords]) + '\n')
    lm = np.array(lm)
    lm[:, 0] = lm[:, 0]*image.shape[1]
    lm[:, 1] = lm[:, 1]*image.shape[0]
    lm = lm.astype(int)
    xmin = np.min(lm[:, 0])
    xmax = np.max(lm[:, 0])
    ymin = np.min(lm[:, 1])
    ymax = np.max(lm[:, 1])
    ymax = int(ymax + (ymax-ymin)/2)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('embeddings/original_image' + str(frame_no) + '.png', image)
    cv2.imwrite('embeddings/cropped_image' + str(frame_no) + '.png', image[ymin:ymax, xmin:xmax])
  # cv2.imshow('mediapipe facemesh', image)
  if cv2.waitKey(5) & 0xff == 27:
    break
df = pd.DataFrame(landmark_coords, columns=['x', 'y', 'frame_no']).set_index('frame_no')
df.to_parquet('eyes.pq')
face_mesh.close()
cap.release()
