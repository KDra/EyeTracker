from types import FrameType
import tensorflow as tf
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
from pathlib import Path
from sys import argv
from json import dump as jdump

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

eye_landmarks = {7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155,
      157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 276, 282, 283, 285, 293, 295,
      296, 300, 334, 336, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390,
      398, 466}

def extract_frames(video_path, every=1, save_frames=False, classes=['left', 'other', 'right']):
    """
    Extract frames from a video using OpenCVs VideoCapture

    :param video_path: path of the video
    :param frame_dict: list of frames
    :param every: frame spacing
    :return: count of images saved
    """
    # The three classes used in training the prediction network
    # This will be the output
    logits = []

    # Load the cropping network
    face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            min_detection_confidence=0.5,
                            # min_tracking_confidence=0.5
                        )
    # Load the prediction network
    interpreter = tf.lite.Interpreter(model_path="classifier2.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_shape = (input_shape[2], input_shape[1])

    video_filename = video_path.stem  # get the video path and filename from the path
    print('Processing file: ' + video_filename)

    assert video_path.exists()  # assert the video file exists

    capture = cv2.VideoCapture(str(video_path))  # open the video using OpenCV
    
    start = 0
    end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = {}
    is_range = True
    for i in range(start, end, every):
        frame_dict[i] = 'none'
    frame_no = np.min(list(frame_dict.keys()))

    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    # Make output directory
    if save_frames:
        frames_dir = video_path.parent/'output'/video_filename
        print(video_filename, len(frame_dict))
        if len(frame_dict)>1000:
            save_dir = frames_dir/f'{(frame_no//1000):03d}'/f'{(frame_no%1000//100)}'
            build_tree = True
        else:
            save_dir = frames_dir
            build_tree = False
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through frames
    for frame_no in frame_dict:  # lets loop through the frames until the end
        capture.set(1, frame_no)
        success, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip
        
        while_safety = 0  # reset the safety count
        
        # Process frame
        if save_frames:
            orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB colorspace
        image.flags.writeable = False # Helps process the frame faster
        results = face_mesh.process(image) # Extract face landmarks

        if results.multi_face_landmarks:
            lm = []
            for it in eye_landmarks:
                face_landmark = results.multi_face_landmarks[0].landmark[it]
                coords = [face_landmark.x, face_landmark.y]
                lm.append(coords)
            lm = np.array(lm)
            lm[:, 0] = lm[:, 0]*image.shape[1]
            lm[:, 1] = lm[:, 1]*image.shape[0]
            lm = lm.astype(int)
            xmin = np.min(lm[:, 0])
            xmax = np.max(lm[:, 0])
            ymin = np.min(lm[:, 1])
            ymax = np.max(lm[:, 1])
            ymax = int(ymax + (ymax-ymin))
            image = image[ymin:ymax, xmin:xmax].copy()
            image = cv2.resize(image, input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], image[np.newaxis])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0].tolist()
            # get the proper prediction class
            prediction_class = classes[np.argmax(output_data)]
            frame_dict[frame_no] = prediction_class
            output_data.append(prediction_class)
            output_data.append(frame_no)
            output_data.append(video_filename)
            logits.append(output_data)

            if save_frames:
                if (frame_no % 100 == 0) and build_tree:
                    save_dir = frames_dir/f'{(frame_no//1000):03d}'/f'{(frame_no%1000//100)}'
                    save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir/f"{frame_no:09d}_{frame_dict[frame_no]}.jpg"
                # print(prediction_class, frame_no, save_path)
                cv2.imwrite(str(save_path), orig_image)
                cv2.imwrite(str(save_path).replace('.jpg', '_cropped.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    capture.release()  # after the while has finished close the capture
    logits = pd.DataFrame(logits, columns=classes+['prediction_label', 'frame_no', 'video_name'])
    return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames and get as an output the subject gaze direction')
    parser.add_argument('videos', metavar='N', type=str, nargs='+',
                    help='a list of video files')
    parser.add_argument('-e', '--every', type=int, default=20, help='the number of frames to skip (default 20)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Wether to output the detected frames. Useful for debugging. (default False)')
    
    args = parser.parse_args()
    classes = ['left', 'other', 'right']
    videos = [Path(v) for v in args.videos]
    videos = [v for v in videos if v.exists()]

    output = Parallel(n_jobs=-1)(delayed(extract_frames)(video, every=args.every, save_frames=args.verbose, classes=classes) for video in videos)
    # output = [extract_frames(video, every=args.every, save_frames=args.verbose, classes=classes) for video in videos]
    df = pd.concat(output, ignore_index=True)
    df.to_csv(f"output_run_{pd.to_datetime('now').strftime('%Y%b%d-%H')}.csv")
    print('Success!')
