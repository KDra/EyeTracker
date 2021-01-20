from types import FrameType
import cv2
import pandas as pd
from icecream import ic
from pandas.core import frame
from psutil import cpu_count
from joblib import Parallel, delayed
from pathlib import Path
from sys import argv
from utils import *
from re import compile, IGNORECASE
import os

rgx = compile(r'(prepilot\d+).*', flags=IGNORECASE)

# blazeface_tf = tf.keras.models.load_model("./blazeface_tf.h5")

# def predict_frame(orig_frame):
#     orig_h, orig_w = orig_frame.shape[0:2]
#     frame = create_letterbox_image(orig_frame, 128)
#     h,w = frame.shape[0:2]
#     input_frame = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
#     input_tensor = np.expand_dims(input_frame.astype(np.float32), 0) / 127.5 - 1
#     result = blazeface_tf.predict(input_tensor)[0]
#     final_boxes, landmarks_proposals = process_detections(result,(orig_h, orig_w),5, 0.5, 0.5, pad_ratio=0.)
#     if len(final_boxes) == 0: return orig_frame
#     out_image = get_landmarks_crop(orig_frame, landmarks_proposals, (160, 256))
#     out_image = ((out_image + 1) * 127.5).astype(np.uint8)
#     return out_image


def extract_frames(video_path, frames_dir, frame_list=None):
    """
    Extract frames from a video using OpenCVs VideoCapture

    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    blazeface_tf = tf.keras.models.load_model("./blazeface_tf.h5")

    def predict_frame(orig_frame):
        orig_h, orig_w = orig_frame.shape[0:2]
        frame = create_letterbox_image(orig_frame, 128)
        h,w = frame.shape[0:2]
        input_frame = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(input_frame.astype(np.float32), 0) / 127.5 - 1
        result = blazeface_tf.predict(input_tensor)[0]
        final_boxes, landmarks_proposals = process_detections(result,(orig_h, orig_w),5, 0.5, 0.5, pad_ratio=0.)
        if len(final_boxes) == 0: return orig_frame
        out_image = get_landmarks_crop(orig_frame, landmarks_proposals, (160, 256))
        out_image = ((out_image + 1) * 127.5).astype(np.uint8)
        return out_image[0]
    
    video_filename = video_path.stem  # get the video path and filename from the path

    assert video_path.exists()  # assert the video file exists
    # Make output directory
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))  # open the video using OpenCV

    if not (frame_list is None):
        start = frame_list.min()
        end = frame_list.max()
    else:
        start = 0
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_no = start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    save_dir = frames_dir/f'{(frame_no//1000):03d}'/f'{(frame_no%1000//100)}'
    save_dir.mkdir(parents=True, exist_ok=True)
    every = 1
    for frame_no in frame_list:  # lets loop through the frames until the end
        capture.set(1, frame_no)
        success, image = capture.read()  # read an image from the capture
        if not success:
            continue

        if frame_no % 100 == 0:
            save_dir = frames_dir/f'{(frame_no//1000):03d}'/f'{(frame_no%1000//100)}'
            save_dir.mkdir(parents=True, exist_ok=True)

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame_no % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = save_dir/f"{frame_no:010d}.jpg" # create the save path
            if (not save_path.exists()):# or overwrite:  # if it doesn't exist or we want to overwrite anyways
                image = predict_frame(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame_no += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved

if __name__ == '__main__':
    input_csv = pd.read_csv(argv[1], index_col=0).reset_index(drop=True)
    input_csv['file'] = input_csv['file'].apply(lambda x: x + '.mp4' if not x.endswith('mp4') else x)
    csv_mapping = {rgx.sub(r'\1', i): i for i in input_csv['file'].unique()}
    frame_list = input_csv.groupby('file').frame.unique()
    videos = list(Path('videos/').glob('*'))
    output = Path('images_bf/')

    processed_videos = [csv_mapping[rgx.sub(r'\1', i.name)] for i in output.glob('*/')]
    frame_list = frame_list.loc[~frame_list.index.isin(processed_videos)]
    csv_mapping = {rgx.sub(r'\1', i): i for i in frame_list.index}
    num_processes = 6#cpu_count()
    if len(videos) < num_processes:
        num_processes = len(videos)
    existing_frames = {}
    non_existing = []
    existing_videos = []
    for v in videos:
        name = rgx.sub(r'\1', v.name)
        if name in csv_mapping.keys():
            existing_frames[v.name] = frame_list[csv_mapping[name]]
            existing_videos.append(v)
        else:
            non_existing.append(str(v))
    input(', '.join(non_existing))
    input(f"Videos to process: {', '.join(map(str, existing_videos))}")
    frame_dirs = [output/i.stem for i in existing_videos]    
    for d in frame_dirs:
        d.mkdir(parents=True, exist_ok=True)
    # frame_counts = []
    frame_counts = Parallel(n_jobs=num_processes)(delayed(extract_frames)(video, output_dir, existing_frames[video.name]) for video, output_dir in zip(existing_videos, frame_dirs))
    # frame_counts = [extract_frames(video, output_dir) for video, output_dir in zip(videos, frame_dirs)]
    print('Success!')
    for i, v in enumerate(existing_videos):
        print(f'{v.name}\t{frame_counts[i]} frames written to {frame_dirs[i]}')
    print('Could not process:' + "\n\t".join(non_existing))
