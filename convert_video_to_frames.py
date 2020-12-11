from types import FrameType
import cv2
import pandas as pd
from pandas.core import frame
from psutil import cpu_count
from joblib import Parallel, delayed
from pathlib import Path
from sys import argv
import os

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

    capture.set(1, start)  # set the starting frame of the capture
    frame_no = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    save_dir = frames_dir/f'{(frame_no//1000):03d}'/f'{(frame_no%1000//100)}'
    save_dir.mkdir(parents=True, exist_ok=True)
    every = 1
    while frame_no < end:  # lets loop through the frames until the end
        success, image = capture.read()  # read an image from the capture
        if not (frame_no in frame_list):
            frame_no += 1
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
            if not save_path.exists() or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(str(save_path), image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame_no += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved

if __name__ == '__main__':
    input_csv = pd.read_csv(argv[1], index_col=0).reset_index(drop=True)
    frame_list = input_csv.groupby('file').frame.unique()
    videos = list(Path('videos/').glob('*'))
    output = Path('images/')
    num_processes = cpu_count()
    if len(videos) < num_processes:
        num_processes = len(videos)
    existing_frames = {}
    non_existing = []
    existing_videos = []
    for v in videos:
        if v.name in frame_list.index:
            existing_frames[v.name] = frame_list[v.name]
            existing_videos.append(v)
        else:
            non_existing.append(str(v))
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
