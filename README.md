# EyeTracker
Eye tracker for cognitive science experiments with babies

## Requirements
Run `pip install -r requirements.txt` in the project folder to install all dependencies.

## Usage
Type `python process_frames.py -h` to see the options list.

In essence you should type `python process_frames.py file1.mp4 file2.avi` etc.
Use the `-e` command to specify how many frames to skip, the default is to skip 20 frames, so that we still have subsecond sampling.
