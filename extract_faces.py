from pathlib import Path
from joblib import Parallel, delayed
from json import dump as jdump
from json import load as jload
from yaml import load as yload
from yaml import SafeLoader
from copy import deepcopy
import numpy as np
import cv2
from mtcnn import MTCNN

def get_closest_to_coord_face(faces, x_center, y_center):
    coords = [face["keypoints"]["nose"] for face in faces]
    coords = np.array(coords) 
    diff = coords-np.array([y_center, x_center])
    idx = np.linalg.norm(diff, axis=1).argmin()
    return faces[idx]

# Get configuration
config_file = Path('./config.yml')
assert config_file.exists(), "ERROR!\nMake sure the 'config.yml' file exists in the current folder."
config = yload(open(config_file, 'r'))

img_dir = Path(config['img_dir'])
video_dir = Path(config['video_dir'])
embeddings_dir = Path(config['embeddings_dir'])
y_center = config['y_center']
x_center = config['x_center']

# Set up config
videos = video_dir.glob('*')
face_outputs = Path('tmp_faces/')
face_outputs.mkdir(parents=True, exist_ok=True)

img_paths = list(map(str, img_dir.rglob("*.jpg"))) + list(map(str, img_dir.rglob("*.jpeg")))

convert_images = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

detector = MTCNN()

def crop_image(pt):
    # detector = MTCNN()
    fn = Path(str(pt).replace(str(img_dir), str(embeddings_dir)))
    if fn.exists():
        return None
    image = convert_images(pt)
    resolution = list(image.shape[:-1])[::-1]
    detection = detector.detect_faces(image)
    for face in detection:
        if len(face["keypoints"]["nose"]) != 2:
            return {str(pt): {'face': [], 'resolution': resolution}}
    
    face = get_closest_to_coord_face(
        faces = detection,
        x_center = x_center * resolution[1],
        y_center = y_center * resolution[0]
    )
    eyes = np.array([face['keypoints']['left_eye'], face['keypoints']['right_eye']])
    buf = np.std(eyes).astype(int)
    # buf[0] = buf[0]/2
    # buf[1] = buf[1]*10
    # buf = (buf * resolution).astype(int)
    xy_min, xy_max = eyes.min(axis=0) - buf, eyes.max(axis=0) + buf
    cropped_img = image[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]
    # fn = 'cropped_' + Path(k[::-1].replace('.', f'.{it}_', 1)[::-1]).name
    fn = Path(str(pt).replace(str(img_dir), str(embeddings_dir)))
    if not fn.parent.exists():
        fn.parent.mkdir(parents=True, exist_ok=True)
    print(f'Cropped {pt} in {fn}.')
    try:
        cv2.imwrite(str(fn), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    except:
        print(f"Error writing {embeddings_dir/fn}")
    return {str(pt): {'face': face, 'resolution': resolution}}

output = Parallel(n_jobs=-1, require='sharedmem')(delayed(crop_image)(pt) for pt in img_paths)
outputs = {}
for out in output:
    outputs.update(out)

jdump(outputs, open(embeddings_dir/'out.json', 'w'), indent=4)