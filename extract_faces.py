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
images = Parallel(n_jobs=-1)(delayed(convert_images)(i) for i in img_paths)

detector = MTCNN()

outputs = Parallel(n_jobs=-1)(delayed(detector.detect_faces)(i) for i in images)

# jdump(outputs, open(embeddings_dir/'pure_out.json', 'w'), indent=4)
outputs = {k: {'faces': v, 'resolution': list(i.shape[:-1])[::-1]} for k, v, i in zip(img_paths, outputs, images)}
for k, v in outputs.items():
    outputs[k] = {}
    outputs[k]['face'] = get_closest_to_coord_face(
        faces = v['faces'],
        x_center = x_center * v['resolution'][1],
        y_center = y_center * v['resolution'][0]
    )
    outputs[k]['resolution'] = v['resolution']

jdump(outputs, open(embeddings_dir/'out.json', 'w'), indent=4)

for k, v in outputs.items():
    img_idx = img_paths.index(k)
    face = v['face']
    resolution = v['resolution']
    eyes = np.array([face['keypoints']['left_eye'], face['keypoints']['right_eye']])
    buf = np.std(eyes/resolution, axis=0)
    buf[1] = buf[1]*4
    buf = (buf * resolution).astype(int)
    xy_min, xy_max = eyes.min(axis=0) - buf, eyes.max(axis=0) + buf
    cropped_img = images[img_idx][xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]
    # fn = 'cropped_' + Path(k[::-1].replace('.', f'.{it}_', 1)[::-1]).name
    fn = Path(str(k).replace(str(img_dir), str(embeddings_dir)))
    if not fn.parent.exists():
        fn.parent.mkdir(parents=True, exist_ok=True)
    print(f'Cropped {k} in {fn}.')
    try:
        cv2.imwrite(str(fn), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    except:
        print(f"Error writing {embeddings_dir/fn}")
    # bounding_box = face['box']
    # keypoints = face['keypoints']
    # image = deepcopy(images[img_idx])
    # fn = Path(k[::-1].replace('.', f'.{it}_', 1)[::-1]).name

    # cv2.rectangle(image,
    #             (bounding_box[0], bounding_box[1]),
    #             (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
    #             (0,155,255),
    #             2)

    # cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    # try:
    #     cv2.imwrite(str(embeddings_dir/fn), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # except:
    #     print(f"Error writing {embeddings_dir/fn}")
