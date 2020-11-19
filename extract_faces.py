from pathlib import Path
from mtcnn import MTCNN
from joblib import Parallel, delayed
from json import dump as jdump
from copy import deepcopy
import cv2

img_dir = Path('images/')
face_outputs = Path('tmp_faces/')
face_outputs.mkdir(parents=True, exist_ok=True)
embeddings_dir = Path('embeddings/')

img_paths = list(map(str, img_dir.glob("*")))
convert_images = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
images = Parallel(n_jobs=-1)(delayed(convert_images)(i) for i in img_paths)

detector = MTCNN()

outputs = [detector.detect_faces(i) for i in images]
outputs = {k: v for k, v in zip(img_paths, outputs)}

jdump(outputs, open(embeddings_dir/'out.json', 'w'), indent=4)
for k, faces in outputs.items():
    img_idx = img_paths.index(k)
    for it, face in enumerate(faces):
        bb = face['box']
        cropped_img = images[img_idx][bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        fn = 'cropped_' + Path(k[::-1].replace('.', f'.{it}_', 1)[::-1]).name
        cv2.imwrite(str(embeddings_dir/fn), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        bounding_box = face['box']
        keypoints = face['keypoints']
        image = deepcopy(images[img_idx])
        fn = Path(k[::-1].replace('.', f'.{it}_', 1)[::-1]).name

        cv2.rectangle(image,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                    (0,155,255),
                    2)

        cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

        cv2.imwrite(str(embeddings_dir/fn), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
