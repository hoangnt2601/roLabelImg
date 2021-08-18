import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import math
import cv2
import imutils
import os


def npxywha2vertex(box):
    """
    use radian
    X=x*cos(a)-y*sin(a)
    Y=x*sin(a)+y*cos(a)
    """
    batch = box.shape[0]

    center = box[:,:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = np.empty((batch,2), dtype=np.float32)
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)

    hori = np.empty((batch,2), dtype=np.float32)
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)

    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)

json_path = "/mnt/sdb1/Data/Phamacity_datasets/train/linhdam1_final.json"
data = {}
json_path = Path(json_path)
data_dir = json_path.parents[0].joinpath(json_path.stem)
with open(str(json_path), 'r') as f:
    json_data = json.load(f)

path_id = {d['id']:d['file_name'] for d in json_data['images']}
desc = f"{json_path.stem} - Scanning '{data_dir}' images and labels..."
for ann in tqdm(json_data['annotations'], desc=desc):
    image_path = data_dir.joinpath(path_id[ann['image_id']])
    # assert image_path.is_file(), f"File not found: {str(image_path)}"
    if not image_path.is_file():
        continue

    l = data.get(str(image_path), [np.empty((0, 9)), None, []])
    if l[1] is None:
        img = Image.open(image_path)
        shape = img.size
        l[1] = shape
    shape = l[1]
    bbox = ann['bbox']
    # bbox[0] /= shape[0]
    # bbox[1] /= shape[1]
    # bbox[2] /= shape[0]
    # bbox[3] /= shape[1]
    bbox[4] = bbox[4] / 180 * math.pi
    bbox = np.array(bbox)
    bbox = npxywha2vertex(bbox[np.newaxis])
    bbox = np.concatenate([np.zeros((1, 1)), bbox], axis=1)
    
    # l[0].append(bbox)
    l[0] = np.concatenate([l[0], np.array(bbox)])
    data[str(image_path)] = l

for image_path, det in data.items():
    image = cv2.imread(image_path)

    for conf, *xyxyxyxy in det[0]:
        xyxyxyxy = [int(i) for i in xyxyxyxy]
        pts = np.array([xyxyxyxy[i:i+2] for i in range(0, len(xyxyxyxy), 2)], np.int32)

        pts = pts.reshape((-1, 1, 2))
        # cv2.rectangle(image, (x1, y1), (x3, y3), (0, 0, 255), 3)
        # cv2.putText(image, f'{conf:.2f}', (xyxyxyxy[0],xyxyxyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        image = cv2.polylines(image, [pts], True, (0, 0, 255), 2)
        
    cv2.imshow("image", imutils.resize(image, width=720))
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
cv2.destroyAllWindows()