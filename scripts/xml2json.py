import math
import os
from pathlib import Path
from pprint import pprint
from xml import etree
from xml.dom import minidom
from xml.etree import ElementTree

import cv2
import imutils
import numpy as np
from imutils import paths
from PIL import Image
from tqdm import tqdm

from utils import etree_to_dict, npxywha2vertex, parse_json

images_path = paths.list_images("/mnt/sdb1/Data/Phamacity_datasets/timecity2_final")

anns = {}

for image_path in tqdm(images_path):
    image_path = Path(image_path)
    xml_path = image_path.with_suffix('.xml')
    # print(xml_path)
    if not os.path.exists(xml_path):
        os.remove(xml_path.replace("xml", "jpg"))
        continue
    tree = ElementTree.parse(xml_path)
    data = etree_to_dict(tree.getroot())["annotation"]

    image_path = data['path']
    image_id = data['filename']
    objs = []
    if isinstance(data['object'], dict):
        objs.append(data['object'])
    else:
        objs = data['object']

    l = data.get(str(image_path), [np.empty((0, 9)), None, []])
    if l[1] is None:
        img = Image.open(image_path)
        if img is None:
            print(image_path)
            continue
    
    for obj in objs:
        robndbox = obj['robndbox']
        # print(robndbox)

        bbox = np.asarray([int(float(robndbox['cx'])), int(float(robndbox['cy'])),
                        int(float(robndbox['w'])), int(float(robndbox['h'])),
                        float(robndbox['angle'])])
        # bbox[4] = bbox[4] / 180 * math.pi
        bbox = np.array(bbox)
        bbox = npxywha2vertex(bbox[np.newaxis])
        bbox = np.concatenate([np.zeros((1, 1)), bbox], axis=1)
        
        l[0] = np.concatenate([l[0], np.array(bbox)])
        anns[str(image_path)] = l

for image_path, det in anns.items():
    print(image_path)
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
