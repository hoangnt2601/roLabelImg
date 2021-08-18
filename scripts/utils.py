import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math
from PIL import Image
	


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


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


def parse_json(json_path):
    # path json label to {"img_path": [array, shape]}, array: (n, 6) - label,cx,cy,w,h,a(rad)
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

        l = data.get(str(image_path), [np.empty((0, 6)), None])
        if l[1] is None:
            img = Image.open(image_path)
            shape = img.size
            l[1] = shape
        shape = l[1]
        bbox = ann['bbox']
        bbox[4] = bbox[4] / 180 * math.pi
        bbox = np.array(bbox)[np.newaxis]
        bbox = np.concatenate([np.zeros((1, 1)), bbox], axis=1)
        
        l[0] = np.concatenate([l[0], np.array(bbox)])
        data[str(image_path)] = l
    return data
