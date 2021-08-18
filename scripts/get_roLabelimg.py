'''
Lấy kết quả detect làm nhãn
'''
from pathlib import Path
from tqdm import tqdm
import cv2
from xml.etree import ElementTree
from xml.dom import minidom
import math
import os
import sys
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, ".."))

from scripts.detect import RapidDetect


def gen_xml(dts, xml_path):
    root = ElementTree.Element("annotation", {"verified": "no"})
    
    folder = ElementTree.Element("folder")
    folder.text = image_path.parents[0].stem
    root.append(folder)
    
    filename = ElementTree.Element("filename")
    filename.text = image_path.stem
    root.append(filename)
    
    path = ElementTree.Element("path")
    path.text = str(image_path)
    root.append(path)
    
    source = ElementTree.Element("source")
    root.append(source)
    database = ElementTree.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ElementTree.Element("size")
    root.append(size)
    width = ElementTree.SubElement(size, "width")
    width.text = str(w)
    height = ElementTree.SubElement(size, "height")
    height.text = str(h)
    depth = ElementTree.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ElementTree.Element("segmented")
    segmented.text = "0"
    root.append(segmented)

    for label in dts:
        label = label[:5].tolist()
        label[-1] = label[-1] * math.pi / 180
        
        object = ElementTree.Element("object")
        root.append(object)
        type = ElementTree.SubElement(object, "type")
        type.text = "robndbox"
        name = ElementTree.SubElement(object, "name")
        name.text = "person"
        pose = ElementTree.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ElementTree.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ElementTree.SubElement(object, "difficult")
        difficult.text = "0"
        robndbox = ElementTree.SubElement(object, "robndbox")
        robndbox_field = ["cx", "cy", "w", "h", "angle"]
        for i in range(5):
            robndbox_i = ElementTree.SubElement(robndbox, robndbox_field[i])
            robndbox_i.text = str(label[i])
    
    rough_string = ElementTree.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(xml_path, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))


weights_path = os.path.join(pwd, "../weights/pL1_HBCP608_Apr14_6000.ckpt")
image_dir = Path("/datasets/fish-eye/lyquocsu/lyquocsu_quay")
rd = RapidDetect(weights_path, input_size=608, conf_thres=0.15)

start_idx = 1056
for image_path in tqdm(image_dir.glob("lyquocsu_quay_*.jpg")):
    idx = int(image_path.stem.split("_")[-1])
    if idx < start_idx:
        continue

    xml_path = image_path.with_suffix('.xml')
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    dts = rd.detect(image)

    gen_xml(dts, xml_path)


