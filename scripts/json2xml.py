from xml.etree import ElementTree
from xml.dom import minidom
import os 
from utils import parse_json
from pathlib import Path


json_path = "/mnt/sdb1/Data/Phamacity_datasets/timecity2_final.json"

data = parse_json(json_path)

for image_path in list(data.keys()):
    image_path = Path(image_path)
    xml_path = image_path.with_suffix('.xml')
    annos, shape = data[str(image_path)]

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
    width.text = str(shape[0])
    height = ElementTree.SubElement(size, "height")
    height.text = str(shape[1])
    depth = ElementTree.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ElementTree.Element("segmented")
    segmented.text = "0"
    root.append(segmented)

    for label in annos:
        label = label[1:].tolist()
        
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
        print(xml_path)
        f.write(reparsed.toprettyxml(indent="  "))
