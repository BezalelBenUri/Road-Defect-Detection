import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Define the class mapping
class_mapping = {
    "D40": 0,
    "D00": 1,
    "D10": 2,
    "D20": 3
}

def convert_bbox(size, box):
    """
    Converts bounding box coordinates from pixel space to normalized space.
    
    Args:
      size: Tuple of two integers representing the image size (width, height).
      box: Tuple of four floats representing the bounding box coordinates (xmin, xmax, ymin, ymax).
    
    Returns:
      Tuple of four floats representing the normalized bounding box coordinates (x_center, y_center, width, height).
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, annotations_folder):
    """
    Parses an XML annotation file and converts it to a YOLO format text file.
    
    Args:
      xml_file: Path to the XML annotation file.
      annotations_folder: Path to the folder where the YOLO format text files will be saved.
      
    Returns:
      Tuple of two strings:
          - image_name: Name of the image corresponding to the annotation file.
          - bounding_boxes: List of bounding boxes in YOLO format (class_id x_center y_center width height).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    file_name = root.find("filename").text
    txt_file = os.path.join(annotations_folder, file_name.replace(".jpg", ".txt"))

    print(f"Processing file: {file_name}")

    bounding_boxes = []

    with open(txt_file, "w") as out_file:
        for obj in root.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue
            cls = obj.find("name").text
            if cls not in class_mapping:
                continue
            cls_id = class_mapping[cls]
            xmlbox = obj.find("bndbox")
            b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
                 float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + "\n")
            bounding_boxes.append(b)

    print(f"Saved annotation to: {txt_file}")
    return file_name, bounding_boxes

def convert_dataset(data_folder, annotations_folder):
    """
    Converts all XML annotation files in a dataset folder to YOLO format text files.
    Args:
      data_folder: Path to the folder containing the XML annotation files.
      annotations_folder: Path to the folder where the YOLO format text files will be saved.
    """
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    for filename in os.listdir(data_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(data_folder, filename)
            image_name, bounding_boxes = convert_annotation(xml_file, annotations_folder)
            visualize_annotation(data_folder, image_name, bounding_boxes)


def remove_empty_annotations(data_folder, annotations_folder):
    """
    Removes empty annotation files and their corresponding image files.

    This function iterates through all text files in the specified annotations folder.
    If a text file is empty (has a size of 0 bytes), it is deleted along with its
    corresponding image file (with the same name but .jpg extension) from the data folder.

    Args:
        data_folder (str): Path to the folder containing the image files.
        annotations_folder (str): Path to the folder containing the annotation files.
    """

    for txt_file in os.listdir(annotations_folder):
        txt_path = os.path.join(annotations_folder, txt_file)
        if os.path.getsize(txt_path) == 0:
            image_file = txt_file.replace(".txt", ".jpg")
            image_path = os.path.join(data_folder, image_file)
            os.remove(txt_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            print(f"Removed empty annotation and corresponding image: {txt_file}, {image_file}")


def visualize_annotation(data_folder, image_name, bounding_boxes):
    """
      Visualizes an image with its corresponding bounding boxes.

      Args:
        data_folder: Path to the folder containing the image.
        image_name: Name of the image file.
        bounding_boxes: List of bounding boxes, where each bounding box is a tuple of (xmin, xmax, ymin, ymax).
    """
    image_path = os.path.join(data_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image {image_name}")
        return

    for bbox in bounding_boxes:
        x_min, x_max, y_min, y_max = bbox
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(image_name)
    plt.show()


# Set the path to your data folder
data_folder = "Data"
annotations_folder = os.path.join(data_folder, "Annotations")
convert_dataset(data_folder, annotations_folder)