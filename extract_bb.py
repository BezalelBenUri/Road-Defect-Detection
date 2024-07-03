# Import Libaries
import os
import json
import csv

# Get the current working directory
json_dir = os.getcwd()

# Prepare the CSV file
csv_file = open("annotations.csv", mode = 'w', newline = '')
csv_writer = csv.writer(csv_file)

# Write the header
csv_writer.writerow(["file_name", "image_height", "image_width", "bbox_x", "bbox_y", "bbox_width", "bbox_height"])

# Iterate through each file in the current directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        
        # Load the COCO JSON file
        with open(json_path) as f:
            coco_data = json.load(f)

        # Create a dictionary to map image_id to file_name, height, and width
        image_info = {image['id']: image for image in coco_data['images']}

        # Extract bounding box data and write to CSV
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            image = image_info[image_id]
            file_name = image['file_name']
            image_height = image['height']
            image_width = image['width']
            bbox = annotation['bbox']

            csv_writer.writerow([file_name, image_height, image_width, bbox[0], bbox[1], bbox[2], bbox[3]])

# Close the CSV file
csv_file.close()
