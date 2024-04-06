import numpy as np
import cv2
import random
import  os
import csv

data_folder = 'output_folder'
label_csv = 'Scleral Spur Coordiantes_Manual  - Sheet1.csv'

# Load image file paths from the data folder
image_paths = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.endswith('.png')]

# Calculate New Coordinates_Mirrored Images
def calculate_new_coordinates(original_x, original_y):
    width = 496
    middle = width // 2
    if original_x < middle:
        return [original_x, original_y]
    else:
        new_x = width - original_x - 1
        return [new_x, original_y]

# Load labels from the CSV file
labels = []
with open(label_csv, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        left_ss_x = int(row[1])
        left_ss_y = int(row[2])
        right_ss_x = int(row[3])
        right_ss_y = int(row[4])
        labels.append([left_ss_x, left_ss_y])
        labels.append(calculate_new_coordinates(right_ss_x, right_ss_y))

img_crop_total = []
label_crop_total = []
k = 0

for image_path in image_paths:
    # Load image using OpenCV
    img = cv2.imread(image_path)

    if img.shape[0] == 248:
        for i in range(5):
            if i != 0:
                x = img.shape[0]
                y = img.shape[1]
                random_scale_x = random.randrange(0, int(x - 224))
                random_scale_y = random.randrange(0, int(y - 224))
                img_crop = img[random_scale_y:random_scale_y + 224, random_scale_x:random_scale_x + 224, :]
                label = [labels[k][0] - random_scale_x, labels[k][1] - random_scale_y]

                img_crop_total.append(img_crop)
                label_crop_total.append(label)
            else:
                x = img.shape[0]
                y = img.shape[1]
                scale_x = int((x - 224) / 2)
                scale_y = int((y - 224) / 2)
                img_crop = img[scale_y:scale_y + 224, scale_x:scale_x + 224]
                label = [labels[k][0] - scale_x, labels[k][1] - scale_y]
                img_crop_total.append(img_crop)
                label_crop_total.append(label)
        k += 1
    else:
        img_crop_total.append(img)
        label_crop_total.append(labels[k])

# Shuffle data
index = np.arange(0, len(img_crop_total))
np.random.shuffle(index)
x = np.array(img_crop_total)
y = np.array(label_crop_total)
x_shuffle = x[index]
y_shuffle = y[index]

# Train-test split
p = int(len(x) * 0.85)
train_x = x_shuffle[:p]
train_y = y_shuffle[:p]
test_x = x_shuffle[p:]
test_y = y_shuffle[p:]

# Save data
if not os.path.exists('npy'):
    os.makedirs('npy')
np.save('npy/x_train', train_x)
np.save('npy/y_train', train_y)
np.save('npy/x_test', test_x)
np.save('npy/y_test', test_y)