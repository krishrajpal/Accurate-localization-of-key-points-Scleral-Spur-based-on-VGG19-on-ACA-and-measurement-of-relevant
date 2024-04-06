import numpy as np
import cv2
import random
import  os
import csv

pathes = []
images = []
labels = []

def calculate_new_coordinates(original_x, original_y):
    width = 496
    middle = width // 2
    if original_x < middle:
        return [original_x, original_y]
    else:
        new_x = width - original_x - 1
        return [new_x, original_y]

# Define the path to your CSV file
csv_file_path = "/Users/swc/Desktop/projects/project/Scleral Spur Coordiantes_Manual  - Sheet1.csv"
# Read the CSV file
with open(csv_file_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    # Skip the header row if it exists
    next(csv_reader)
    # Read each row in the CSV file
    for row in csv_reader:
        # Extract data from the row
        image_id = int(row[0])
        left_ss_x = int(row[1])
        left_ss_y = int(row[2])
        right_ss_x = int(row[3])
        right_ss_y = int(row[4])
        
        # Append the data to respective lists
        pathes.append("output_folder/" + str(image_id) + "_left_half.png")
        pathes.append("output_folder/" + str(image_id) + "_mirrored_right_half.png")
        labels.append([left_ss_x, left_ss_y])
        labels.append(calculate_new_coordinates(right_ss_x, right_ss_y))


for idx, path in enumerate(pathes):
    # iidx = int(path.split('\\')[1].split('.')[0]) - 1
    # print(pathes)
    # print(labels)
    img = cv2.imread(path)
    images.append(img)



x = np.array(images)
y = np.array(labels)
# x_shuffle = x[index]
# y_shuffle = y[index]
# print(np.shape(img_crop_total))
# # print(np.shape(label_crop_total))

print(x[0:3].shape)
print(x.shape)
print(y[0:3].shape)
print(y.shape)
# 拆分成训练集和测试集数据
p = int(len(x) * 0.85)
train_x = x[:p]
train_y = y[:p]
test_x = x[p:]
test_y = y[p:]


if not os.path.exists('npy'):
    os.makedirs('npy')
np.save('npy/x_train', train_x)
np.save('npy/y_train', train_y)
np.save('npy/x_test', test_x)
np.save('npy/y_test', test_y)

# for i in range(10):
#     img = img_crop_total[i]
#     clone_img_1 = img.copy()
#     print(img.shape)
#     cv2.circle(clone_img_1, (label_crop_total[i][0], label_crop_total[i][1]), 3, (0, 0, 255), -1)
#     cv2.imshow('img', clone_img_1)
#     cv2.waitKey(0)