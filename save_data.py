import numpy as np
import cv2
import random
import  os
import csv

pathes = []
img_crop_total = []
label_crop_total = []
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
csv_file_path = "E:\Projects\Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant\Scleral Spur Coordiantes_Manual  - Sheet1.csv"
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
    # print(img.shape)
    if img.shape[0] == 248:
        for i in range(5):
            # 进行中心的crop
            if i != 0:
                x = img.shape[0]
                y = img.shape[1]
                random_scale_x = random.randrange(0, int(x - 224))
                random_scale_y = random.randrange(0, int(y - 224))
                img_crop = img[random_scale_y:random_scale_y + 224, random_scale_x:random_scale_x + 224, :]
                val1 = labels[idx][0] - random_scale_x
                val2 = labels[idx][1] - random_scale_y
                label = [val1, val2]
                img_crop_total.append(img_crop)
                label_crop_total.append(label)
            else:
                x = img.shape[0]
                y = img.shape[1]
                scale_x = int((x - 224) / 2)
                scale_y = int((y - 224) / 2)
                img_crop = img[scale_y:scale_y + 224, scale_x:scale_x + 224]
                img_crop_total.append(img_crop)
                label = [labels[idx][0] - scale_x, labels[idx][1] - scale_y]
                label_crop_total.append(label)
    else:
        img_crop_total.append(img)
        label_crop_total.append(labels[idx])

# 进行洗牌操作
index = np.arange(0, len(img_crop_total))
random.shuffle(index)
x = np.array(img_crop_total)
y = np.array(label_crop_total)
x_shuffle = x[index]
y_shuffle = y[index]
# print(np.shape(img_crop_total))
# # print(np.shape(label_crop_total))

print(x_shuffle[0:3].shape)
# 拆分成训练集和测试集数据
p = int(len(x) * 0.85)
train_x = x_shuffle[:p]
train_y = y_shuffle[:p]
test_x = x_shuffle[p:]
test_y = y_shuffle[p:]


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