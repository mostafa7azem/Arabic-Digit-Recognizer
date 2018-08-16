import os
import cv2

data_path = os.path.join("Data", "Original")

imgs = os.listdir(data_path)

for img in imgs:
    image = cv2.imread(os.path.join(data_path, img))
    image = cv2.resize(image, (48, 64))
    cv2.imwrite("Data/Scaled/" + img, image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Data/Grayed/" + img, image)
