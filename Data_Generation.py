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

    blur = cv2.GaussianBlur(image, (3, 3), 0)
    _, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("Data/BlackMasked/" + img, threshold)
