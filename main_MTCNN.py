import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN

# Завантаження зображення
image = cv2.imread('assets/group.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Використання MTCNN для розпізнавання облич
detector = MTCNN()
faces = detector.detect_faces(image_rgb)

# Відображення результатів
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Виявлення облич')
plt.show()
