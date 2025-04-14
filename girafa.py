import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# Carregar e converter imagem para RGB
img = cv2.imread('GIRAFA.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Escala de cinza
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Blur
img_blur = cv2.blur(img_gray, ksize=(5, 5))

# Binarização (threshold)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a/2*1.7, a, cv2.THRESH_BINARY_INV)

# Detecção de bordas com Canny (usando imagem borrada)
edges_blur = cv2.Canny(image=img_blur, threshold1=a/2, threshold2=a/2)

# Contornos 
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)

etapas = [img, img_gray, img_blur, thresh, edges_blur, final]
titulos = ['Original', 'Cinza', 'Blur', 'Threshold', 'Canny', 'Contornos']

plt.figure(figsize=(15, 8))
for i in range(len(etapas)):
    plt.subplot(2, 3, i+1)
    cmap = 'gray' if len(etapas[i].shape) == 2 else None
    plt.imshow(etapas[i], cmap=cmap)
    plt.title(titulos[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
