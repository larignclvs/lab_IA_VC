import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# Carregar e converter imagem para RGB
img = cv2.imread('satelite.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Escala de cinza
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Desfoque (Blur)
tam_kernel = (5, 5)
img_blur = cv2.blur(img_gray, ksize=tam_kernel)

# Limiarização (Threshold)
valor_max = img_gray.max()
_, thresh = cv2.threshold(img_gray, valor_max * 0.85, valor_max, cv2.THRESH_BINARY_INV)

# Detecção de bordas com Canny
edges = cv2.Canny(img_gray, 100, threshold2=valor_max)

# Encontrar contornos
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img_rgb.copy()
cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)

# Exibir resultados
etapas = [img_rgb, img_gray, img_blur, thresh, edges, img_contours]
titulos = ['Original RGB', 'Escala de Cinza', 'Blur', 'Threshold', 'Canny', 'Contornos']

plt.figure(figsize=(15, 8))
for i in range(len(etapas)):
    plt.subplot(2, 3, i+1)
    cmap = 'gray' if len(etapas[i].shape) == 2 else None
    plt.imshow(etapas[i], cmap=cmap)
    plt.title(titulos[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
