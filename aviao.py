import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Carrega imagem e converte para RGB
foto = cv2.imread('./Aviao.jpeg')
foto_rgb = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)

# Converte para escala de cinza e aplica limiarização invertida
cinza = cv2.cvtColor(foto_rgb, cv2.COLOR_RGB2GRAY)
valor_max = cinza.max()
_, binaria = cv2.threshold(cinza, valor_max * 0.85, valor_max, cv2.THRESH_BINARY_INV)

# Aplica abertura morfológica para limpar a imagem
tam_kernel = 5
estrutura = np.ones((tam_kernel, tam_kernel), np.uint8)
binaria_filtrada = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, estrutura)

# Desfoque simples
cinza_desfocado = cv2.blur(cinza, (tam_kernel, tam_kernel))

# Bordas com Canny
bordas = cv2.Canny(cinza, valor_max / 3, valor_max)

# Encontra contornos e desenha
conts, _ = cv2.findContours(bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
conts = sorted(conts, key=cv2.contourArea, reverse=True)
imagem_final = foto_rgb.copy()
cv2.drawContours(imagem_final, conts, -1, (255, 0, 0), 2)

# Exibição
figuras = [foto_rgb, cinza_desfocado, cinza, bordas, imagem_final]
cols = math.ceil(len(figuras)**0.5)
rows = cols if cols * cols - len(figuras) <= cols else cols - 1

for i in range(len(figuras)):
    plt.subplot(rows, cols, i+1)
    plt.imshow(figuras[i], cmap='gray')
    plt.axis('off')

plt.show()
