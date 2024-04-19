from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# print()
# cv2.imshow("Google color gray", gray)
#
# text = pytesseract.image_to_string(gray)
# print(text)

img = cv2.imread("text-recognize/Imagens/Aula2-undersampling.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Complex Text", rgb)

print(rgb.shape)
print()

text = pytesseract.image_to_string(rgb, lang='por')
print(text)

# PSM - Page Segmentation Mode
# Testing some types of PSM available
img = cv2.imread("text-recognize/Imagens/Aula2-undersampling.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Complex Text", rgb)

texto = pytesseract.image_to_string(rgb, lang='por')
print(texto)

config_tesseract = '--psm 6'
texto = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(texto)

# Exemplo com PSM errado
config_tesseract = '--psm 7'
texto = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(texto)

config_tesseract = '--psm 8'
texto = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(texto)

config_tesseract = '--psm 10'
texto = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(texto)

# Mais exemplos
img = cv2.imread("text-recognize/Imagens/Aula2-Saida.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(rgb.shape)
print()
cv2.imshow("Exit Box", rgb)

config_tesseract = '--psm 7'
texto = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(texto)

cv2.waitKey(0)

#############
# Using PIL module
# OSD: image metadata about text data
img = Image.open(".venv/text-recognize/Imagens/Aula2-livro.png")
plt.imshow(img)

# pytesseract.image_to_osd(img) ERROR:
#  + There is a bug with metadata image PIL when we try use with OSD tesseract
print(pytesseract.image_to_osd("text-recognize/Imagens/Aula2-livro.png"))

plt.waitforbuttonpress(0)
