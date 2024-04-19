import pytesseract
import numpy as np
import cv2

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread("text-recognize/Imagens/Aula1-teste.png")
cv2.imshow("Text", img)

text = pytesseract.image_to_string(img)
print(text)

img = cv2.imread("text-recognize/Imagens/Aula1-ocr.png") # read image as BGR matrix
print(img.shape)
print()
cv2.imshow("Google", img)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
print()
cv2.imshow("Google color invert", rgb)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
print()
cv2.imshow("Google color gray", gray)

text = pytesseract.image_to_string(rgb)
print(text)

text = pytesseract.image_to_string(gray)
print(text)

img = cv2.imread("text-recognize/Imagens/Aula2-undersampling.png")
cv2.imshow("Complex Text", img)


cv2.waitKey(0)

