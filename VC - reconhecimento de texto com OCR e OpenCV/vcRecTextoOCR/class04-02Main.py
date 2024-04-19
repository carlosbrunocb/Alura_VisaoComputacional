from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output
import pytesseract
import re
import numpy as np
import cv2

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def caixa_texto(resultado, img, cor = (255, 100, 0)):
  x = resultado['left'][i]
  y = resultado['top'][i]
  w = resultado['width'][i]
  h = resultado['height'][i]

  cv2.rectangle(img, (x, y), (x+w, y+h), cor, 2)

  return x, y, img

def escreve_texto(texto, x, y, img, fonte, tamanho_texto=32):
  fonte = ImageFont.truetype(fonte, tamanho_texto)
  img_pil = Image.fromarray(img) # Convert an array image to PIL
  draw = ImageDraw.Draw(img_pil) # Object to draw in the image
  draw.text((x, y - tamanho_texto), texto, font = fonte) # Draw a text
  img = np.array(img_pil) # Convert PIL object to array image

  return img

img = cv2.imread('text-recognize/Imagens/Aula4-caneca2.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("caneca", rgb)

config_tesseract = '--tessdata-dir tessdata --psm 6'
resultado = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
print(resultado)

fonte = 'text-recognize/Imagens/calibri.ttf'
min_conf = 40 #@param {type: 'slider', min: 0, max: 100}

img_copia = rgb.copy()

for i in range(0, len(resultado['text'])):
  confianca = float(resultado['conf'][i])

  if confianca > min_conf:
    x, y, img = caixa_texto(resultado, img_copia)

    texto = resultado['text'][i]
    img_copia = escreve_texto(texto, x, y, img_copia, fonte)

cv2.imshow("caneca-box", img_copia)

img_copia01 = rgb.copy()

for i in range(0, len(resultado['text'])):
  confianca = float(resultado['conf'][i])

  if confianca > min_conf:

    texto = resultado['text'][i]
    if not texto.isspace() and len(texto) > 1:
      x, y, img = caixa_texto(resultado, img_copia01)
      img_copia01 = escreve_texto(texto, x, y, img_copia01, fonte)

cv2.imshow("caneca-box-more-2-char", img_copia01)


cv2.waitKey(0)