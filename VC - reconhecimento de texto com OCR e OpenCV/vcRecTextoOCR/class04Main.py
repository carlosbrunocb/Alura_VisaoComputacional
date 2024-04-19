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

img = cv2.imread('text-recognize/Imagens/Aula4-tabela_teste.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("Tabela-teste", rgb)

config_tesseract = '--tessdata-dir tessdata'
resultado = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
print(resultado)

fonte = 'text-recognize/Imagens/calibri.ttf'
padrao_data = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
min_conf = 25 #@param {type: 'slider', min: 0, max: 100}

img_copia = rgb.copy()
for i in range(0, len(resultado['text'])):
  confianca = float(resultado['conf'][i])
  if confianca > min_conf:
    texto = resultado['text'][i]

    if re.match(padrao_data, texto):
      x, y, img = caixa_texto(resultado, img_copia)
      img_copia = escreve_texto(texto, x, y, img_copia, fonte, 16)
    else:
      x, y, img_copia = caixa_texto(resultado, img_copia)

cv2.imshow("Box-Tabela-teste", img_copia)


cv2.waitKey(0)