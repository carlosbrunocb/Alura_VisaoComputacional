from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output
import pytesseract
import numpy as np
import cv2

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

min_conf = 25 #@param {type: 'slider', min: 0, max: 100}

# function: caixa_texto [bounding box]
# input:
#  - resultado :: dictionary obtained of output pytesseract.image_to_data() function
#  - img :: image
#  - cor = (255, 100, 0) :: tuple in BGR format that defines the color of the bounding box
def caixa_texto(resultado, img, cor = (255, 100, 0)):
  x = resultado['left'][i]
  y = resultado['top'][i]
  w = resultado['width'][i]
  h = resultado['height'][i]

  # function rectangle of OpenCV used to draw a bounding box in the image
  # input:
  #  - img        :: image input
  #  - (x, y)     :: ponto inicial
  #  - (x+w, y+h) :: ponto final
  #  - cor        :: bounding box color
  #  - 2          :: bounding box thickness
  cv2.rectangle(img, (x, y), (x+w, y+h), cor, 2)

  return x, y, img

# Inserting new fonts using the PIL Lib
# To apply it using putText method
fonte = 'text-recognize/Imagens/calibri.ttf'
img = cv2.imread('text-recognize/Imagens/Aula1-ocr.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

config_tesseract = '--tessdata-dir tessdata'
resultado = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
print(resultado)

cv2.imshow("Aula1-OCR", rgb)

def escreve_texto(texto, x, y, img, fonte, tamanho_texto=32):
  fonte = ImageFont.truetype(fonte, tamanho_texto)
  img_pil = Image.fromarray(img) # Convert an array image to PIL
  draw = ImageDraw.Draw(img_pil) # Object to draw in the image
  draw.text((x, y - tamanho_texto), texto, font = fonte) # Draw a text
  img = np.array(img_pil) # Convert PIL object to array image
  return img

img_copia = rgb.copy()

for i in range(len(resultado['text'])):
  confianca = float(resultado['conf'][i])
  if confianca > min_conf:
    x, y, img = caixa_texto(resultado, img_copia)
    texto = resultado['text'][i]
    # cv2.putText(img_copia, texto, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255))
    img_copia = escreve_texto(texto, x, y, img, fonte)


logo_tesseract = 'images_tesseract/logo_tesseract.png'
cv2.imwrite(logo_tesseract, img_copia)

cv2.imshow("Aula1-OCR - text", img_copia)

cv2.waitKey(0)

