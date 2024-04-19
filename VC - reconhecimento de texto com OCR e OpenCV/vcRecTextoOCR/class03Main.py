from pytesseract import Output
import pytesseract
import numpy as np
import cv2

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('text-recognize/Imagens/Aula3-testando.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Hand Text", rgb)

config_tesseract = '--tessdata-dir tessdata'
resultado = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
print(resultado)
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

img_copia = rgb.copy()

for i in range(len(resultado['text'])):
  confianca = float(resultado['conf'][i])
  if confianca > min_conf:
    x, y, img = caixa_texto(resultado, img_copia)
    texto = resultado['text'][i]
    cv2.putText(img_copia, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

cv2.imshow("Bounding Box", img)

# Inserting new fonts in OpenCV
# To apply it using putText method
img = cv2.imread('text-recognize/Imagens/Aula1-ocr.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

config_tesseract = '--tessdata-dir tessdata'
resultado = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
resultado

cv2.imshow("Aula1-OCR", rgb)
cv2.waitKey(0)

