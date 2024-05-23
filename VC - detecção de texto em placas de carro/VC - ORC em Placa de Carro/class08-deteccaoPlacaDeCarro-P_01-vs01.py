from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import pytesseract
import re
import os

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# '''
# Função escreve_texto: escreve um texto na imagem
#  @returna
#     img :: imagem do bounding box
# '''
def escreve_texto(texto, x, y, img, fonte_dir, cor=(50, 50, 255), tamanho=16):
  fonte = ImageFont.truetype(fonte_dir, tamanho)
  img_pil = Image.fromarray(img)
  draw = ImageDraw.Draw(img_pil)
  draw.text((x, y-tamanho), texto, font = fonte, fill = cor)
  img = np.array(img_pil)

  return img


img = cv2.imread('imagens/placa_carro2.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
altura, largura = img.shape[:2]
imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Placa do Carro - RGB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - RGB', largura, altura)
cv2.imshow("Placa do Carro - RGB", img)

cv2.namedWindow('Placa do Carro - Gray', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - Gray', largura, altura)
cv2.imshow("Placa do Carro - Gray", imagem)

bordas = cv2.Canny(imagem, 100, 200)
cv2.namedWindow('Placa do Carro - Bordas de Canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - Bordas de Canny', largura, altura)
cv2.imshow("Placa do Carro - Bordas de Canny", bordas)

contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:10]

for contorno in contornos:
  epsilon = 0.02 * cv2.arcLength(contorno, True)
  aproximacao = cv2.approxPolyDP(contorno, epsilon, True)
  if cv2.isContourConvex(aproximacao) and len(aproximacao) == 4:
    localizacao = aproximacao
    break
print(localizacao)

imgCopy = img.copy()
cv2.drawContours(imgCopy, [localizacao], 0, (0,255,0), 3)
cv2.namedWindow('Placa do Carro - Contorno na Placa', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - Contorno na Placa', largura, altura)
cv2.imshow("Placa do Carro - Contorno na Placa", imgCopy)

x, y, w, h = cv2.boundingRect(localizacao)
print(f'x={x}, y={y}, w={w}, h={h}')

placaBRG = cv2.merge([img[y:y+h, x:x+w, 0], img[y:y+h, x:x+w, 1], img[y:y+h, x:x+w, 2]])
placa = imagem[y:y+h, x:x+w]
cv2.imshow("Placa do Carro - Recortada", placa)
cv2.imshow("Placa do Carro BGR - Recortada", placaBRG)

valor, lim_otsu = cv2.threshold(placa, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(f'Limiar: {valor}')
cv2.imshow("Placa do Carro - Limiar de Otsu - Recortada", lim_otsu)

config_tesseract = '--tessdata-dir tessdata --psm 6'
texto = pytesseract.image_to_string(lim_otsu, lang = 'por', config = config_tesseract)
print('Texto extraído da Placa')
print(texto)

texto_extraido = re.search('\w{3}\d{1}\w{1}\d{2}', texto)
print('Texto filtrado: Código da placa do carro')
print(texto_extraido.group(0))

x1 = 40
y2 = 30

fonte_dir = 'content/calibri.ttf'

imgPlacaTexto = escreve_texto(texto_extraido.group(0), x1, y2, placaBRG.copy(), fonte_dir, (0,255,255), 25)
imgCarroTexto = escreve_texto(texto_extraido.group(0), x, y, imgCopy, fonte_dir, (0,255,255), 50)

cv2.imshow("Placa do Carro Recortada - Resultado", imgPlacaTexto)

cv2.namedWindow('Placa do Carro - Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - Resultado', largura, altura)
cv2.imshow("Placa do Carro - Resultado",imgCarroTexto)


cv2.waitKey(0)
