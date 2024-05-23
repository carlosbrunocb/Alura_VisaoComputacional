from PIL import ImageFont, ImageDraw, Image
from skimage.segmentation import clear_border
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


img = cv2.imread('imagens/placa_carro3.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
altura, largura = img.shape[:2]
imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Placa do Carro - RGB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa do Carro - RGB', largura, altura)
cv2.imshow("Placa do Carro - RGB", img)
cv2.imshow("Placa do Carro - Gray", imagem)

kernel_retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (40,13))
chapeu_preto = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, kernel_retangular)
cv2.imshow("Placa do Carro - Chapeu Preto", chapeu_preto)

sobel_x = cv2.Sobel(chapeu_preto, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
sobel_x = np.absolute(sobel_x)
sobel_x = sobel_x.astype('uint8')
# cv2.namedWindow('Placa do Carro - Bordas de Sobel', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Placa do Carro - Bordas de Sobel', largura, altura)
cv2.imshow("Placa do Carro - Bordas de Sobel - diferencial x", sobel_x)

sobel_y = cv2.Sobel(chapeu_preto, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = 1)
sobel_y = np.absolute(sobel_y)
sobel_y = sobel_y.astype('uint8')
cv2.imshow("Placa do Carro - Bordas de Sobel - diferencal y", sobel_y)

# Filtro de Borramento (Passa-baixa)
sobel_x = cv2.GaussianBlur(sobel_x, (5,5), 0)
sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, kernel_retangular)
cv2.imshow("Placa do Carro - Borramento - diferencial x", sobel_x)

valor, limiarizacao = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Placa do Carro - Limiarizacao", limiarizacao)

kernel_quadrado = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
limiarizacao = cv2.erode(limiarizacao, kernel_quadrado, iterations = 2)
limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations = 2)
cv2.imshow("Placa do Carro - Transformacao Morfologica", limiarizacao)

fechamento = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel_quadrado)
valor, mascara = cv2.threshold(fechamento, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Placa do Carro - Limiarizacao - Fechamento", mascara)

limiarizacao = cv2.bitwise_and(limiarizacao, limiarizacao, mask = mascara)
limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations = 2)
limiarizacao = cv2.erode(limiarizacao, kernel_quadrado)
cv2.imshow("Placa do Carro - And + Transf. Morf. + Limiar", limiarizacao)

# Remoção de ruídos
# Usando a função clear_border() do Scikit Image para remover bloco de pixels que estão tocando a borda

limiarizacao = clear_border(limiarizacao)
cv2.imshow("Placa do Carro - Remocao de ruido das bordas", limiarizacao)

# Detecção de Contornos
contornos, hierarquia = cv2.findContours(limiarizacao, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:10]
# print(contornos)

imgCopy = img.copy()

for contorno in contornos:
  x, y, w, h = cv2.boundingRect(contorno)
  proporcao = float(w)/h
  if proporcao >=3 and proporcao <= 3.5:
    placa = imagem[y:y+h, x:x+w]
    valor, regiao_interesse = cv2.threshold(placa, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    regiao_interesse = clear_border(regiao_interesse)
    cv2.imshow("Placa do Carro", placa)
    cv2.imshow("Placa do Carro - ROI", regiao_interesse)

    # Desenha o contorno da placa
    # cv2.drawContours(imgCopy, [contorno], 0, (0,255,0), 3)
    coordPlaca = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    cv2.drawContours(imgCopy, [coordPlaca], 0, (0,255,0), 3)
    break

cv2.imshow("Placa do Carro - Bounding Box", imgCopy)

config_tesseract = '--tessdata-dir tessdata --psm 6'
texto = pytesseract.image_to_string(regiao_interesse, lang = 'por', config = config_tesseract)
print(texto)
texto_extraido = re.search('\w{3}\d{1}\w{1}\d{2}', texto)

if (texto_extraido != None):
  print(texto_extraido.group(0))
else:
  print("Error")

fonte_dir = '/content/calibri.ttf'

x1 = 40
y2 = 30

placaBRG = cv2.merge([img[y:y+h, x:x+w, 0], img[y:y+h, x:x+w, 1], img[y:y+h, x:x+w, 2]])

imgPlacaTexto = escreve_texto(texto_extraido.group(0), x1, y2, placaBRG.copy(), fonte_dir, (0,0,0), 25)
imgCarroTexto = escreve_texto(texto_extraido.group(0), x, y, imgCopy, fonte_dir, (0,0,0), 50)

cv2.imshow("Placa do Carro - Resultado",imgCarroTexto)
cv2.imshow("Placa do Carro Recortada - Resultado", imgPlacaTexto)

cv2.waitKey(0)
