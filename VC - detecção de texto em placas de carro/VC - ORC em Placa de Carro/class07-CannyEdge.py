import cv2
import pytesseract
import re

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('imagens/placa_carro1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - RGB", img)

imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Placa do Carro - Gray", imagem)

valor, lim_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Placa do Carro - Limiar de Otsu", lim_otsu)
print(f'Limiar: {valor}')

bordas = cv2.Canny(imagem, 100, 200)
cv2.imshow("Placa do Carro - Bordas de Canny", bordas)

contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contornos)

for contorno in contornos:
  epsilon = 0.02 * cv2.arcLength(contorno, True)
  aproximacao = cv2.approxPolyDP(contorno, epsilon, True)
  if cv2.isContourConvex(aproximacao) and len(aproximacao) == 4:
    localizacao = aproximacao
    break
print(localizacao)

imgCopy = img.copy()
cv2.drawContours(imgCopy, [localizacao], 0, (0,255,0), 3)
# cv2.polylines(imgCopy, [localizacao], True, (0,0,255), 3)
cv2.imshow("Placa do Carro - Contorno na Placa", imgCopy)

x, y, w, h = cv2.boundingRect(localizacao)
print(f'x={x}, y={y}, w={w}, h={h}')

placa = imagem[y:y+h, x:x+w]
cv2.imshow("Placa do Carro - Recortada", placa)

valor, lim_otsu = cv2.threshold(placa, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
erosao = cv2.erode(lim_otsu, cv2.getStructuringElement(cv2.MORPH_RECT, (4,4)))
cv2.imshow("Placa do Carro - Limiar de Otsu - Recortada", lim_otsu)
cv2.imshow("Placa do Carro - Limiar de Otsu + Erosão- Recortada", erosao)

config_tesseract = '--tessdata-dir tessdata --psm 6'
texto = pytesseract.image_to_string(erosao, lang = 'por', config = config_tesseract)
print('Texto extraído da Placa')
print(texto)

texto_extraido = re.search('\w{3}\d{1}\w{1}\d{2}', texto)
print('Texto filtrado: Código da placa do carro')
print(texto_extraido.group(0))

cv2.waitKey(0)
