import cv2
import pytesseract

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('imagens/placa_carro1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - RGB", img)

imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Placa do Carro - Gray", imagem)

# adaptiveThreshold(imagem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
# @ Parâmetros de entradas:
#   + imagem - imagem que queremos limiarizar;
#   + 255 - valor para o qual transformaremos o pixel caso este seja maior que o limiar;
#   + cv2.ADAPTIVE_THRESH_MEAN_C - tipo de limiarização adaptativa que queremos fazer, neste caso, a média;
#   + cv2.THRESH_BINARY - parâmetro responsável por transformar os pixels para preto ou branco;
#   + 11 - distância, em pixels, da região de vizinhança em torno do pixel de referência;
#   + 8 - valor da constante C, do qual será subtraída a média.
lim_adapt = cv2.adaptiveThreshold(imagem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
cv2.imshow("Placa do Carro - Limiar Adaptativo usando a media", lim_adapt)

# config_tesseract = '--tessdata-dir tessdata'
# config_tesseract = '--tessdata-dir tessdata --psm 5'
config_tesseract = '--tessdata-dir tessdata --psm 6'
# config_tesseract = '--tessdata-dir tessdata --psm 7'
# config_tesseract = '--tessdata-dir tessdata --psm 8'
texto = pytesseract.image_to_string(lim_adapt, lang = 'por', config = config_tesseract)

print('-- Texto Detectado [MEAN] -- ')
print(texto)

lim_adapt = cv2.adaptiveThreshold(imagem, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
cv2.imshow("Placa do Carro - Limiar Adaptativo Gaussiana", lim_adapt)

texto = pytesseract.image_to_string(lim_adapt, lang = 'por', config = config_tesseract)

print('-- Texto Detectado [GAUSSIAN]-- ')
print(texto)

cv2.waitKey(0)
