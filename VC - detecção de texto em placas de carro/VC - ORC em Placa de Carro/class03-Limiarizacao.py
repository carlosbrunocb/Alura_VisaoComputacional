import cv2
import pytesseract

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('imagens/placa_carro1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - RGB", img)

imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Placa do Carro - Gray", imagem)

limiar = 25
valor, lim_simples = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
cv2.imshow("Placa do Carro - Limiar Simples = 25", lim_simples)

config_tesseract = '--tessdata-dir tessdata --psm 6'
texto = pytesseract.image_to_string(lim_simples, lang = 'por', config = config_tesseract)

print('-- Texto Detectado [Limiar = 25] -- ')
print(texto)

limiar = 127
valor, lim_simples = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
cv2.imshow("Placa do Carro - Limiar Simples = 127", lim_simples)

texto = pytesseract.image_to_string(lim_simples, lang = 'por', config = config_tesseract)

print('-- Texto Detectado [Limiar = 125]-- ')
print(texto)

cv2.waitKey(0)
