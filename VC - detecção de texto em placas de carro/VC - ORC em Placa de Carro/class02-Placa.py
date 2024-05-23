import cv2
import pytesseract

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('imagens/placa_carro1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - RGB", img)

imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
assert imagem is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - Gray", imagem)

config_tesseract = '--tessdata-dir tessdata'
texto = pytesseract.image_to_string(imagem, lang = 'por', config = config_tesseract)
texto1 = pytesseract.image_to_string(img, lang = 'por', config = config_tesseract)

print('-- Texto Detectado -- ')
print(texto)
print(texto1)

cv2.waitKey(0)
