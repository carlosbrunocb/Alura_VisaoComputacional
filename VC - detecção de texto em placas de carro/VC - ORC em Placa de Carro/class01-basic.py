import cv2
import pytesseract

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

imagem = cv2.imread('imagens/trecho_livro.png')

cv2.imshow("Text", imagem)

texto = pytesseract.image_to_string(imagem)
print('--- Configuração padrão do tesseract ---')
print(texto)
print('------')

config_tesseract = '--tessdata-dir tessdata --psm 6'
texto = pytesseract.image_to_string(imagem, lang = 'por', config = config_tesseract)
print('\n--- Tesseract configurado com PSM 6 e língua portuguesa---')
print(texto)
print('------')

cv2.waitKey(0)
