import cv2
import pytesseract
import seaborn as sns
import matplotlib.pyplot as plt

print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img = cv2.imread('imagens/placa_carro1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("Placa do Carro - RGB", img)

imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Placa do Carro - Gray", imagem)

ax = sns.histplot(imagem.flatten())
ax.figure.set_size_inches(10,6)
plt.show()

valor, lim_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Placa do Carro - Limiar de Otsu", lim_otsu)
print(f'Limiar: {valor}')

# config_tesseract = '--tessdata-dir tessdata'
# config_tesseract = '--tessdata-dir tessdata --psm 5'
config_tesseract = '--tessdata-dir tessdata --psm 6'
# config_tesseract = '--tessdata-dir tessdata --psm 7'
# config_tesseract = '--tessdata-dir tessdata --psm 8'
texto = pytesseract.image_to_string(lim_otsu, lang = 'por', config = config_tesseract)

print('-- Texto Detectado [Otsu] -- ')
print(texto)

cv2.waitKey(0)
