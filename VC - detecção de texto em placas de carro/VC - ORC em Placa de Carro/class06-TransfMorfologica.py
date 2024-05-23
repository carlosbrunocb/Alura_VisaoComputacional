import cv2
import pytesseract

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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
erosao = cv2.erode(lim_otsu, kernel)
dilatacao = cv2.dilate(lim_otsu, kernel)
abertura = cv2.morphologyEx(lim_otsu, cv2.MORPH_OPEN, kernel)
fechamento = cv2.morphologyEx(lim_otsu, cv2.MORPH_CLOSE, kernel)
gradiente = cv2.morphologyEx(lim_otsu, cv2.MORPH_GRADIENT, kernel)

cv2.imshow("Placa do Carro - Erosao", erosao)
cv2.imshow("Placa do Carro - Dilatacao", dilatacao)
cv2.imshow("Placa do Carro - abertura", abertura)
cv2.imshow("Placa do Carro - fechamento", fechamento)
cv2.imshow("Placa do Carro - gradiente", gradiente)

cartola = cv2.morphologyEx(lim_otsu, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Placa do Carro - cartola", cartola)

# #------
# imgInverso = cv2.bitwise_not(cartola)
# cv2.imshow("Placa do Carro - Inverso Cartola", imgInverso)
# imgSum = cv2.bitwise_and(lim_otsu, imgInverso)
# cv2.imshow("Placa do Carro - Otsu or Cartola", imgSum)
# #------

kernel_retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (40,13))
cartola = cv2.morphologyEx(lim_otsu, cv2.MORPH_TOPHAT, kernel_retangular)
cv2.imshow("Placa do Carro - cartola (Kernel [40,13] + Otsu)", cartola)

cartolaImg = cv2.morphologyEx(imagem, cv2.MORPH_TOPHAT, kernel_retangular)
cv2.imshow("Placa do Carro - cartola Img (Kernel [40,13])", cartolaImg)

chapeu_preto = cv2.morphologyEx(lim_otsu, cv2.MORPH_BLACKHAT, kernel_retangular)
cv2.imshow("Placa do Carro - chapeu_preto (Kernel [40,13])", chapeu_preto)

chapeu_pretoImg = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, kernel_retangular)
cv2.imshow("Placa do Carro - chapeu_preto Img (Kernel [40,13])", chapeu_pretoImg)

# config_tesseract = '--tessdata-dir tessdata'
# config_tesseract = '--tessdata-dir tessdata --psm 4'
# config_tesseract = '--tessdata-dir tessdata --psm  5'
config_tesseract = '--tessdata-dir tessdata --psm 6'
# config_tesseract = '--tessdata-dir tessdata --psm 7'
# config_tesseract = '--tessdata-dir tessdata --psm 8'
texto = pytesseract.image_to_string(lim_otsu, lang = 'por', config = config_tesseract)
textoErosao = pytesseract.image_to_string(erosao, lang = 'por', config = config_tesseract)
textoDilatacao = pytesseract.image_to_string(dilatacao, lang = 'por', config = config_tesseract)
textoAbertura = pytesseract.image_to_string(abertura, lang = 'por', config = config_tesseract)
textoFechamento = pytesseract.image_to_string(fechamento, lang = 'por', config = config_tesseract)
textoGradiente = pytesseract.image_to_string(gradiente, lang = 'por', config = config_tesseract)
textoCartola = pytesseract.image_to_string(cartola, lang = 'por', config = config_tesseract)
textoChapeu_preto = pytesseract.image_to_string(chapeu_preto, lang = 'por', config = config_tesseract)

# #-----
# textoImgSum = pytesseract.image_to_string(imgSum, lang = 'por', config = config_tesseract)
# print('\n-- Texto Detectado [imgSum] -- ')
# print(textoImgSum)
# #-----

print('-- Texto Detectado [Otsu] -- ')
print(texto)

print('\n-- Texto Detectado [Erosão] -- ')
print(textoErosao)

print('\n-- Texto Detectado [Dilatação] -- ')
print(textoDilatacao)

print('\n-- Texto Detectado [Fechamento] -- ')
print(textoFechamento)

print('\n-- Texto Detectado [Abertura] -- ')
print(textoAbertura)

print('\n-- Texto Detectado [Gradiente] -- ')
print(textoGradiente)

print('\n-- Texto Detectado [Cartola] -- ')
print(textoCartola)

print('\n-- Texto Detectado [ChapeuPreto] -- ')
print(textoChapeu_preto)

cv2.waitKey(0)
