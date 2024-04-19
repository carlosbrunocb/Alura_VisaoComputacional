# PROJETO FINAL
#
# Objetivo:
# + Detectar e reconhecer textos em imagens por meio do tesseract
# + Buscar termos específicos
# + Salvar resultados em um arquivo txt
# + Mostrar os resultados sobre as imagens dos termos específicos

from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output

import pytesseract
import numpy as np
import cv2
import re
import os
import matplotlib.pyplot as plt


print(pytesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
config_tesseract = "--tessdata-dir tessdata"

projeto = "text-recognize/Imagens/Projeto"
caminho = [os.path.join(projeto, f) for f in os.listdir(projeto)]
print(caminho)

fonte_dir = '.venv/text-recognize/Imagens/calibri.ttf'  # local da fonte calibri

# '''
# Função caixa_texto: desenhar um caixa em volta de um texto na imagem
#  @returna
#     x, y :: posição inicial do texto detectado
#     img :: imagem do bounding box
# '''
def caixa_texto(i, resultado, img, cor=(255, 100, 0)):
  x = resultado["left"][i]
  y = resultado["top"][i]
  w = resultado["width"][i]
  h = resultado["height"][i]

  cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)

  return x, y, img

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

# '''
# Função mostrar: exibir as imagens na tela
# '''
def mostrar(img):
  fig = plt.gcf() # busca a figura atual
  fig.set_size_inches(20, 10) #define o tamanho
  plt.axis("off") #remove a visualização dos eixos
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #faz a conversão de cores com o OpenCV
  plt.show() # mostra a imagem

# '''
# Função OCR_processa: processa o reconhecimento de texto em imagen usando o tesseract
#  @returna
#     texto :: string com textos retirados da imagem
# '''
def OCR_processa(img, config_tesseract):
  texto = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
  return texto

# '''
# Função OCR_processa_imagem: processa o reconhecimento de texto em imagen usando o tesseract e
#  realizar uma pesquisa usando um termo sobre os caracteres reconhecidos na image.
#  @returna
#     img :: imagem com bouding box sobre os termos encontrados na imagem
#            e com palavra escrita acima do bounding box
#     num_ocorrencias :: a quantidades de termos encontradas na imagem
# '''
def OCR_processa_imagem(img, termo_pesquisa, config_tesseract, min_conf):
  resultado = pytesseract.image_to_data(img, config=config_tesseract, lang='por', output_type=Output.DICT) #imagem para dados, que já fizemos anteriormente
  num_ocorrencias = 0 #inicializando como 0

  for i in range(0, len(resultado['text'])): # vai de 0 ao tamanho do número de valores do texto
    confianca = float(resultado['conf'][i]) # qual a confiança da detecção
    if confianca > min_conf: # se a confiança for maior que o valor mínimo, passa para a linha abaixo
      texto = resultado['text'][i] #texto será igual ao resultado text no momento i
      # texto = texto.lower() # transforma todo o texto em lowercase
      # if termo_pesquisa in texto: #se o termo de pesquisa estiver no texto:
      if termo_pesquisa.lower() in texto.lower(): #se o termo de pesquisa estiver no texto:
        x, y, img = caixa_texto(i, resultado, img, (0,0,255)) # faz a caixa de bounding box
        img = escreve_texto(texto, x, y, img, fonte_dir, (50,50,225), 14) #escreve o texto

        num_ocorrencias += 1 #faz a iteração no num de ocorrências e volta para o laço até acabar o texto
  return img, num_ocorrencias

os.makedirs('result_project', exist_ok=True)

texto_completo = ''
nome_txt = 'resultados_ocr.txt'

min_conf = 30
termo_pesquisa = 'learning'

for imagem in caminho:
  img = cv2.imread( imagem)
  img_original = img.copy()
  # nome_imagem :: recebe os nomes e diretórios das imagens, quebrados,
  #                precisamos apenas do -1 (última posição do diretório)
  nome_imagem = os.path.split(imagem)[-1]
  print('====================\n' + str(nome_imagem))  # separação + nome da imagem
  nome_divisao = '===================\n' + str(nome_imagem)  # divisão + nome da imagem que está sendo vista
  # texto_completo :: recebe o texto completo + a divisão + /n para pular a linha
  texto_completo = texto_completo + nome_divisao + '\n'
  texto_file = OCR_processa(img, config_tesseract)  # passa a imagem que vamos utilizar, no caso em cada imagem
  texto_completo = texto_completo + texto_file  # concatena as duas variáveis
  ocorrencias = [i.start() for i in re.finditer(termo_pesquisa.lower(), texto_file.lower())]  # usando o finditer novamente no texto

  print('Número de ocorrências para o termo: {}: {}'.format(termo_pesquisa, len(ocorrencias)))
  # primeira chaves é para termo de pesquisa e a segunda é para ocorrencias
  print('\n')

  img, numero_ocorrencias = OCR_processa_imagem(img, termo_pesquisa, config_tesseract, min_conf)

  if numero_ocorrencias > 0:
    mostrar(img)
    novo_nome_imagem = 'OCR_' + nome_imagem
    nova_imagem = 'result_project/' + str(novo_nome_imagem)
    cv2.imwrite(nova_imagem, img)

arquivo_txt = open('result_project/' + nome_txt, 'w+') # a+ é para colocar no final do arquivo, w+ para sobre escrever o arquivo
arquivo_txt.write(texto_completo + '\n') #passa o texto que quer adicionar
arquivo_txt.close()