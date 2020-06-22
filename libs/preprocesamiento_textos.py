#Normalización de textos
import numpy as np
import nltk, re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import spacy
from gensim.models.phrases import Phrases, Phraser
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from gensim.models import Word2Vec

#No se puede hacer %matplotlib inline desde un script
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Cargar modelos Spacy y Stemmer NLTK en Español
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
#nlp = spacy.load('es', disable=['parser', 'ner'])
#nlp = spacy.load('es-core-news-sm')
stemmer = SnowballStemmer('spanish')

#Para Lemmatization se usa Spacy, sobre la cadena de texto
def lemmatized_text(tokens, allowed_postags=['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(tokens)) 
    #doc=nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

#Separamos en tokens
def tokenizar(texto):
    palabras=word_tokenize(texto)
    return palabras

#Para Stemming se usa NLTK
def stemmed_tokens(tokens):
    return [stemmer.stem(i) for i in tokens]

#Removemos signos de puntuación de las palabras
def quitar_signos_puntuacion(tokens_palabras):
    palabras = []
    for palabra in tokens_palabras:
        nueva_palabra = re.sub(r'[^\w\s]', '', palabra)
        if nueva_palabra != '':
            palabras.append(nueva_palabra)
    return palabras
        
#Removemos números de las palabras
def quitar_digitos(tokens_palabras):
    palabras = []
    for palabra in tokens_palabras:
        nueva_palabra = re.sub(r'[0-9]', '', palabra)
        if nueva_palabra != '':
            palabras.append(nueva_palabra)
    return palabras

def quitar_stopwords(tokens_palabras):
    """Remove stop words from list of tokenized words"""
    palabras = []
    for palabra in tokens_palabras:
        if palabra not in stopwords.words('spanish'):
            palabras.append(palabra)
    return palabras
  
def obtener_bigrams(oraciones_tokenizadas):
  phrases = Phrases(oraciones_tokenizadas, min_count=100, progress_per=10000)
  bigram = Phraser(phrases)
  #return bigram[oraciones_tokenizadas]
  return bigram

def guardar_datos_pickle(datos,rutaArchivo):
  with open(rutaArchivo, "wb") as fp:   #Pickling
    pickle.dump(datos, fp)

def cargar_datos_pickle(rutaArchivo):
  with open(rutaArchivo, "rb") as fp:   # Unpickling
    datos = pickle.load(fp)
  return datos

def frecuencias_palabras(textos_tokenizados):
  from collections import defaultdict

  #relatos_fase2[0]
  word_freq = defaultdict(int)
  for sent in textos_tokenizados:
    for i in sent:
      word_freq[i] += 1
  return word_freq

#Armar modelo de lenguaje: recibe un array de textos tokenizados
#y devuelve un modelo Word2Vec entrenado sobre los textos
def modelo_word2vec(dataset):
  cores = multiprocessing.cpu_count() # Count the number of cores in a computer
  w2v_model = Word2Vec(min_count=20, window=5, size=300,
    sample=6e-5, alpha=0.03, min_alpha=0.0007,
    negative=20, workers=cores-1)
  print('Construyendo vocabulario...')
  w2v_model.build_vocab(dataset, progress_per=1000)
  print('Entrenando modelo...')
  w2v_model.train(dataset, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
  print('Finalizando...')
  w2v_model.init_sims(replace=True)
  return w2v_model

#Dibuja palabras de un modelo w2v de acuerdo a 
#distancia en un plano bidimensional. word es
#la palabra central y list_names otras palabras
def grafico_modelo_w2v(model, word, list_names):
  sns.set_style("darkgrid")
  arrays = np.empty((0, 300), dtype='f')
  word_labels = [word]
  color_list  = ['red']
  # adds the vector of the query word
  arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
  # gets list of most similar words
  close_words = model.wv.most_similar([word])
  # adds the vector for each of the closest words to the array
  for wrd_score in close_words:
      wrd_vector = model.wv.__getitem__([wrd_score[0]])
      word_labels.append(wrd_score[0])
      color_list.append('blue')
      arrays = np.append(arrays, wrd_vector, axis=0)
  # adds the vector for each of the words from list_names to the array
  for wrd in list_names:
      wrd_vector = model.wv.__getitem__([wrd])
      word_labels.append(wrd)
      color_list.append('green')
      arrays = np.append(arrays, wrd_vector, axis=0)
  # Reduces the dimensionality from 300 to 50 dimensions with PCA
  #reduc = PCA(n_components=50).fit_transform(arrays)
  reduc = PCA(n_components=14).fit_transform(arrays)
  # Finds t-SNE coordinates for 2 dimensions
  np.set_printoptions(suppress=True)
  Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
  # Sets everything up to plot
  df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                     'y': [y for y in Y[:, 1]],
                     'words': word_labels,
                     'color': color_list})
  fig, _ = plt.subplots()
  fig.set_size_inches(9, 9)
  # Basic plot
  p1 = sns.regplot(data=df,
                   x="x",
                   y="y",
                   fit_reg=False,
                   marker="o",
                   scatter_kws={'s': 40,
                                'facecolors': df['color']
                               }
                  )
  # Adds annotations one by one with a loop
  for line in range(0, df.shape[0]):
       p1.text(df["x"][line],
               df['y'][line],
               '  ' + df["words"][line].title(),
               horizontalalignment='left',
               verticalalignment='bottom', size='medium',
               color=df['color'][line],
               weight='normal'
              ).set_size(15)
  plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
  plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
  plt.title('t-SNE visualization for {}'.format(word.title()))

def mediana_de_cant_palabras(relatos_tokenizados):
  num_words = [len(s) for s in relatos_tokenizados]
  return np.median(num_words)

def distribucion_cant_palabras(relatos_tokenizados):
  plt.hist([len(s) for s in relatos_tokenizados], 50)
  plt.xlabel('Cantidad de palabras')
  plt.ylabel('Cantidad de relatos')
  plt.title('Distribución de cantidad de palabras por relato')
  plt.show()