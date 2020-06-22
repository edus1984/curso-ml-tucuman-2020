import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Función que recibe un dataset pandas con un componente de oraciones partidas en tokens (comp 0) y la etiqueta de clase correspondiente (comp 1),
# y devuelve un dataset listo para alimentar a un clasificador:
#Devuelve 3 valores:
#Matriz X con las palabras representadas por números de tokens para cada texto y rellenadas a la izquierda
#Matriz y con las etiquetas codificadas en one-hot
#Objeto tokenizer con los ids de las palabras mapeadas
#Parámetros:
#datos: dataset pandas con dos columnas de datos (textos y etiquetas)
#num_words_vocab: Cantidad de palabras del vocabulario a armar
#maxlen: Máximo largo de la representación de los textos (del vector de cada individuo)
def procesar_textos_tokenizados(datos,num_words_vocab,maxlen):
	print('Uniendo tokens de oraciones')
	oraciones=[" ".join(tokens) for tokens in tqdm(datos[datos.columns[0]])]

	#Armamos vectores de tokens
	tokenizer = Tokenizer(num_words=num_words_vocab)
	tokenizer.fit_on_texts(oraciones)
	X = tokenizer.texts_to_sequences(oraciones)
	vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

	#Rellenado de secuencias (sequence padding)
	X = pad_sequences(X, padding='pre', maxlen=maxlen)

	#Armamos encodings para las categorías
	encoder = LabelEncoder()
	encoder.fit(datos[datos.columns[1]].values)
	encoded_Y = encoder.transform(datos[datos.columns[1]].values)
	y_onehot = np_utils.to_categorical(encoded_Y) #convert integers to dummy variables (i.e. one hot encoded)

	return X,y_onehot,tokenizer

#Función para crear matriz de embeddings desde un modelo de lenguaje
#Parámetros:
#Modelo de lenguaje
#Objeto tokenizer con ids de las palabras
#Dimensión de los embeddings (típicamente 300)
def matriz_embeddings(mod_len,tokenizer,embedding_dim):
	vocab_size=len(tokenizer.word_index)+1
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	for word in mod_len.wv.vocab:
	  if word in tokenizer.word_index:
	    idx = tokenizer.word_index[word]
	    embedding_matrix[idx] = np.array(mod_len.wv[word], dtype=np.float32)[:embedding_dim]
	return embedding_matrix

#Función para mostrar gráficos de performance de un modelo. Recibe como parámetro al objeto history del entrenamiento
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

#Matriz de confusión y Reporte de clasificación con Precisión, cobertura y F1-score
def plot_reporte_clasificacion(modelo,X,y_onehot_gt):
	#Valores predichos para el test set
	y_test_predichos=modelo.predict(X)

	#Creamos la matriz de confusión
	snn_cm = confusion_matrix(np.argmax(y_onehot_gt, axis=1), np.argmax(y_test_predichos, axis=1))

	# Visualizamos la matriz de confusión
	snn_df_cm = pd.DataFrame(snn_cm)
	plt.figure(figsize = (20,14))  
	sn.set(font_scale=1.4) #for label size  
	sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 7}) # font size  
	plt.show()

	snn_report = classification_report(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_predichos, axis=1))
	print(snn_report)
