# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:29:02 2024

@author: Miguel Sánchez


Algoritmo de recomendación de películas basado en evaluaciones de otras películas.
Problema conocido como Netflix Prize

Algoritmo desarrollado: Versión de Collaborative filtering

DataSet empleado: MovieLens (100.000) + Encuesta realizada a 60 personas
    Este programa solo emplea el dataset de MovieLens ya que es suficientemente
    grande como para realizar estadística de error.

Error obtenido RMSE:
    En torno a 0.8 (estadística analizándolo en Excel)
    Menor de 1.25 considerado éxito
    
"""

import pandas as pd
import numpy as np
from Prediccion_v2 import predecir_peliculas
import time

def quitar_rating(Matriz, usuario, peli):
    rating = Matriz[usuario][peli]
    Matriz[usuario][peli] = None
    return Matriz, rating

# Para calcular errores
def rmse(vector_predicho, vector_real):
    # Calcula el error cuadrático entre los dos vectores
    error_cuadratico = np.square(vector_predicho - vector_real)
    # Calcula la media del error cuadrático
    mean_error_cuadratico = np.mean(error_cuadratico)
    # Calcula la raíz cuadrada de la media del error cuadrático para obtener el RMSE
    rmse = np.sqrt(mean_error_cuadratico)
    return rmse


# Registra el tiempo de inicio
inicio = time.time()


# Leer el fichero y generar la matriz numpy:
path = '.\\u.csv' 
df = pd.read_csv(path)
Matriz_panda = df.pivot(index='UserID', columns='ItemId', values='Rating')
Matriz_panda = Matriz_panda.where(pd.notna(Matriz_panda), None)
Matriz = np.array(Matriz_panda)


Filas = len(Matriz)
Columnas = len(Matriz[0])
Matriz_Rellena = np.zeros((Filas, Columnas))
# Eliminamos unos pocos valores aleatorios
"""
OJO: Si se desea predecir un valor muy alto, o la matriz tiene demasiados valoeres
None, SE DEBE CAMBIAR EL MODO DE COGER LOS VALORES ALEATORIOS, ya que tal como está
ahora eventualmente sería muy improbable dar con un número distinto de None
"""
num_pel_predecir = 10
ratings_quitados = np.zeros((num_pel_predecir, 3)) # Almacenamos aquí los ratings de las pelis quitadas (col 1 us, col 2 peli col 3 rating)
for i in range(0, num_pel_predecir):
    ratings_quitados[i][0] = np.random.randint(0, Filas)
    ratings_quitados[i][1] = np.random.randint(0, Columnas)
    while (Matriz[int(ratings_quitados[i][0])][int(ratings_quitados[i][1])] == None):
        ratings_quitados[i][0] = np.random.randint(0, Filas)
        ratings_quitados[i][1] = np.random.randint(0, Columnas)        
    Matriz, ratings_quitados[i][2] = quitar_rating(Matriz, int(ratings_quitados[i][0]), 
                                                   int(ratings_quitados[i][1]))

# Predecimos los valores que hemos eliminado: PIEZA CENTRAL DEL PROBLEMA
ratings_predichos = np.hstack((ratings_quitados[:, :2], np.zeros((num_pel_predecir, 1)))) 
    # Almacenamos aquí los ratings de las pelis predichas (col 1 us, col 2 peli col 3 rating)
    # ratings_predichos tiene las dos primeras cols iguales a num_pel_predecir para ver 
    # qué valores se asignan a cada usuario y cada peli

# Calcula el tiempo de ejecución
fin = time.time()
tiempo_ejecucion = fin - inicio
print("Time: ", tiempo_ejecucion)
for i in range(0, num_pel_predecir):
    ratings_predichos[i][2] = predecir_peliculas(Matriz, 
                                                int(ratings_predichos[i][0]), 
                                                int(ratings_predichos[i][1]),
                                                Filas, Columnas)
    # Calcula el tiempo de ejecución
    fin = time.time()
    tiempo_ejecucion = fin - inicio
    print("Recom: ", i, " Hecha")
    print("Time: ", tiempo_ejecucion)

# Calcula el RMS de la predicción:
error = rmse(ratings_predichos[:, 2], ratings_quitados[:, 2])
print("Error Final: ", error)







    
    
    
    
    
