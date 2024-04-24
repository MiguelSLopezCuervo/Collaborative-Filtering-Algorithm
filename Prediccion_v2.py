# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:09:42 2024

@author: Miguel Sánchez
"""


import numpy as np
from scipy import spatial # PARA LA MEDIDA COSENO
import copy

"""
Remove_Nones
Input: dos vectores de misma length y con algunos valores None
Output: Copia de mismos vectores sin valores None. Las entradas en las que
    alguno de ellos tiene None se eliminan de ambos
Sirve para calcular distancias adecuadamente sólo con las pelis que ambos
usuarios han evaluado
"""
def Remove_Nones(a, b):
    a_r = copy.deepcopy(a)
    b_r = copy.deepcopy(b)
    final = len(a_r)
    i = 0
    while i < final:
        if a_r[i] == None or b_r[i] == None:
            a_r = np.delete(a_r, i)
            b_r = np.delete(b_r, i)
            final = final - 1
            i = i-1
        i = i+1
    return a_r, b_r

"""
Find_Nones: 
Input: a, vector con nones (o sin)
Output: Nones, Vector que indica las posiciones en las que se encuentran los 
        Nones en a
"""
def Find_Nones(a):
    Nones = []
    for i in range(len(a)):
        if a[i] == None: Nones.append(i)
    return np.array(Nones)

"""
Transf_Media: calcula la media de las notas de cada usuario
Input: M, matriz reseñas
Output: medias, vector con la media de las reseñas de cada usuario
Sirve para conocer el factor de transformación que aplicar a cada usuario
"""
def Transf_Media(Matriz, rows, columns):
    N = np.zeros(rows)
    for j in range(rows):
        suma = 0
        cols = 0
        for i in range(0, columns):
            if Matriz[j][i] is not None: 
                suma = suma + Matriz[j][i]
                cols = cols+1
        N[j] = suma/cols
    
    return N

"""
Para mostrar las predicciones de cada usuario.
Recibe el vector con todas las pelis y se queda sólo con las predichas
Input:  vector a, fila de la columna M con las predicciones puestas
        vector nulos, vector que indica cuáles fueron los valores predichos
Output: Predicciones, vector que indica únicamente las predicciones
"""
def predicciones(vector_a, vector_nulos):
    predicciones = [vector_a[i] for i in vector_nulos if i < len(vector_a)]
    return predicciones

"""
Ordena los vectores a y b según el orden del a de mayor a menor
Sirve para mostrar las películas ordenadas de mayor recomendación a menor
"""
def reorganizar_vectores(a, b):
    # Obtener la permutación que ordena el vector a de mayor a menor
    permutacion = sorted(range(len(a)), key=lambda i: a[i], reverse=True)
    
    # Aplicar la misma permutación a ambos vectores
    a_ordenado = [a[i] for i in permutacion]
    b_ordenado = [b[i] for i in permutacion]
    
    return a_ordenado, b_ordenado

"""
Algoritmo de predicción de gusto
Input:  Matriz de ratings
        Usuario y peli cuyo rating predecir
        rows y columns de la matriz
Ouput:  Predicción del rating (indica el valor de la predicción del rating
                               de la película y el usuario pasados)
"""
def predecir_peliculas(Matriz, usuario, peli, rows, columns):
    
    # CALCULO LA DISTANCIA DE CADA USUARIO AL USUSARIO
    distance = np.zeros(rows)
    for i in range(0, rows):
        if i == usuario:
            continue # ESTO ES PORQUE NO HAY QUE CALCULAR LA DISTANCIA A SÍ MISMO
        # TENGO QUE QUITAR LOS NULOS
        U_us, U_i = Remove_Nones(Matriz[usuario], Matriz[i])
        distance[i] = spatial.distance.cosine(U_us, U_i) + .5 # Está entre .01 y 2.01 (así no se dividirá por 0 nunca)
        """
        OJO: el parámetro de 0.1 es importante, ya que a menor sea este más peso se le da
        a los usuarios cercanos
        """
    """
    EN ESTA VERSIÓN, LA MEDIA PONDERADA LA HAGO CON LOS 100 MÁS PARECIDOS,
    ASÍ EVITO TENER EN CUENTA USUARIOS QUE NO SE PARECEN
    """
    dis_ord = sorted(distance, reverse=True)
    n_mayor = dis_ord[400]
    # RATIO DE TRANSFORMACIÓN DE LA NOTA MEDIA DE CADA USUSARIO
    medias = Transf_Media(Matriz, rows, columns)

    """
    PIEZA CENTRAL DEL ALGORITMO:
        Tomo la nota de cada usuario y la transformo por regla de tres a lo que
        evaluaría el usuario 0 M[i][nones[j]] * medias[0] / medias[i]
        Pondero por el inverso de la distancia al usuario 0 1/(distance[i])
        Normalizo res * sum_pesos**-1
        
        El bucle del i deberá ser cambiado en las próximas versiones cuando
        no sólo se le haga la predicción al primer usuario
        Se deberá hacer que este bucle sea una función y que se le pase el usuario
        al que calcularle la predicción, y entonces el bucle deberá ir de 0 a rows
        saltándose el usuario en cuestión (la distancia es 0 por lo que se dividiría
        por infinito)
    """
    prediccion = 0
    sum_pesos = 0
    for i in range(0, rows):
        if i == usuario:
            continue # ESTO ES PORQUE NO HAY QUE CALCULAR LA DISTANCIA A SÍ MISMO
        if Matriz[i][peli] is not None and distance[i] <= n_mayor:
            prediccion = prediccion + 1/(distance[i]) * Matriz[i][peli] * medias[usuario] / medias[i]
            sum_pesos = sum_pesos + 1/distance[i]
    
    prediccion = prediccion * sum_pesos**-1    
    
    return prediccion























