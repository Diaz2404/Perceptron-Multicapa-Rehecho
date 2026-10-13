import numpy as np


from Subtareas.entrenamiento import entrenamiento
from Subtareas.Funciones_Activaciones import activacion_logistica, activacion_logistica_derivada






entradas =np.array([
    # 0
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,1,1,
     1,0,1,0,1,
     1,1,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 1
    [0,0,1,0,0,
     0,1,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,1,1,1,0],

    # 2
    [0,1,1,1,0,
     1,0,0,0,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,0,1,0,0,
     0,1,0,0,0,
     1,1,1,1,1],

    # 3
    [0,1,1,1,0,
     1,0,0,0,1,
     0,0,0,0,1,
     0,0,1,1,0,
     0,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 4
    [0,0,0,1,0,
     0,0,1,1,0,
     0,1,0,1,0,
     1,0,0,1,0,
     1,1,1,1,1,
     0,0,0,1,0,
     0,0,0,1,0],

    # 5
    [1,1,1,1,1,
     1,0,0,0,0,
     1,1,1,1,0,
     0,0,0,0,1,
     0,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 6
    [0,0,1,1,0,
     0,1,0,0,0,
     1,0,0,0,0,
     1,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 7
    [1,1,1,1,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,0,1,0,0,
     0,1,0,0,0,
     0,1,0,0,0,
     0,1,0,0,0],

    # 8
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 9
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,1,1,0,0]])


y_esperado = np.array([
    [1,0,0,0,0,0,0,0,0,0],  
    [0,1,0,0,0,0,0,0,0,0],  
    [0,0,1,0,0,0,0,0,0,0],  
    [0,0,0,1,0,0,0,0,0,0],  
    [0,0,0,0,1,0,0,0,0,0],  
    [0,0,0,0,0,1,0,0,0,0],  
    [0,0,0,0,0,0,1,0,0,0],  
    [0,0,0,0,0,0,0,1,0,0],  
    [0,0,0,0,0,0,0,0,1,0],  
    [0,0,0,0,0,0,0,0,0,1]])   


i , w , error = entrenamiento(entradas, y_esperado, Activacion_oculto = activacion_logistica, Activacion_salida= activacion_logistica, derivada_oculto= activacion_logistica_derivada , derivada_salida=activacion_logistica_derivada, neuronas_ocultas= 32, neuronas_salidas= 10, cota = 10000, n= 0.1)

print(f"Iteraciones: {i}, Error: {error}")


print("\n\nPredicciones sin ruido en las entradas:")
w_oculto = w[0]
w_salida = w[1]
neuronas_ocultas = len(w_oculto)
neuronas_salidas = len(w_salida)

# Predecimos los digitos sin ruido
for muestra in range(len(entradas)):
    x_i = entradas[muestra]
    y_i = y_esperado[muestra]
    

    # Propagación hacia adelante
    o_oculto = np.zeros(neuronas_ocultas)
    for i in range(neuronas_ocultas):
        o_oculto[i] = activacion_logistica(w_oculto[i] @ x_i + 1)

    o_salida = np.zeros(neuronas_salidas)
    for j in range(neuronas_salidas):
        o_salida[j] = activacion_logistica(w_salida[j] @ o_oculto + 1)

    # o_salida = np.round(o_salida, 3)

    print(f"Entrada: {muestra}, Salida esperada: {y_i}, Salida de la red: {o_salida}, Predicción: {np.argmax(o_salida)}")


#añadimos ruido a las entradas
print("\n\nAñadiendo ruido a las entradas...\n")

entradas_ruido = entradas.copy()

for muestra_2 in range(len(entradas)):
    x_i = entradas_ruido[muestra_2].copy()

    for i_2 in range(len(x_i)):
        if np.random.rand() < 0.02:
            print(f"Cambiamos bit de la entrada: {muestra_2}")
            x_i[i_2] = 1 - x_i[i_2] # cambiar de 1 a 0 o 0 a 1
        
    entradas_ruido[muestra_2] = x_i


print("\n\nPredicciones con ruido en las entradas:")
for muestra_3 in range(len(entradas_ruido)):
    x_i = entradas_ruido[muestra_3]
    y_i = y_esperado[muestra_3]

    # propagar para capa oculta
    o_oculto = np.zeros(neuronas_ocultas)
    for i in range(neuronas_ocultas):
        o_oculto[i] = activacion_logistica(w_oculto[i] @ x_i + 1)

    # propagar para capa de salida
    o_salida = np.zeros(neuronas_salidas)
    for i in range(neuronas_salidas):
        o_salida[i] = activacion_logistica(w_salida[i] @ o_oculto + 1)

    print(f"Entrada: {muestra_3}, Salida esperada: {y_i}, Salida de la red: {o_salida}, Predicción: {np.argmax(o_salida)}")



