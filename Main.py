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


i , w , error = entrenamiento(entradas, y_esperado, Activacion_oculto = activacion_logistica, Activacion_salida= activacion_logistica, derivada_oculto= activacion_logistica_derivada , derivada_salida=activacion_logistica_derivada, neuronas_ocultas= 4, neuronas_salidas= 10, cota = 10000, n= 0.1)

# w_oculto = np.zeros((4, entradas.shape[1]))
# print(f"Entradas: {entradas[0]}, y_esperado: {w_oculto[0], {entradas.shape}, {w_oculto.shape}}")
# resultado =entradas[0] @ w_oculto[0] +1
# print(f"Resultado: {resultado}, type: {type(resultado)}")


# # Predecimos los digitos sin ruido
# for i in range(10):
#     y_pred = predecir(entradas[i], w[0], w[1], Activacion_oculto= activacion_logistica, Activacion_salida= activacion_logistica)
#     print(f"Entrada: {entradas[i]}, Salida esperada: {y_esperado[i]}, Salida de la red: {y_pred}")

# # Predecimos los digitos con ruido
# for j in range(10):
#     generar_ruido_entrada(entradas[j], ruido=0.02)
#     y_pred = predecir(entradas[j], w[0], w[1], Activacion_oculto= activacion_logistica, Activacion_salida= activacion_logistica)
#     print(f"Entrada: {entradas[j]}, Salida esperada: {y_esperado[j]}, Salida de la red: {y_pred}")
