import numpy as np

def entrenamiento(entradas, y_esperado, Activacion_oculto, Activacion_salida, derivada_oculto, derivada_salida, neuronas_ocultas, neuronas_salidas, cota=10000, n=0.1):
    # Creamos los pesos que conectan la capa de entrada con la capa oculta, fila= cant neuronas ocultas, columna= entrada
    w_oculto = np.zeros((neuronas_ocultas, entradas.shape[1]))
    # Creamos los pesos que conectan la capa oculto con la capa de salida, fila= cant de salidas, columna= cant neurdonas ocultas
    w_salida = np.zeros((neuronas_salidas, neuronas_ocultas))

    # entradas matriz de 10x35
    # w_oculto matriz de 4x35
    # w_salida matriz de 10x4
  

    error = 1
    error_min = 80
    w_min = [w_oculto.copy(), w_salida.copy()]
    i = 0
    p = len(entradas)

    while i < cota:
        i_x = np.random.randint(0, p)
        x_i = entradas[i_x]   # un vector de 35 elementos
        y_i = y_esperado[i_x] # un vector de 10 elementos

        # Progagación hacia adelante
        # Capa oculta
        print(f"w_oculto: {w_oculto.shape}, w_salida: {w_salida.shape}, entradas: {entradas.shape}")
        print(f"w_oculto:\n{w_oculto}, \nw_salida:\n{w_salida}, \nentradas:\n{entradas}")
        h_oculto = np.zeros((neuronas_ocultas))
        o_oculto = np.zeros((neuronas_ocultas))
        for i in range(neuronas_ocultas):
            print(f"Suma ponderada de la capa oculta: w_oculto[{i}] @ x_i + 1 = {w_oculto[i]} @ {x_i} + 1")
            h_oculto[i] = w_oculto[i] @ x_i + 1
            o_oculto[i] = Activacion_oculto(h_oculto[i])
        print(f"h_oculto: {h_oculto}, o_oculto: {o_oculto}")

        # Capa de salida
        h_salida = np.zeros((neuronas_salidas))
        o_salida = np.zeros((neuronas_salidas))
        for i in range(neuronas_salidas):
            print(f"Suma ponderada de la capa de salida: w_salida[{i}] @ o_oculto + 1 = {w_salida[i]} @ {o_oculto} + 1")
            h_salida[i] = w_salida[i] @ o_oculto + 1
            o_salida[i] = Activacion_salida(h_salida[i])
        print(f"h_salida: {h_salida}, o_salida: {o_salida}")

        # Retropropagación del error con la predición de la salida
        # Capa de salida
        print("\nRetropropagación del error, capa de salida")
        print(f"y_esperado[{i_x}]: {y_esperado[i_x]}, o_salida: {o_salida},h_salida: {h_salida}")

        delta_salida = np.zeros((neuronas_salidas))
        for i in range(neuronas_salidas):
            delta_salida[i] = (y_i[i] - o_salida[i]) * derivada_salida(h_salida[i])
            print(f"delta_salida[{i}]= (y_i[{i}] - o_salida[{i}]) * derivada_salida(h_salida[{i}]) = ({y_i[i]} - {o_salida[i]}) * {derivada_salida(h_salida[i])} = {delta_salida[i]}")
        print(f"delta_salida: {delta_salida}")

        # Capa oculta
        delta_oculto = np.zeros((neuronas_ocultas))
        print("\nRetropropagación del error, capa oculta")
        print("Parametros:")
        print(f"delta_salida: {delta_salida}, h_oculto: {h_oculto}, \nw_salida: \n{w_salida}, \nw_salida[i][j]: {w_salida[0][0]}\n\n")
        for i in range(len(delta_oculto)):
            sumatoria = 0
            for j in range(len(delta_salida)):
                sumatoria += w_salida[j][i] * delta_salida[j]
            delta_oculto[i] = derivada_oculto(h_oculto[i]) * sumatoria
            print(f"delta_oculto[{i}] = derivada_oculto(h_oculto[{i}]) * sumatoria = {derivada_oculto(h_oculto[i])} * {sumatoria} = {delta_oculto[i]}")
        print(f"delta_oculto: {delta_oculto}")


        # Actualización de los pesos
        print("\nActualización de los pesos")
        fila, columna = w_salida.shape
        for i in range(fila):
            for j in range(columna):
                w_salida[i][j] += n * delta_salida[i] * o_oculto[j]
                print(f"w_salida[{i}][{j}] += n * delta_salida[{i}] * o_oculto[{j}] = {w_salida[i][j]} + {n} * {delta_salida[i]} * {o_oculto[j]} = {w_salida[i][j]}")
                
        print(f"\nw_salida actualizada: \n{w_salida}")

        fila, columna = w_oculto.shape
        for i in range(fila):
            for j in range(columna):
                w_oculto[i][j] += n * delta_oculto[i] * x_i[j]

        print(f"\nw_oculto actualizada: \n{w_oculto}")

        # Cálculo del error
        error = 0
        for i in range(neuronas_ocultas):
        
        return
    


