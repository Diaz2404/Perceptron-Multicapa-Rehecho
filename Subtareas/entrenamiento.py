import numpy as np

def entrenamiento(entradas, y_esperado, Activacion_oculto, Activacion_salida, derivada_oculto, derivada_salida, neuronas_ocultas, neuronas_salidas, cota=10000, n=0.1):
    # Creamos los pesos que conectan la capa de entrada con la capa oculta, fila= cant neuronas ocultas, columna= entrada
    w_oculto = np.zeros((neuronas_ocultas, entradas.shape[1]))
    # Creamos los pesos que conectan la capa oculto con la capa de salida, fila= cant de salidas, columna= cant neurdonas ocultas
    w_salida = np.zeros((neuronas_salidas, neuronas_ocultas))

    # entradas matriz de 10x35
    # w_oculto matriz de 4x35
    # w_salida matriz de 10x4
  

    
    error_min = 80
    w_min = [w_oculto.copy(), w_salida.copy()]
    i_cota = 0
    p = len(entradas)

    while i_cota < cota:
        i_x = np.random.randint(0, p)
        x_i = entradas[i_x]   # un vector de 35 elementos
        y_i = y_esperado[i_x] # un vector de 10 elementos

        # Progagación hacia adelante
        # Capa oculta
        # print(f"w_oculto: {w_oculto.shape}, w_salida: {w_salida.shape}, entradas: {entradas.shape}")
        # print(f"w_oculto:\n{w_oculto}, \nw_salida:\n{w_salida}, \nentradas:\n{entradas}")
        h_oculto = np.zeros((neuronas_ocultas))
        o_oculto = np.zeros((neuronas_ocultas))
        for i_1 in range(neuronas_ocultas):
            # print(f"Suma ponderada de la capa oculta: w_oculto[{i}] @ x_i + 1 = {w_oculto[i]} @ {x_i} + 1")
            h_oculto[i_1] = w_oculto[i_1] @ x_i + 1
            o_oculto[i_1] = Activacion_oculto(h_oculto[i_1])
        # print(f"h_oculto: {h_oculto}, o_oculto: {o_oculto}")

        # Capa de salida
        h_salida = np.zeros((neuronas_salidas))
        o_salida = np.zeros((neuronas_salidas))
        for i_2 in range(neuronas_salidas):
            # print(f"Suma ponderada de la capa de salida: w_salida[{i}] @ o_oculto + 1 = {w_salida[i]} @ {o_oculto} + 1")
            h_salida[i_2] = w_salida[i_2] @ o_oculto + 1
            o_salida[i_2] = Activacion_salida(h_salida[i_2])
        # print(f"h_salida: {h_salida}, o_salida: {o_salida}")

        # Retropropagación del error con la predición de la salida
        # Capa de salida
        # print("\nRetropropagación del error, capa de salida")
        # print(f"y_esperado[{i_x}]: {y_esperado[i_x]}, o_salida: {o_salida},h_salida: {h_salida}")

        delta_salida = np.zeros((neuronas_salidas))
        for i_3 in range(neuronas_salidas):
            delta_salida[i_3] = (y_i[i_3] - o_salida[i_3]) * derivada_salida(h_salida[i_3])
        #     print(f"delta_salida[{i}]= (y_i[{i}] - o_salida[{i}]) * derivada_salida(h_salida[{i}]) = ({y_i[i]} - {o_salida[i]}) * {derivada_salida(h_salida[i])} = {delta_salida[i]}")
        # print(f"delta_salida: {delta_salida}")

        # Capa oculta
        delta_oculto = np.zeros((neuronas_ocultas))
        # print("\nRetropropagación del error, capa oculta")
        # print("Parametros:")
        # print(f"delta_salida: {delta_salida}, h_oculto: {h_oculto}, \nw_salida: \n{w_salida}, \nw_salida[i][j]: {w_salida[0][0]}\n\n")
        for i_4 in range(len(delta_oculto)):
            sumatoria = 0
            for j_1 in range(len(delta_salida)):
                sumatoria += w_salida[j_1][i_4] * delta_salida[j_1]
                # print(f"sumatoria += w_salida[{j_1}][{i_4}] * delta_salida[{j_1}] = {w_salida[j_1][i_4]} * {delta_salida[j_1]} = {w_salida[j_1][i_4] * delta_salida[j_1]}, sumatoria = {sumatoria}")
            delta_oculto[i_4] = derivada_oculto(h_oculto[i_4]) * sumatoria
        #     print(f"delta_oculto[{i_4}] = derivada_oculto(h_oculto[{i_4}]) * sumatoria = {derivada_oculto(h_oculto[i_4])} * {sumatoria} = {delta_oculto[i_4]}")
        # print(f"delta_oculto: {delta_oculto}")


        
        # Actualización de los pesos
        # print("\nActualización de los pesos")
        # print(f"Parametros pesos: w_salida: \n{w_salida}")
        fila, columna = w_salida.shape
        for i_5 in range(fila):
            for j_2 in range(columna):
                # print(f"w_salida[{i_5}][{j_2}] += n * delta_salida[{i_5}] * o_oculto[{j_2}] = {w_salida[i_5][j_2]} + {n} * {delta_salida[i_5]} * {o_oculto[j_2]} = { w_salida[i_5][j_2] + n * delta_salida[i_5] * o_oculto[j_2]}")
                w_salida[i_5][j_2] += n * delta_salida[i_5] * o_oculto[j_2]
                
        # print(f"\nw_salida actualizada: \n{w_salida}")
    

        fila_oculto, columna_oculto = w_oculto.shape
        for i_6 in range(fila_oculto):
            for j_3 in range(columna_oculto):
                w_oculto[i_6][j_3] += n * delta_oculto[i_6] * x_i[j_3]

        # print(f"\nw_oculto actualizada: \n{w_oculto}")

        # print("\nCálculo del error total")
        # Cálculo del error
        print("\nCálculo del error total")
        error = 0
        for muestra in range(p):
            x_i_error = entradas[muestra]
            y_i_error = y_esperado[muestra]
            #Propagacion hacia adelante
            o_oculto_nuevo = np.zeros((neuronas_ocultas))
            for i_7 in range(neuronas_ocultas):
                o_oculto_nuevo[i_7] = Activacion_oculto(w_oculto[i_7] @ x_i_error + 1)
            # Capa de salida
            o_salida_nuevo = np.zeros((neuronas_salidas))
            for i_8 in range(neuronas_salidas):
                o_salida_nuevo[i_8] = Activacion_salida(w_salida[i_8] @ o_oculto_nuevo + 1)

            # diferencia de la prediccion con el esperado, se resta elemento por elemento 
            print(f"Parametros iterativos: {muestra}/{p}, y_i: {y_esperado[muestra]}, o_salida_nuevo: {o_salida_nuevo}, error: {error}")
            for i_9 in range(10):
                error += (y_i_error[i_9] - o_salida_nuevo[i_9]) ** 2
            error = error / 2
            print(f"Error total sumado: {error}")

        print(f"{i_cota} / {cota}, error total: {error}, error minimo: {error_min},")

        if error < error_min:
            error_min = error
            w_min = [w_oculto.copy(), w_salida.copy()]
            # print(f"Nuevo error mínimo: {error_min}, pesos guardados.")
        
        i_cota += 1

    return i_cota, w_min, error_min
    


