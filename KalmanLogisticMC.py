'''
11/08/22
Hernández Rosario Christian de Jesús
Calculo de Mínimos Cuadradros 
Modificaciones:
17-19 Agosto -> Realizacion de la función predicciones, la cual recibirá como argumento la matriz F
22-23 Agosto -> Modificación de la funcion MinimosCuadrados para que dependiera de Q y R 
23-24 Agosto -> Modificación de la funcion MinimosCuadrados para que dependiera de X, donde X es un arreglo de dimencion 11.
'''
#Sección importar librerías
import numpy as np
import matplotlib.pyplot as plt
from KalmanLogistic import KalmanFilter
from scipy.optimize import minimize
#---------------------------
def MinimosCuadrados(X):
    """
    Función para calcular minimosCuadrados 
    Parameters:
    X : Type: Array 1-D
        Description: Arreglo de dimensión 11, con:
                     X[0:9] <- F, Matriz Modelo de Transición de Estado
                     X[9] <- Elemento de las diagonal de Q, Covarianza del ruido del proceso
                     X[10] <- R, Covarianza del ruido de observación
    Returns:
    jx : Type: float
        Description: Norma Vectorial **2
    """
    F = X[0:9]                       #Equivalente a F = (X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7],X[8])
    F = np.array(F).reshape(3,3)     #Transformación de arreglo -> Matriz 3x3 (Modelo Lineal)
    #print(F)
    Q = np.array([[X[9], 0.0, 0.0], [0.0, X[9], 0.0], [0.0, 0.0,X[9]]]) #Obtenemos matriz diagonal
    #print(Q) 
    R = np.array([X[10]]).reshape(1, 1) #Obtencion de matriz de un solo elemento
    #print(R)
    H = np.array([1, 0, 0]).reshape(1, 3)   #H <- Remodelar en una matriz de 1x3

    x = np.loadtxt("time.txt")              #Tomamos los valores para el tiempo
    measurements = np.loadtxt("data.txt")   #Capturamos los datos a utilizar 

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    predictions = predicciones(kf,F)            #Para graficar
    #SECCIÓN graficar----------------------------------------------------------------
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    print(measurements) #datos sintéticos
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Asimilación de datos')
    print(predictions)  #datos ajustados
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------------------  
    jx = np.linalg.norm(measurements-predicciones(kf,F))**2#+0.5*np.linalg.norm(F)**2
    return jx  

def predicciones(self,F):
    """
    Función para calcular predicciones dependiendo de F
    Parameters:
    F : Type: Array 3-D
        Description: Matriz Modelo de Transición de Estado
    Returns:
    prediction : Type: Array 
        Description: Array de predicciones
    """
    self.F = F                              #Inicializar F
    x = np.loadtxt("time.txt")              #Tomamos los valores para el tiempo
    measurements = np.loadtxt("data.txt")   #Capturamos los datos a utilizar 

    predictions = []                        #Array de predicciones
    #Ciclo para calcular predicciones basándonos en las observaciones obtenidas de la LEY
    for z in measurements:
        predictions.append(np.dot(self.H,  self.predict())[0])
        self.update(z)
    return predictions

#INICIO MAIN---------------------
if __name__ == '__main__':
    #F <- MODELO LINEAL
    F = np.random.uniform(low=0, high=1.0, size=9).tolist() #Elemenos para la matriz F = [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]                                     
    Q = 0.05                                                #Elemento de la diagonal de la matriz Q
    R = 0.0005                                              #Elementos de la matriz R 
    X = (F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[8],Q,R)  #Parameters
    
    result = minimize(MinimosCuadrados,X)
    print(result)

#FIN programa