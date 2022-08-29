'''
11/08/22
Hernández Rosario Christian de Jesús

Asimilación de datos con Filtro de Kalman a datos sintéticos de la ecuación logística
Datos almacenados en archivos de texto time.txt y data.txt
'''
#Sección importar librerías
import numpy as np
import matplotlib.pyplot as plt
#---------------------------

#Clase Filtro kALMAN
class KalmanFilter(object):
    #CONSTRUCTOR
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        '''
            Fk -> el modelo de transición de estado
            Hk -> el modelo de observación
            Qk -> la covarianza del ruido del proceso
            Rk -> la covarianza del ruido de observación
            Bk -> el modelo de control-entrada 
            Uk -> el vector de control (la entrada de control en el modelo de control-entrada)
        '''
    #----------------------------------------------------------------
        #Verificación de la existencia del modelo de transición de estado y del modelo de observación  
        if(F is None or H is None):
            raise ValueError("Establecer la dinámica adecuada del sistema")
        #----------------------------------------------------------------

        #Inicialización de Atributos de la Clase
        self.n = F.shape[1] #n <- Dimensión del array F
        self.m = H.shape[1] #m <- Dimensión del array H

        self.F = F
        self.H = H
        self.B = 0 if B is None else B

        #Creamos una matriz con 1´s en la diagonal y de tamaño indicado dentro del paréntesis (número de filas)
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P

        #Devuelve una matriz rellena de 0´s con dimensiones nx1
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        #FIN CONSTRUCTOR ----------------------------------------------------------------
    #MÉTODO PREDICCIÓN
    def predict(self, u = 0):
        #Producto punto F.x   +  Producto punto B.u
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        #Producto punto (F.P).T  + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    #----------------------------------------------------------------

    #MÉTODO ACTUALIZACIÓN
    def update(self, z):
        #y <- z - H.x 
        y = z - np.dot(self.H, self.x)
        #S <- R + (H.(P.T))
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        #K <- ((P.T).inversa de S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        #x <- x + K.y  
        self.x = self.x + np.dot(K, y)
        #T <- Matriz de nxn con 1´s en la diagonal
        I = np.eye(self.n)
        #P <- ((I-(K.H)).P) . ((I-(K.H)).T + ((K.R).T))
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    #----------------------------------------------------------------
#------------------------------ fin clase

#INICIO MAIN---------------------
if __name__ == '__main__':

    #Inicialización de variables para caso EJEMPLO
    dt = 1.0/60 #incremento de tiempo
    #Matriz 3x3
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    #H <- Remodelar en una matriz de 1x3
    H = np.array([1, 0, 0]).reshape(1, 3)
    #Q <- Matriz 3x3
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    #R <- Remodelar en una matriz de 1x1 (vector de un solo elemento)
    R = np.array([0.5]).reshape(1, 1)

    #Tomamos los valores para el tiempo
    x = np.loadtxt("time.txt")
    #Capturamos los datos a utilizar 
    measurements = np.loadtxt("data.txt")
    #DECLARACIÓN DE Objeto para ajustar datos (predicción)
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    #Array de predicciones a graficar
    predictions = []

    #Ciclo para calcular predicciones basándonos en las observaciones obtenidas de la LEY
    for z in measurements:
        #Llamada a funcion para predecir y agregando al ARRAY
        predictions.append(np.dot(H,  kf.predict())[0])
        #Actualizar dato
        kf.update(z)
        #print(z)

    #SECCIÓN graficar ambos Arrays----------------------------------------------------------------
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    print(measurements) #datos sintéticos
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    print(predictions)  #datos ajustados
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------------------
    
 
#FIN programa