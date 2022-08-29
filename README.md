# ModeloML.py
Ejemplo de Asimilación de Datos a un modelo logístico con Filtro de Kalman. 

Este proyecto trata de reproducir un modelo de una ecuación logística con ayuda del algoritmo Filtro de Kalman.
Incluye códigos en Python con las siguientes características:

Logistic: 
  Se encuentra la clase Logistic con atributos necesarios para la ecuación logística del crecimiento poblacional. 
  Utilizamos un integrador numérico para resolverla. Capturamos los datos sintéticos obtenidos y el rango de tiempo en archivos de texto.
  
KalmanLogistic: 
  Realiza la Asimilación de Datos aplicando Filtro de Kalman (ajuste de datos). 
  Los datos sintéticos fueron tomados del los archivos de texto generados por el programa Logistic.py. 
  En el main se grafica los datos sintéticos y el ajuste de datos.
  
KalmanLogisticMC: 
  Tenemos la función MinimosCuadrados que depende del modelo lineal inicial propuesto, la covarianza del ruido del proceso y de observación. 
  Además de una función predicciones que depende exclusivamente del modelo lineal F que va cambiando por cada iteración de la primera función. 
  Finalmente, invoco la función minimize de Scipy para encontrar los valores óptimos.
