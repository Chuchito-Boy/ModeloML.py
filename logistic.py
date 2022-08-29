#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:09:06 2022

@author: Marcos Capistrán
Modificado por: Christian Hernández, 10/08/22
"""
#Sección importar librerías
import numpy as np
import scipy.stats as ss
from scipy import integrate
import matplotlib.pyplot as plt
#-------------------------------
#Clase logistic 
class logistic():
    
    def __init__(self, b1=2, b2=5, d1=1, d2=5, n0=1, ti=0, tf=30):
        """
        Inicializa la clase logistic. Define la tasa de crecimiento,
        la capacidad de carga, la condicion inicial y el tiempo de
        integracion
        
            Parameters
            ----------
            b1 : TYPE. float
                DESCRIPTION. Parte constante de la tasa de nacimiento
            b2 : TYPE. Float
                DESCRIPTION. Parte lineal de la tasa de nacimiento
            d1 : TYPE. Float
                DESCRIPTION. Parte constante de la tasa de muete
            d2 : TYPE. Float
                DESCRIPTION. Parte lineal de la tasa de muerte
                
            Returns
            -------
            None.
                
        """
        self.r = b1-d1             # Calcula la tasa de crecimiento para b1 diferente de d1
        self.k = (b2+d2)/self.r    # Calcula la capacidad de carga
        self.n0 = (n0,)            # Estado actual de la tupla
        self.teval = np.linspace(0.0,20.0,21)  #Tomar 21 muestras desde 0-20     
        
    def f(self, t, x):
        """
        Lado rerecho de la ecuacion logistica
        
        Parameters
        ----------
        t : TYPE. Float
            DESCRIPTION. Tiempo
        x : TYPE. Float
            DESCRIPTION. Poblacion

        Returns
        -------
        fx : TYPE. Float
            DESCRIPTION. Lado derecho de la ecuacion diferencial

        """
        fx = self.r*x*(1-x/self.k)      #Ecuacion logistica a resolver, donde N=x 
        return fx
    """
    Método para resolver el lado rerecho de la ecuacion logistica
    ----------
    Parameters:
    None
    ----------
    Return:
    ----------
    Array de datos
    """    
    def solve(self):
        ti = self.teval[0]
        tf = self.teval[-1]+0.5
        #INTEGRADOR NUMERICO
        return integrate.solve_ivp(self.f, (ti,tf), self.n0, method='RK45', 
                                   t_eval = self.teval)     
#Main 
if __name__ == '__main__':
    run_logistic = logistic()                                           # Inicializa la clase     
    soln = run_logistic.solve()                                         # Resuelve el problema de valores iniciales
    time = run_logistic.teval                                           # Obtenemos arreglo con los valores para el tiempo
    data = soln.y + ss.norm.rvs(scale=0.1,size=len(run_logistic.teval)) # suma ruido a los datos
    np.savetxt('time.txt',time.flatten())                               # guarda los tiempos de solucion
    np.savetxt('data.txt',data.flatten())                               # guarda los datos 
   
   #Sección para graficar ambas listas de datos
    t = np.loadtxt("time.txt")
    y = np.loadtxt("data.txt")
    plt.plot(t,y)
    plt.show()    
    #Mostrar contenido de archivos txt
    print(t)
    print(y)