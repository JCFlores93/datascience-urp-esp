#########################################################################
#------- Programa de Especializacion en Data Science ------------########
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Tema: Series de Tiempo
# version: 2.0
#########################################################################

# Librerias basicas para el estudio de series temporales

library(ggplot2)  # Graficas y visualizacion
library(TSA)      # Formato y trabajar con series de tiempo
library(forecast) # Estimaciones y pronosticos de series de tiempo
library(scales)   # Preprocesamiento de los datos
library(stats)    # Pruebas estadisticas


#Leer la serie de tiempo, desde un archivo csv.
#setwd("C:/Users/Andre Chavez/Desktop/Data")
pbi=read.csv("pbi.csv",header = T,sep="",dec=",")
View(pbi)

##Creacion de una serie de tiempo
pbits<-ts(pbi,start=c(1991,1),end=c(2009,12),frequency=12)
fechas = seq(as.Date("1991/1/1"), length.out = length(pbits), by = "months")
pbits

# Grafica de la serie de tiempo
plot(pbits,
     main="PBI (Enero 1991 - Diciembre 2009)",
     ylab="PBI",
     xlab="Años")

# Graficamos las cajas agregadas para observar si existe estacionalidad
windows(width=800,height=350) 
boxplot(split(pbits, cycle(pbits)), names = month.abb, col = "gold")


#############################################################
############ Enfoque de Descomposicion ######################
#############################################################

# Usamos el comando decompose para descomponer la serie de tiempo
Yt_desc = decompose(pbits,type = "multiplicative",filter = NULL)

# Objetos output del metodo de descomposicion

Yt_desc$x # Series original
Yt_desc$seasonal # Estimacion de la estacionalidad
Yt_desc$trend # Estimacion de la tendencia
Yt_desc$random # Estimacion de la aleatorialidad

# Grafico de descomposicion de la serie
plot(Yt_desc , xlab='Año')

# Serie original
pbits_original<-Yt_desc$x
View(pbits_original)

# Coeficientes estacionales
Coeficientes_Estacionales<-Yt_desc$seasonal
plot(Coeficientes_Estacionales)

# Tendencia de la serie
Yt_desc$trend

# Tipo de modelo aplicado
Yt_desc$type

# A la serie original,le quitamos la componente de estacionalidad, nos quedamos
# solo con la tendencia.
Tendencia_pbi<-pbits_original/Coeficientes_Estacionales
plot(Tendencia_pbi)

# Debido a que nos hemos quedado solo con la tendencia
Tendencia_pbi<-as.double(Tendencia_pbi)

#Viendo solo la componente tendencia, le ajuste la curva que mejor modele su
#comportamiento o que mejor la ajuste.

# Ajusto cualquire modelo para 
T = length(Tendencia_pbi)
yi = Tendencia_pbi[1:T]


# Ajustar 4 modelos: lineal, cuadratico, cubico
t = seq(1:T) # Creo un termino lineal
t2 = t**2 # Creo un termino cuadr�tico
t3 = t**3 # Creo un termino c�bico

# Ajuste de Polinomiales a la Componente Tendencia

mod.lin = lm(yi~t)
mod.cuad = lm(yi~t+t2)
mod.cub = lm(yi~t+t2+t3)


summary(mod.lin)
summary(mod.cuad)
summary(mod.cub)

# Tenemos las estimaciones del modelo lineal, cuadratico y cubico
ajust_lineal  <- mod.lin$fitted.values # Estimacion de la tendencia
ajust_cuadrado<- mod.cuad$fitted.values
ajust_cubico  <- mod.cub$fitted.values

# Construimos la estimacion de la Zt serie de tiempo
estimacion_lineal    <- ajust_lineal*Coeficientes_Estacionales
estimacion_cuadratico<- ajust_cuadrado*Coeficientes_Estacionales
estimacion_cubico    <- ajust_cubico*Coeficientes_Estacionales

# Pronostico 
forecast(estimacion_lineal, h = 100)

# Graficamos
# plot(forecast(estimacion_cubico,h=12),col=3)
plot(forecast(estimacion_cubico,h=12),col=3)
lines(pbits,col=1)
lines(estimacion_lineal, col=1)
lines(estimacion_cuadratico, col=7)

legend("topleft", lty=1, col=c(2,3,1,7),
       legend=c("Est.Cubica","Datos Ori.",
                "Est.Lineal","Est.Cuadratica"),
       bty="n")

# Validacion de Modelos
accuracy(estimacion_lineal,pbits)

# FIN!!
