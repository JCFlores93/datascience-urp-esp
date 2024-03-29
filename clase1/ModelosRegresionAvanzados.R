
rm(list=ls())
##########################################################################
##### -- Programa de Especializaci�n en Data Science - Nivel I -- ########
##########################################################################
######## Tema : Modelos de Regresion Avanzados ###########################
######## Autores: Jose Cardenas - Andre Chavez  ########################## 
##########################################################################

############################################
# AN�LISIS DE REGRESION SIMPLE Y MULTIPLE #
############################################

############################################
# AN�LISIS DE REGRESION SIMPLE#############
############################################

rm(list=ls())

# Librerias necesarias para el analisis

library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ISLR)
library(foreign)

# Leemos la data y nos hacemos las siguientes preguntas:

# �Existe una relaci�n entre el presupuesto de marketing y las ventas?
# que fuerte es la relaci�n (si existe)?
# que tipo de medio contribuye más a las ventas?
# que precisamente podemos estimar el efecto de cada uno de los tipos de medios sobre las ventas?
# que precisamente podemos predecir las ventas futuras?
# �La relaci�n es lineal?
# �Hay complementariedad entre los tipos de medio?

data <- read.csv("Advertising.csv") ## Cargar la data
str(data) # ver la estructura de la data
View(data)
# Analisis Univariado de la data
summary(data) # Tabla resumen
boxplot(data[,2:5]) # gr�fico de cajas
cor(data$TV,data$radio) # por defecto utiliza pearson
cor(data$TV,data$radio, method = 'spearman') # correlacion de spearman , si es similiar significa que tiende a ser una distribucion normal
#Si no supera al 0.6, puede ir al modelo
# Analisis Bivariado de la data
correlacion<-cor(data[,2:5], method = 'spearman') # Calculando las matrices de correlaciones.
# Usamos el metodo de spearman para evitar el supuesto de normalidad(no param�trica).
# Las correlaciones no deben superar 0.6, en las Xs
# Si superan al 0.6, se toma la que tenga mayor correlacion con el target
library(corrplot)
corrplot(correlacion, method="number", type="upper")
library("PerformanceAnalytics")
chart.Correlation(data[,2:5], histogram=TRUE, pch=19)
library(psych)
pairs.panels(data[,2:5], scale=TRUE)
library(corrplot)
corrplot.mixed(cor(data[,2:5]), order="hclust", tl.col="black")
library(GGally)
ggpairs(data[,2:5])
ggcorr(data[,2:5], nbreaks=8, palette='RdGy', label=TRUE, label_size=5, label_color='white')
library(ggcorrplot)
ggcorrplot(cor(data[,2:5]), p.mat = cor_pmat(mtcars), hc.order=TRUE, type='lower')

# Ajustamos un modelo lineal entre las ventas y el monto invertido en publicidad por TV
m <- lm(sales ~ TV, data = data)

# Vemos un resumen del modelo
summary(m)

# Predecir sobre nuevos registros
x_nuevos<-data.frame(TV=c(45,1000))
predict(m,x_nuevos)

# Guardar un Modelo Predictivo
saveRDS(m,"Modelo_Regresion.rds")

# Obtenemos los valores ajustados o predichos
# para cada x sacamos un y estimado
data$fitted <- m$fitted.values
# Podemos ver también los residuales
data$residual <- m$residuals

ggplot(data = data, aes(x = TV, y = sales)) + geom_point(color = "red") +
        geom_line(aes(y = fitted), color = "blue") +
        geom_segment(aes(x = TV, xend = TV, y = sales, yend = fitted, color="Distancia"), color = "grey80") +
        labs(xlab = "Presupuesto para TV", ylab = "Ventas") + 
        theme_bw()


##############################################
# AN�LISIS DE REEGRESION MULTIPLE#############
##############################################

# Ajustamos un modelo lineal entre las ventas y el monto invertido en publicidad por TV
mm <- lm(sales ~ TV+radio, data = data)

# Vemos un resumen del modelo
summary(mm)

# Obtenemos los valores ajustados o predichos
data$fittedmm <- mm$fitted.values
# Podemos ver también los residuales
data$residualmm <- mm$residuals

ggplot(data = data, aes(x = TV, y = sales)) + geom_point(color = "red") +
        geom_line(aes(y = fittedmm), color = "blue") +
        geom_segment(aes(x = TV, xend = TV, y = sales, yend = fittedmm, color="Distancia"), color = "grey80") +
        labs(xlab = "Presupuesto para TV", ylab = "Ventas") + 
        theme_bw()

# Ejercicio

# 1. Genera los modelos simples para TV, radio y peri�dico.
# 2. Genera el modelo conjunto para los 3 tipos de medio.
# 3. Compara los coeficientes entre los modelos simples y el modelo conjunto.


# SUPUESTOS TIPICOS DE LOS MODELOS LINEALES
# Los problemas más comunes al ajustar un modelo de regresión son:

# 1. La no linealidad de las relaciones entre la respuesta y los predictores.
# 2. Correlaci�n entre los errores
# 3. Varianza no-constante en los términos de error (*heterocedasticidad*)
# 4. Outliers
# 5. Puntos con alta influencia
# 6. Colinealidad

##################
## No linealidad##
##################
# Si la relaci�n entre la respuesta y los regresores no es lineal, las conclusiones
# que extraemos de un modelo lineal no son generalmente correctas. Además, nuestro
# poder predictivo se ve muy reducido.
plot(data$residual)

#######################
## Heterocedasticidad##
#######################

# Otro supuesto importante en los modelos de regresión es que los términos de 
# error tienen varianza constante, es decir, $var(\epsilon_i) = \sigma^2$.
# Los errores estándar de los coeficientes, los intervalos de confianza y las pruebas
# de hipoesis que asociamos a un modelo de regresi�n *dependen* de que este 
# supuesto se cumpla. 

# En la realidad, este supuesto se viola f�cilmente. Por ejemplo, las varianzas 
# pueden aumentar en la medida en que aumenta el valor de la variable de respuesta.
plot(data$fitted,data$residual)

#############
## Outliers##
#############
# Un outlier es un putno para el que $y_i$ está muy lejos del valor $\hat{y_i}$.
# Los outliers pueden ser un error en la captación de la información o puede 
# ser simplemente una observación extraña verdadera.

# Podemos introducir outliers a una base de datos para ejemplificar su efecto

cars1 <- cars[1:30, ]  # original data
cars_outliers <- data.frame(speed=c(19,19,20,20,20), dist=c(190, 186, 210, 220, 218))  # introduce outliers.
cars2 <- rbind(cars1, cars_outliers)

cars.compare <- rbind(
        mutate(cars1, base = "sin.outliers"), mutate(cars2, base = "con.outliers")
)

ggplot(cars.compare, aes(x = speed, y = dist)) + geom_point() + facet_wrap(~ base) + 
        geom_smooth(method = 'lm')


# Podemos detectar outliers de varias maneras, una de ellas es a trav�s de 
# las distancias de cook.
m <- lm(data = cars2, dist ~ speed)
cooksd <- cooks.distance(m)
cooksd
plot(m)

# Eliminar outliers es una decisi�n muy fuerte. Normalmente, se busca ser cuidadoso
# cuando se toma una decisión como estas.
#saveRDS(m,"Reg_Lineal.rds")
#modelo_lr    <-   readRDS("Reg_Lineal.rds")

##########################################################

## Practicar
#ingreso=read_excel("polingreso.xlsx")
#nuevo=read_excel("nuevoproducto.xlsx")

############################################
# AN�LISIS DE REGRESION PENALIZADAS ########
############################################

######## REGRESI�N RIDGE Y LASSO ###########

## Cargar la data

carros <- read.csv("carros2011imputado2.csv")
str(carros)
#Objetivo modelar el precio promedio del auto.
matrizcorrelacion <- cor(carros[, 2:17], method = 'spearman') # Escogemos s�lo las variables cuantitativas, y sin el ID.
write.csv(matrizcorrelacion, "matriz_carros_correlacion.csv")

## modelo Regresion Ridge

## Particion Muestral (train 80% y test 20%)
set.seed(1234) 
# sample.int -> asigna un n�mero entero
# nrow(carros) -> n�mero total de registros
# .8*nrow(carros) -> 80% de la data
sample <- sample.int(nrow(carros), round(.8*nrow(carros)))
# Asignamos el 80% de la data en train
# Guardamos la data en matriz
carros.train<- carros[sample, ]
# Asignamos el 20% de la data en validation/train
# Guardamos la data en matriz
carros.validation <- carros[-sample, ]

# x train 
precio.train=as.matrix(carros.train$precio_promedio)
# x test
predictores.train=as.matrix(carros.train[,2:17])

# y train
precio.validation=as.matrix(carros.validation$precio_promedio)
# y test
predictores.validation=as.matrix(carros.validation[,2:17])

library(glmnet)

## Modelado

# alpha=0 -> es ridge
fitridge=glmnet(predictores.train,precio.train,alpha=0)
fitridge$beta
plot(fitridge) # las q se alejan mas son las mas importantes

## Encontrar los mejores coeff
# cv -> cross validation 
# glmnet -> penalizacion
# nfolds -> cantida de pruebas
foundrigde=cv.glmnet(predictores.train,precio.train,alpha=0,nfolds=5)
plot(foundrigde) # con landa de log de 0 a 2 se estabiliza
attributes(foundrigde)
foundrigde$lambda
foundrigde$lambda.1se # muestra el landa optimo sugerencia
foundrigde$lambda.min 

coef(fitridge,s=foundrigde$lambda.1se) # Muestra los coeficintes para cada lambda
coef(fitridge,s=foundrigde$lambda.min)

## Indicadores
# El menor error es mejor.
#Estamos prediciendo en base al lambda del negocio
prediridge=predict(foundrigde,predictores.validation,s="lambda.min")
ridgemse=sqrt(mean((prediridge-precio.validation)^2))
ridgemse

#Estamos prediciendo en base al lambda optimo
prediridge1=predict(foundrigde,predictores.validation,s="lambda.1se")
ridgemse1=sqrt(mean((prediridge1-precio.validation)^2))
ridgemse1

## modelo Regresion Lasso

## aplha 1 es cambiar la norma con la 1
fitlasso=glmnet(predictores.train,precio.train,alpha=1)

## Encontrar los mejores coeff

founlasso=cv.glmnet(predictores.train,precio.train,alpha=1,nfolds=5) 
plot(founlasso)
founlasso$lambda.1se # muestra el landa optimo sugerencia
founlasso$lambda.min # muestra el landa de negocio

# Cuando se muestre un punto significa que se ha eliminado
coef(fitlasso,s=founlasso$lambda.1se)
coef(fitlasso,s=founlasso$lambda.min)

## Indicadores

# Lambda de negocio
predilasso=predict(founlasso,predictores.validation,s="lambda.min")
lassomse=sqrt(mean((predilasso-precio.validation)^2))
lassomse

# Lambda optimo
predilasso1=predict(founlasso,predictores.validation,s="lambda.1se")
lassomse1=sqrt(mean((predilasso1-precio.validation)^2))
lassomse1

## modelo Regresion mediante redes elasticas

## aplha 0.5 es cambiar la norma con la 0.5
fitnet=glmnet(predictores.train,precio.train,alpha=0.5)

## Encontrar los mejores coeff

founnet=cv.glmnet(predictores.train,precio.train,alpha=0.5,nfolds=5) 
plot(founnet)
founnet$lambda.1se # muestra el lambda optimo sugerencia
founnet$lambda.min # muestra el lambda de negocio

coef(fitnet,s=founnet$lambda.1se)
coef(fitnet,s=founnet$lambda.min)

## Indicadores

predinet=predict(founnet,predictores.validation,s="lambda.min")
netmse=sqrt(mean((predinet-precio.validation)^2))
netmse

predinet1=predict(founnet,predictores.validation,s="lambda.min")
netmse1=sqrt(mean((predinet1-precio.validation)^2))
netmse1


#comparacion
cbind(ridgemse,lassomse,netmse,ridgemse1,lassomse1,netmse1)

