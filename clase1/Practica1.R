rm(list=ls())

library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ISLR)
library(foreign)

# Leemos la data y nos hacemos las siguientes preguntas:

# ¿Existe una relación entre el presupuesto de marketing y las ventas?
# que fuerte es la relación (si existe)?
# que tipo de medio contribuye mÃ¡s a las ventas?
# que precisamente podemos estimar el efecto de cada uno de los tipos de medios sobre las ventas?
# que precisamente podemos predecir las ventas futuras?
# ¿La relación es lineal?
# ¿Hay complementariedad entre los tipos de medio?

data <- read.csv("grasacorporal.csv") ## Cargar la data
str(data) # ver la estructura de la data
View(data)

# Analisis Univariado de la data
summary(data) # Tabla resumen

correlacion<-cor(data[,], method = 'spearman') # Calculando las matrices de correlaciones.

grasacorporal <- data$grasacorp
data[2]<-NULL
data$grasacorp <- grasacorporal

library(corrplot)
corrplot(correlacion, method="number", type="upper")

## Particion Muestral (train 80% y test 20%)
set.seed(1234)

sample <- sample.int(nrow(data), round(.7*nrow(data)))

data.train<- data[sample, ]
data.validation <- data[-sample, ]


# x train 
grasacorp.train=as.matrix(data.train$grasacorp)
# x test
predictores.train=as.matrix(data.train[,1:14])

# y train
grasacorp.validation=as.matrix(data.validation$grasacorp)
# y test
predictores.validation=as.matrix(data.validation[,1:14])

# Como seleccionar las variables
#library(dplyr)
#x = train %>% select(-grasacorp)
#y = train$grasacorp

#x= train[1,3:46]

## Variables seleccionadas por tener un buen p-value
#mm1 <- lm(grasacorp ~ densidad
#          +edad+peso+pecho+abdomen+tobillo
#          +biceps, data = data)
#summary(mm1)

## Eliminamos por no tener significancia para el modelo
mm1 <- lm(grasacorp ~ densidad, data = data.train)
summary(mm1)
x_nuevos<-data.frame(predictores.validation)
lmpredict <- predict(mm1,x_nuevos)
lmpredict

predictlm <- predict(mm1,grasacorp.validation$grasacorp)

lmse=sqrt(mean((predictlm-grasacorp.validation$grasacorp)^2))
lmse


# Obtenemos los valores ajustados o predichos
data$fittedmm <- mm1$fitted.values
# Podemos ver tambiÃ©n los residuales
data$residualmm <- mm1$residuals

ggplot(data = data, aes(x = densidad, y = grasacorp)) + geom_point(color = "red") +
  geom_line(aes(y = fittedmm), color = "blue") +
  geom_segment(aes(x = densidad, xend = densidad, y = grasacorp, yend = fittedmm, color="Distancia"), color = "grey80") +
  labs(xlab = "Densidad", ylab = "Grasa corporal") + 
  theme_bw()

# Predicción
#predict(mm1,)

################# Modelo Ridge ##################
library(glmnet)

## Modelado

## predictores
## grasacorp
# alpha=0 -> es ridge
fitridge=glmnet(predictores.train,grasacorp.train,alpha=0)
fitridge$beta
plot(fitridge) # las q se alejan mas son las mas importantes

## Encontrar los mejores coeff
# cv -> cross validation 
# glmnet -> penalizacion
# nfolds -> cantida de pruebas
foundrigde=cv.glmnet(predictores.train,grasacorp.train,alpha=0,nfolds=5)
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
ridgemse=sqrt(mean((prediridge-grasacorp.validation)^2))
ridgemse

#Estamos prediciendo en base al lambda optimo
prediridge1=predict(foundrigde,predictores.validation,s="lambda.1se")
ridgemse1=sqrt(mean((prediridge1-grasacorp.validation)^2))
ridgemse1


######################### modelo Regresion Lasso ##########################

## aplha 1 es cambiar la norma con la 1
fitlasso=glmnet(predictores.train,grasacorp.train,alpha=1)

## Encontrar los mejores coeff

founlasso=cv.glmnet(predictores.train,grasacorp.train,alpha=1,nfolds=5) 
plot(founlasso)
founlasso$lambda.1se # muestra el landa optimo sugerencia
founlasso$lambda.min # muestra el landa de negocio

# Cuando se muestre un punto significa que se ha eliminado
coef(fitlasso,s=founlasso$lambda.1se)
coef(fitlasso,s=founlasso$lambda.min)

## Indicadores

# Lambda de negocio
predilasso=predict(founlasso,predictores.validation,s="lambda.min")
lassomse=sqrt(mean((predilasso-grasacorp.validation)^2))
lassomse

# Lambda optimo
predilasso1=predict(founlasso,predictores.validation,s="lambda.1se")
lassomse1=sqrt(mean((predilasso1-grasacorp.validation)^2))
lassomse1


######################## modelo Regresion mediante redes elasticas ##############################

## aplha 0.5 es cambiar la norma con la 0.5
fitnet=glmnet(predictores.train, grasacorp.train,alpha=0.5)

## Encontrar los mejores coeff

founnet=cv.glmnet(predictores.train,grasacorp.train,alpha=0.5,nfolds=5) 
plot(founnet)
founnet$lambda.1se # muestra el lambda optimo sugerencia
founnet$lambda.min # muestra el lambda de negocio

coef(fitnet,s=founnet$lambda.1se)
coef(fitnet,s=founnet$lambda.min)

## Indicadores

predinet=predict(founnet,predictores.validation,s="lambda.min")
netmse=sqrt(mean((predinet-grasacorp.validation)^2))
netmse

predinet1=predict(founnet,predictores.validation,s="lambda.min")
netmse1=sqrt(mean((predinet1-grasacorp.validation)^2))
netmse1

#comparacion
cbind(ridgemse,lassomse,netmse,ridgemse1,lassomse1,netmse1)


