#########################################################################
#------- Programa de Especializacion en Data Science ------------########
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema: Regresion Logistica - Arboles Decision - Indicadores de Validacion
      # Naive Bayes - KNN
# version: 1.0
#########################################################################


#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls()) ; dev.off()

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#---------------------------------------------------------
# Paquetes 
library(MASS) 
library(pROC)
library(foreign)
library(gmodels)
library(InformationValue)
library(partykit)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(ggplot2)
library(MLmetrics)
library(ISLR)

##############################
#  CASO 2. Fuga de Clientes  #
##############################

# El objetivo es predecir clientes propensos a desafilarse 
# (tambien llamado churn o attrition) en una empresa de 
# telecomunicaciones.
# Se cuenta con una data de 1345 clientes de una empresa de 
# telecomunicaciones donde algunos siguen siendo clientes (Actual) 
# y otros han fugado de manera voluntaria (Fuga).
#
# Variable Dependiente:  
#           CHURN   (0=Cliente Actual, 1=Fuga Voluntaria)
# Variables Independentes:
#           EDAD    (Edad del cliente en anos)  
#           SEXO    (Sexo del cliente, 1=Fememino 2=Masculino)
#           CIVIL   (Estado civil del cliente, 1=Casado 2=Soltero) 
#           HIJOS   (Numero de hijos del cliente)
#           INGRESO (Ingresos anuales del cliente)
#           AUTO    (Si el cliente es dueno de un auto, 1=Si 2=No)  
# Variable de identificacion: 
#           ID      (Codigo del cliente)

#----------------------------------------------------------
# Preparacion de los datos
library(foreign)
datos.r <-read.spss("Churn-arboles.sav",
                  use.value.labels=TRUE, 
                  to.data.frame=TRUE)
str(datos.r)
View(datos.r)
# No considerar la variable de identificaci?n ID
datos.r$ID <- NULL
str(datos.r)

# Etiquetando las opciones de las variables categoricas
levels(datos.r$SEXO)
levels(datos.r$SEXO)  <- c("Fem","Masc")
levels(datos.r$CIVIL) <- c("Casado","Soltero")
levels(datos.r$AUTO)  <- c("Si","No")
levels(datos.r$CHURN) <- c("No Fuga","Fuga")

str(datos.r)

# Direccionar a r, al dataset
attach(datos.r)

# Para cambiar la categoria de referencia
datos.r$SEXO = relevel(datos.r$SEXO,ref="Masc") # Cambio de referencia
contrasts(datos.r$SEXO)

#-------------------------------------------------------------------
# Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
library(caret)
set.seed(123) 
index <- createDataPartition(datos.r$CHURN, p=0.7, list=FALSE)
training <- datos.r[ index, ] # Entrenamiento del modelo
testing <-  datos.r[-index, ] # Validacion o test del modelo

# Verificando la estructura de los datos particionados
prop.table(table(datos.r$CHURN)) # Distribucion de y en el total
prop.table(table(training$CHURN))
prop.table(table(testing$CHURN))

###################################################
######## REGRESION LOGISTICA BINARIA ##############
###################################################

options(scipen=999)

modelo_churn <- glm(CHURN ~ . ,  # Todas las variables
                    family=binomial,
                    data=training) # Data de entrenammiento

summary(modelo_churn)
coef(modelo_churn)

#--------------------------------------------------------------------------
# Cociente de ventajas (Odd Ratio)
exp(coef(modelo_churn))

# (Intercept)       EDAD    SEXOFem  CIVILSoltero        HIJOS      INGRESO 
#   0.1613089  1.0081315  12.3560732    1.0774856    0.9425963    0.9999911 
#      AUTONo 
#   0.9016556 

# Para el caso de SEXO, el valor estimado 12.356 significa que, 
# manteniendo constantes el resto de las variables, 
# que las personas del g?nero FEMENINO tienen 12.356 veces m?s ventaja 
# de FUGAR que los sujetos que son del g?nero MASCULINO.

# Para el caso de la EDAD, ante un incremento en una unidad de medida de 
# la EDAD (un a?o), provocar? un incremento multiplicativo por un factor 
# de 1.008 de la ventaja de FUGA 

cbind(Coeficientes=modelo_churn$coef,ExpB=exp(modelo_churn$coef))

#------------------------------------------------------------------------
# Cociente de ventajas e Intervalo de Confianza al 95% 
library(MASS)
exp(cbind(OR = coef(modelo_churn),confint.default(modelo_churn)))

#----------------------------------------------------------
# Importancia de las variables, Feature Selection*
varImp(modelo_churn)

#----------------------------------------------------------
# Seleccion de Variables  
library(MASS)
step <- stepAIC(modelo_churn,direction="backward", trace=FALSE)
step$anova

#----------------------------------------------------------
# Modelo 2 con las variables mas importantes
modelo_churn2 <- glm(CHURN ~ EDAD + SEXO + INGRESO, 
                     family=binomial,
                     data=training)

summary(modelo_churn2)
coef(modelo_churn2)

library(MASS)
exp(cbind(OR = coef(modelo_churn2),confint.default(modelo_churn2)))

# Como tenemos un modelo parsimonioso creado , lo guardamos.
saveRDS(modelo_churn2,"Rg_Logistica.rds")
#----------------------------------------------------------
######### Despliegue o Score de Nuevos Indiviuos ##########
#----------------------------------------------------------
# Prediccion para nuevos individuos  
nuevo1 <- data.frame(EDAD=57, SEXO="Fem",INGRESO=27535.30)
predict(modelo_churn2,nuevo1,type="response")

nuevo2 <- data.frame(EDAD=80, SEXO="Fem",INGRESO=12535.50)
predict(modelo_churn2,nuevo2,type="response")


############################################
#  INDICADORES PARA EVALUACION DE MODELOS  #
############################################

#-----------------------------------------------------------------
# Para la evaluacion se usara el modelo_churn2 obtenido con la 
# muestra training y se validara en la muestra testing

# Prediciendo la probabilidad
proba.pred <- predict(modelo_churn2,testing,type="response")
head(proba.pred)

# Prediciendo la clase (con punto de corte = 0.5)
clase.pred <- ifelse(proba.pred >= 0.5, 1, 0)

head(clase.pred)

str(clase.pred)

# Convirtiendo a factor
clase.pred <- as.factor(clase.pred)          

levels(clase.pred) <- c("No Fuga","Fuga")

str(clase.pred)

head(cbind(testing,proba.pred,clase.pred),8)

write.csv(cbind(testing,proba.pred,clase.pred),
          "Testing con clase y proba predicha-Logistica.csv")

# Graficando la probabilidad predicha y la clase real
ggplot(testing, aes(x = proba.pred, fill = CHURN)) + 
  geom_histogram(alpha = 0.25)

#############################
# 1. Tabla de clasificacion #
#############################

#---------------------------------------------
# Calcular el % de acierto (accuracy)
accuracy <- mean(clase.pred==testing$CHURN)
accuracy

#---------------------------------------------
# Calcular el error de mala clasificacion
error <- mean(clase.pred!=testing$CHURN)
error

library(gmodels)
CrossTable(testing$CHURN,clase.pred,
           prop.t=FALSE, prop.c=FALSE,prop.chisq=FALSE)

# Usando el paquete caret
library(caret)
caret::confusionMatrix(clase.pred,testing$CHURN,positive="Fuga")

############################
# 2. Estadistico de Kappa  #
############################

# Tabla de Clasificaci?n
addmargins(table(Real=testing$CHURN,Clase_Predicha=clase.pred))

#           Clase_Predicha
# Real     Actual Fuga Sum
# Actual      168   81 249
# Fuga         25  128 153
# Sum         193  209 402

# pr_o es el Accuracy Observado o la Exactitud Observada del modelo
pr_o <- (168+128)/402 ; pr_o

# pr_e es el Accuracy Esperado o la Exactitud Esperada del modelo
pr_e <- (249/402)*(193/402) + (153/402)*(209/402) ; pr_e

# Estad?stico de Kappa
# Mientras mas cerca a 1 es mejor.
k <- (pr_o - pr_e)/(1 - pr_e) ; k

#####################################
# 3. Estadistico Kolgomorov-Smirnov #
#####################################

#---------------------------------
# Calculando el estadistico KS
# Areas de marketing,
library(InformationValue)

ks_stat(testing$CHURN,proba.pred, returnKSTable = T)
ks_stat(testing$CHURN,proba.pred)

# Graficando el estadistico KS 
# De menor a mayor, construimos deciles.
# la maxima diferencia entra las fugas y las no fugas.
ks_plot(testing$CHURN,proba.pred)

#####################################
# 4. Curva ROC y area bajo la curva #
#####################################

#----------------------------------------------
# Usando el paquete pROC
library(pROC)

# Area bajo la curva
roc <- roc(testing$CHURN,proba.pred)
roc$auc

#---------------------------------------------------
# Curva ROC usando el paquete caTools
library(caTools)
AUC <- colAUC(proba.pred,testing$CHURN, plotROC = TRUE)
abline(0, 1,col="red") 

AUC  # Devuelve el area bajo la curva

puntos.corte <- data.frame(prob=roc$thresholds,
                           sen=roc$sensitivities,
                           esp=roc$specificities)

# Mientras mejor sensibilidad y mejor especificidad es mejor.
head(puntos.corte)

# Punto de corte optimo (mayor sensibilidad y especificidad) usando pROC
coords(roc, "best",ret=c("threshold","specificity", "sensitivity","accuracy"))
coords(roc, "best")

plot(roc,print.thres=T)

# Graficando la Sensibilidad y Especificidad
ggplot(puntos.corte, aes(x=prob)) + 
  geom_line(aes(y=sen, colour="Sensibilidad")) +
  geom_line(aes(y=esp, colour="Especificidad")) + 
  labs(title ="Sensibilidad vs Especificidad", 
       x="Probabilidad") +
  scale_color_discrete(name="Indicador") +
  geom_vline(aes(xintercept=0.5507937),
             color="black", linetype="dashed", size=0.5) + 
  theme_replace() 


##########################
# 5. Coeficiente de Gini #
##########################

gini <-  2*AUC -1 ; gini

################
# 6. Log Loss  #
################

# Transformar la variable CHURN a numerica
real <- as.numeric(testing$CHURN)
head(real)
# [1] 1 1 2 1 1 2

# Recodificar los 1 y 2 como 0 y 1 respectivamente
real <- ifelse(real==2,1,0)

library(MLmetrics)
LogLoss(proba.pred,real)

###################################################
## ARBOL DE CLASIFICACION CON EL ALGORITMO CART ###
###################################################

library(rpart)
attach(datos.r) # Direcciona r a los datos 
unique(SEXO)
sort(unique(EDAD))

table(CHURN)
prop.table(table(CHURN))

table(SEXO,CHURN)
prop.table(table(SEXO,CHURN),margin = 1)

sort(unique(INGRESO))


#---------------------------------------------------------------
# Ejemplo 1: Arbol con los parametros por defecto
set.seed(123)
arbol1 <- rpart(CHURN ~ . , 
                data=datos.r, 
                method="class")

# Si usa method="anova" es para Modelos de Regresi?n

# Graficando el arbol
library(rpart.plot)

rpart.plot(arbol1, digits=-1, type=0, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=1, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=2, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=3, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=4, extra=101,cex = .7, nn=TRUE)
# Tomar el lado derecho  que es nuestro objetivo

# Mejorando los Graficos
library(partykit)
plot(as.party(arbol1), tp_args = list(id = FALSE))

#------------------------------------------------------------------
# Ejemplo 2: Arbol controlando parametros
# Parametros 
# minsplit:   Indica el numero minimo de observaciones en un nodo para
#             que este sea dividido. Minimo para que un nodo sea padre. 
#             Esta opcion por defecto es 20.
# minbucket:  Indica el numero minimo de observaciones en cualquier
#             nodo terminal. Por defecto esta opci?n es el valor 
#             redondeado de minsplit/3.
# cp:         Par?metro de complejidad. Indica que si el criterio de 
#             impureza no es reducido en mas de cp*100% entonces se 
#             para. Por defecto cp=0.01. Es decir, la reducci?n en la 
#             impureza del nodo terminal debe ser de al menos 1% de la
#             impureza inicial.
# maxdepth:   Condiciona la profundidad maxima del arbol. 
#             Por defecto esta establecida como 30.

arbol1$control
# rpart.control(minsplit = 20, minbucket = round(minsplit/3), cp = 0.01, 
# maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
# surrogatestyle = 0, maxdepth = 30, ...)

set.seed(123)
arbol2 <- rpart(CHURN ~ . , 
                data=datos,
                control=rpart.control(minsplit=90, minbucket=30),
                method="class")

rpart.plot(arbol2, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

#---------------------------------------------------------------------
# Ejemplo 3: Controlando el crecimiento del arbol
# con el parametro de complejidad (cp=0.05)

set.seed(123)
arbol3 <-  rpart(CHURN ~ . , 
                 data=datos,
                 control=rpart.control(minsplit=90, minbucket=30,cp=0.05),
                 method="class")

rpart.plot(arbol3, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

printcp(arbol3)

#----------------------------------------------------------------------
# Ejemplo 4: cp=0.001 para obtener un arbol con mas ramas

set.seed(123)
arbol4 <- rpart(CHURN ~ . ,
                data=datos, 
                method="class",
                cp=0.001)

rpart.plot(arbol4, digits=-1, type=2, extra=101, cex = 0.7,  nn=TRUE)

printcp(arbol4)

#--------------------------------------------------------------------
# Ejemplo 5: Recortar el arbol (prune)

arbol5 <- prune(arbol4,cp=0.1)

rpart.plot(arbol5, digits=-1, type=2, extra=101,cex = 0.7, nn=TRUE)
printcp(arbol5)

arbol6 <- prune(arbol4,cp=0.01)

rpart.plot(arbol6, digits=-1, type=2, extra=101, cex = .7, nn=TRUE)

printcp(arbol6)


#----------------------------------------------
# Ejemplo 6: Valor optimo de CP

set.seed(123)
arbol.completo <- rpart(CHURN ~ . ,
                        data=datos,
                        method="class",
                        cp=0, 
                        minbucket=0)
arbol.completo
printcp(arbol.completo)

plotcp(arbol.completo)

rpart.plot(arbol.completo, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

# Seleccionando el cp optimo
# arbol.pruned <- prune(arbol.completo,cp=0.00292398)
xerr <- arbol.completo$cptable[,"xerror"]
xerr

minxerr <- which.min(xerr)
minxerr

mincp <- arbol.completo$cptable[minxerr, "CP"]
mincp

arbol.pruned <- prune(arbol.completo,cp=mincp)

printcp(arbol.pruned)
plotcp(arbol.pruned)

rpart.plot(arbol.pruned, type=2, extra=101, cex = 0.7, nn=TRUE)

clase.cart <- predict(arbol.pruned,datos,type="class")
proba.cart <- predict(arbol.pruned, datos, type = "prob")
head(proba.cart)
proba.cart <- proba.cart[,2]

datoscart <- cbind(datos,clase.cart, proba.cart)
head(datoscart)

# Usando el arbol para convertir variable numerica a categorica

arbol7 <- rpart(CHURN ~ INGRESO, data=datos, method="class", cp=0.005)

rpart.plot(arbol7, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)         

datos2 <- datos

datos2$INGRESO.CAT <- cut(datos2$INGRESO, 
                          breaks = c(-Inf,27900,Inf),
                          labels = c("Menos de 27900", "De 27900 a m?s"),
                          right = FALSE)

table(datos2$INGRESO.CAT)

prop.table(table(datos2$CHURN))
prop.table(table(datos2$INGRESO.CAT,datos2$CHURN),1)


##########################################################
### REGRESION LOGISTICA CON CARET Y VALIDACION CRUZADA ###
##########################################################

# Relacion de parametros a ajustar de un modelo
modelLookup(model='glm')

# Aplicando el modelo con Validaci?n Cruzada 
ctrl <- trainControl(method="cv",number=10)

set.seed(123)
modelo_log <- train(CHURN ~ ., 
                    data = data.train, 
                    method = "glm", family="binomial", 
                    trControl = ctrl, 
                    tuneLength = 5,
                    metric="Accuracy")
modelo_log

summary(modelo_log)

plot(modelo_log)

varImp(modelo_log)

##########################################
### CART CON CARET Y VALIDACION CRUZADA ##
##########################################

# Relacion de modelos 
library(caret)
names(getModelInfo())

# Relacion de parametros a ajustar de un modelo
modelLookup(model='rpart')

# Aplicando el modelo con Validacion Cruzada 
ctrl <- trainControl(method="cv", number=10)

# Si se desea usar Validaci?n Cruzada Repetida
# ctrl <- trainControl(method="repeatedcv", repeats = 3, number=10)

set.seed(123)
modelo_cart <- train(CHURN ~ ., 
                     data = data.train, 
                     method = "rpart", 
                     trControl = ctrl, 
                     tuneLength= 20,
                     metric="Accuracy")

# tuneGrid = expand.grid(cp=seq(0,0.05,length=100))
modelo_cart

plot(modelo_cart)

varImp(modelo_cart)

##########################################################
########## ARBOL C50 : BOOSTING DE ARBOLES ###############
##########################################################

library(C50)
arbol.c50 <- C5.0(CHURN~.,data = ,
                  trials = 55, # Numero de arboles
                  rules= TRUE, # Reglas de seleccion
                  tree=FALSE,  
                  winnow=TRUE) # Seleccion de variables

proba5=predict(arbol.c50, newdata=,
               type="prob")[,2] # Dame la segunda columna, es decir 
# la probabilidad del default

##########################################################
########## ARBOL CHAID : ARBOLES ESTADISTICOS ############
##########################################################

library(partykit)
# Y ~ X
set.seed(100)
arbol.chaid<-ctree(CHURN~ .,data = ,
                   control=ctree_control(mincriterion=0.98),
                   maxdepth=3)

proba3=predict(arbol.chaid, newdata=,
               type="response")
proba3

tabla=confusionMatrix(proba3,
                      ,positive = "1")
tabla
windows()
plot(modelo3,type='simple') 

##########################################################
########## K VECINOS MAS CERCANOS ############
##########################################################

# Algoritmo de K-NN
library(class)
# knn
pred_knn <- knn(X_train,X_test,
                cl=Y_train,
                10)
pred_knn

# Calcular la matriz de confusion
tabla=confusionMatrix(pred_knn,
                      ,positive = "1")
tabla

##########################################################
########## NAIVE BAYES ##################################
##########################################################

library(e1071)
# Entreno el modelo de ML
naive.B=naiveBayes(CHURN~., # Y ~ X
                   data=training)

# Predecir para validar el modelo
pred_bayes<- predict(naive.B, testing) # le enviamos el 30%

# Lo valido con mi matriz de consusion
tabla=caret::confusionMatrix(pred_bayes, testing$CHURN ,positive = "Fuga")



# FIN !!