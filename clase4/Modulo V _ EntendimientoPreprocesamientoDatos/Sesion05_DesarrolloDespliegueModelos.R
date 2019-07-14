#########################################################################
#------- Programa de Especializacion en Data Science ------------########
#########################################################################

# Capacitador: Andr√© Omar Ch√°vez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema: Arboles Clasificacion: CART - CHAID - C50 - K-NN - Naive Bayes
# version: 2.0
#########################################################################


#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())
dev.off()
options(scipen=999)

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#--------------------------------------------
# Paquetes
library(foreign)
library(gmodels)
library(partykit)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(ggplot2)
library(MLmetrics)
library(randomForest)
library(ISLR)


#######################
# 1. LECTURA DE DATOS #
#######################

library(foreign)
datos <-read.spss("Churn-arboles.sav",
                  use.value.labels=TRUE, 
                  to.data.frame=TRUE)
str(datos)

# No considerar la variable de identificaci?n ID
datos$ID <- NULL
str(datos)

# Etiquetando las opciones de las variables categ?ricas
levels(datos$CHURN)
levels(datos$SEXO)  <- c("Fem","Masc")
levels(datos$CIVIL) <- c("Casado","Soltero")
levels(datos$AUTO)  <- c("Si","No")
levels(datos$CHURN) <- c("Actual","Fuga")

str(datos)

attach(datos) # Direccionar sus datos al dataframe

###################################################
# 2. ARBOL DE CLASIFICACION CON EL ALGORITMO CART #
###################################################

library(rpart)

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
                data=datos, 
                method="class")
arbol1
# Si usa method="anova" es para Modelos de Regresion / variables continuas

# Graficando el arbol
library(rpart.plot)

rpart.plot(arbol1, digits=-1, type=0, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=1, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=2, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=3, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=4, extra=101,cex = .7, nn=TRUE)

# Mejorando los Graficos
library(partykit)
plot(as.party(arbol1), tp_args = list(id = FALSE))

# En arboles debemos correr y luego ir mejorando.

#------------------------------------------------------------------
# Ejemplo 2: Arbol controlando parametros
# Parametros 
# minsplit:   Indica el numero minimo de observaciones en un nodo para
#             que este sea dividido. M?nimo para que un nodo sea padre. 
#             Esta opcion por defecto es 20.
# minbucket:  Indica el numero minimo de observaciones en cualquier
#             nodo terminal. Por defecto esta opcion es el valor 
#             redondeado de minsplit/3.
# cp:         Parametro de complejidad. Indica que si el criterio de 
#             impureza no es reducido en mas de cp*100% entonces se 
#             para. Por defecto cp=0.01. Es decir, la reduccion en la 
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
                control=rpart.control(
                        minsplit=500, 
                        minbucket=200),
                method="class")

rpart.plot(arbol2, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

#---------------------------------------------------------------------
# Ejemplo 3: Controlando el crecimiento del arbol
# con el parametro de complejidad (cp=0.05)

set.seed(123)
arbol3 <-  rpart(CHURN ~ . , 
             data=datos,
             # control=rpart.control(minsplit=90, minbucket=30,cp=0.0001),
             control=rpart.control(cp=0),
             method="class")

rpart.plot(arbol3, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

printcp(arbol3)

#----------------------------------------------------------------------
# Ejemplo 4: cp=0.001 para obtener un arbol con mas ramas

set.seed(123)
arbol4 <- rpart(CHURN ~ . ,
             data=datos, 
             method="class",
             cp=0.00001)

rpart.plot(arbol4, digits=-1, type=2, extra=101, cex = 0.7,  nn=TRUE)

printcp(arbol4)

#--------------------------------------------------------------------
# Ejemplo 5: Recortar el arbol (prune)

arbol5 <- prune(arbol4,cp=0.0029240)

rpart.plot(arbol5, digits=-1, type=2, extra=101,cex = 0.7, nn=TRUE)
printcp(arbol5)

arbol6 <- prune(arbol4,cp=0.01)

rpart.plot(arbol6, digits=-1, type=2, extra=101, cex = .7, nn=TRUE)

printcp(arbol6)

# nsplit = nodos temrinales.
# rel error = cantidad de particiones.
#  xerror = error promedio del arbol cuando se toma.
# xstd = desviacion estandar del error.
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

# Prediccion con Arboles
clase.cart <- predict(arbol.pruned,datos,type="class")    # Prediccion de la clase
proba.cart <- predict(arbol.pruned, datos, type = "prob") # Probabilidad 
head(proba.cart)

# Selecciono la probabilidad de fuga
proba.cart <- proba.cart[,2]

datoscart <- cbind(datos,clase.cart, proba.cart)
head(datoscart)

# Usando el arbol para convertir variable numerica a categorica

arbol7 <- rpart(CHURN ~ INGRESO, data=datos, method="class", cp=0.005)

windows()
rpart.plot(arbol7, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)         

datos2 <- datos # Hago una copia

datos2$INGRESO.CAT <- cut(datos2$INGRESO, 
                       breaks = c(-Inf,27900,Inf),
                       labels = c("Menos de 27900", "De 27900 a mas"),
                       right = FALSE)

table(datos2$INGRESO.CAT)

prop.table(table(datos2$CHURN))
prop.table(table(datos2$INGRESO.CAT,datos2$CHURN),1)


########################################################
# 3. DIVISION DE LA DATA EN MUESTRA DE TRAINING Y TEST #
########################################################

#-------------------------------------------------------------------
# Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
str(datos)                               # 1345 datos

library(caret)
set.seed(123) 

index      <- createDataPartition(datos$CHURN, p=0.7, list=FALSE)
data.train <- datos[ index, ]            # 943 datos trainig             
data.test  <- datos[-index, ]            # 402 datos testing

# Verificando que la particion mantenga las proporciones de la data
round(prop.table(table(datos$CHURN)),3)
round(prop.table(table(data.train$CHURN)),3)
round(prop.table(table(data.test$CHURN)),3)


##########################################
# 4. CART CON CARET Y VALIDACION CRUZADA #
##########################################

# Relacion de modelos 
library(caret)
names(getModelInfo())

# Relacion de parametros a ajustar de un modelo
modelLookup(model='rpart')

# Aplicando el modelo con Validacion Cruzada 
ctrl <- trainControl(method="cv", number=10)

# Si se desea usar Validacion Cruzada Repetida
# ctrl <- trainControl(method="repeatedcv", repeats = 3, number=10)

set.seed(123)
modelo_cart <- train(CHURN ~ ., 
                    data = data.train, # Data de train
                    method = "rpart",  # Algoritmo
                    trControl = ctrl,  # Control
                    tuneLength= 20,
                    metric="Accuracy")

                    # tuneGrid = expand.grid(cp=seq(0,0.05,length=100))
modelo_cart
# Se escoge el mejor cp acorde al accuracy y kappa
# el Ìndice Kappa compara el accuracy real y el accuracy esperado. 
# Kappa que tan bien clasificas entre 0 - 1, en una prediccion probabilistica marginal

plot(modelo_cart)

varImp(modelo_cart)


##################################################
# 5. COMPARANDO EL ENTRENAMIENTO DE LOS MODELOS #
##################################################

modelos  <- list(CART          = modelo_cart)

comparacion_modelos <- resamples(modelos) 
summary(comparacion_modelos)

dotplot(comparacion_modelos)
bwplot(comparacion_modelos)
densityplot(comparacion_modelos, metric = "Accuracy",auto.key=TRUE)

modelCor(comparacion_modelos)

###########################################################################
# 6. PREDICIENDO LA CLASE Y PROBABILIDAD CON LOS MODELOS EN LA DATA TEST #
###########################################################################

# 1. Prediccion de la clase y probabilidad con CART
CLASE.CART <- predict(modelo_cart,newdata = data.test )
head(CLASE.CART)

PROBA.CART <- predict(modelo_cart,newdata = data.test, type="prob")
PROBA.CART <- PROBA.CART[,2]
head(PROBA.CART)


##############################################################
# 7. EVALUANDO LA PERFOMANCE DE LOS MODELOS EN LA DATA TEST #
##############################################################

#------------------------------------------------------------
# 1. Evaluando la performance del modelo CART

# Tabla de clasificacion
library(gmodels)
CrossTable(x = data.test$CHURN, y = CLASE.CART,
           prop.t=FALSE, prop.c=FALSE, prop.chisq = FALSE)

addmargins(table(Real=data.test$CHURN,Clase_Predicha=CLASE.CART))
prop.table(table(Real=data.test$CHURN,Clase_Predicha=CLASE.CART),1)

# Calcular el accuracy
accuracy <- mean(data.test$CHURN==CLASE.CART) ; accuracy

# Calcular el error de mala clasificaci?n
error <- mean(data.test$CHURN!=CLASE.CART) ; error

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.CART,data.test$CHURN,plotROC = TRUE)
abline(0, 1,col="red")

# Log-Loss
real <- as.numeric(data.test$CHURN)
real <- ifelse(real==2,1,0)
LogLoss(PROBA.CART,real)

# Matriz de confusiÛn
library(caret)
caret::confusionMatrix(CLASE.CART, data.test$CHURN, positive= "Fuga")

#------------------------------------------------------
# PARTE 2: DESPLIEGUE Y PRODUCTIVO DE MODELOS
#------------------------------------------------------


library(foreign)
datosn <- read.spss("Churn-nuevos-arboles.sav",
                    use.value.labels=TRUE, 
                    to.data.frame=TRUE)
str(datosn)

# No considerar la variable de identificaci?n ID
datosn <- datosn[,-1]
str(datosn)

# Etiquetando las opciones de las variables categoricas
levels(datosn$SEXO)  <- c("Fem","Masc")
levels(datosn$CIVIL) <- c("Casado","Soltero")
levels(datosn$AUTO)  <- c("Si","No")
# levels(datosn$CHURN) <- c("Actual","Fuga")
str(datosn)


CLASE.NEW.CART <- predict(modelo_cart,newdata = datosn )
head(CLASE.NEW.CART)

PROBA.NEW.CART <- predict(modelo_cart,newdata = datosn, type="prob")
PROBA.NEW.CART <- PROBA.NEW.CART[,2]

head(PROBA.NEW.CART)

datosn$probabilidad <- PROBA.NEW.CART
datosn$clase <- CLASE.NEW.CART
head(datosn)

write.csv(datosn,"Tarea1.csv")

saveRDS(modelo_cart, "Modelo_arbol_1.rds")
# Ejercicio calificado:

# Dado que tenemos algoritmos entrenados y validados, el area
# comercial de la empresa nos ha solicitado scorear o puntuar 
# la leads que van a gestionarse en las campanas comerciales.

# Como se han enterado que el algoritmo de CART es uno
# de los mas sofisticados ha pedido explicitamente su uso.

# Entregables:
# Base completa de leads nuevos, probabilidad de fuga y la clase 
# pronosticada.

##########################
# 9 SELECCION DE VARIABLES #
##########################

# Algoritmo Boruta
library(Boruta)

set.seed(111)

boruta.data <- Boruta(CHURN ~ ., data = datos)

print(boruta.data)

plot(boruta.data,cex.axis=0.5)


# Las variables tentativas ser?n clasificadas como "confirmed" o "rejected"
# comparando la median Z score de las variables con la median Z score de los
# mejores variables shadow 

final.boruta.data <- TentativeRoughFix(boruta.data)
print(final.boruta.data)

getSelectedAttributes(final.boruta.data, withTentative = F)

boruta.df <- attStats(final.boruta.data)
class(boruta.df)
str(boruta.df)
print(boruta.df)
boruta.df[order(-boruta.df$meanImp),]

####################################################
# 10. ARBOL DE CLASIFICACION CON EL ALGORITMO CHAID #
####################################################
library(partykit)
set.seed(100)

# Entrenamos el arbol chaid sobre train
arbol_chaid <- ctree(CHURN~ .,data = data.train,
                     # Mas estrcito mientrsa mas alta sea la significancia
               control=ctree_control(mincriterion=0.95),
               #maxdepth=3)
)

# Validamos el arbol chaid
proba_arbol=predict(arbol_chaid, newdata= data.test,
               type="response")
proba_arbol

data.test$CHURN
tabla=confusionMatrix(proba_arbol,data.test$CHURN,positive = "Fuga")

windows()
# plot(arbol_chaid,type='simple') 
plot(arbol_chaid,type='extended')

####################################################
# 11. ARBOL DE CLASIFICACION CON EL ALGORITMO C50 #
####################################################
library(C50)
modeloc50 <- C5.0(CHURN~.,data = data.train,
                trials = 50, # Numero de arboles
                rules= TRUE, # Reglas de seleccion
                tree=FALSE,  
                winnow=TRUE) # Seleccion de variables

proba=predict(modeloc50, newdata=data.test,
               type="prob")[,2] # Dame la segunda columna, es decir 
# la probabilidad del default
predC50 =predict(modeloc50, newdata=data.test,
                        type="class")

# Ejercicio hallar la matriz de clasificacion.
matrizC = confusionMatrix(predC50,data.test$CHURN,positive = "Fuga")
matrizC
library(pROC)
auc <- roc()
gini <- 2*(auc$auc)-1
gini 


###########################################################
# 12. ALGORITMO NAIVE BAYES ###############################
###########################################################

library(e1071)
modelonaive=naiveBayes(CHURN~., # Y ~ X
                   data=)

# Predecir para validar el modelo
pred_bayes<- predict(,)

# Lo valido con mi matriz de consusion
tabla=confusionMatrix(pred_bayes,
                      ,positive = "1")


#------------------------------------------------------
# PARTE 2: DESPLIEGUE Y PRODUCTIVO DE MODELOS
#------------------------------------------------------

# Deseamos replicar o implementar el modelo.
# Leemos el modelo predictivo.
RegresionLog <- readRDS("Rg_Logistica.rds")

# Leemos el dataset de nuevos leads
library(foreign)
datos_n <-read.spss("Churn-nuevos-arboles.sav",
                    use.value.labels=TRUE, 
                    to.data.frame=TRUE)

# Decodificacion

levels(datos_n$SEXO)  <- c("Fem","Masc")
levels(datos_n$CIVIL) <- c("Casado","Soltero")
levels(datos_n$AUTO)  <- c("Si","No")


# Scorear o puntuar nuevos registros
prob <- predict(RegresionLog,datos_n,"response")
# Convertir a prediccion
pred <- ifelse(prob<=0.50,"No Fuga","Fuga")

# Lo mandamos a gestionar a distintas areas
dataCampanas <-  data.frame(ID = datos_n$ID,
                            Priorizacion = prob,
                            GestionC = pred)


# Exportar el objeto
write.csv(dataCampanas,"Gestion-Jul19.csv")


# FIN !!



# FIN !!