rm(list=ls())
################################################################################
##### -- Programa de Especialización en Data Science - Nivel I -- #############
################################################################################
############### Tema : Regresión Logística ######################################
######## Autores: Jose Cardenas - Andre Chavez  ################################ 
################################################################################

#### -- 1) Librerias a usar ####

library(dplyr)
library(sqldf)
library(pROC)
library(e1071)
library(caret)

#### -- 2) Modelo de Regresion Logistica

## Cargar la data

Train <- read.csv("PimaIndiansDiabetes.csv")
str(Train)
#Train$diabetes <- as.numeric(Train$diabetes) #
Train$diabetes <- ifelse(Train$diabetes == 'pos', 1, 0)

## Observar aleatoriamente 3 valores de la data
sample_n(Train, 3)

## supuestos
correlacion <- cor(Train[,2:ncol(Train)],method = "spearman") 
# calculando la correlacion de spearman para las x
write.csv(correlacion,"correla_1.csv")

#seleccionamos variables con correlacion menor al 0.6
Train <- Train %>% select(
  glucose,
  pressure,
  mass,
  pedigree,
  age,
  diabetes
)

## categorizando las variables categoricas mediante factor
Train$diabetes <- as.factor(Train$diabetes)

## Particion Muestral
set.seed(123)
training.samples <- Train$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- Train[training.samples, ]
test.data <- Train[-training.samples, ]

## Modelado
modelo_logistica=glm(diabetes~.,data=train.data,family="binomial" )
summary(modelo_logistica)

proba1 <- predict(modelo_logistica, newdata=test.data,type="response")
AUC1 <- roc(test.data$diabetes, proba1)
## calcular el AUC
auc_modelo1=AUC1$auc

## calcular el GINI
gini1 <- 2*(AUC1$auc) -1

# Calcular los valores predichos
table(train.data$diabetes)
PRED <-predict(modelo_logistica,test.data,type="response")
PRED
# convertimos a clase las probabilidades
PRED <- ifelse(PRED<=0.5,0,1) # pto de corte 0.5
PRED <- as.factor(PRED)

# Calcular la matriz de confusi?n
tabla <- confusionMatrix(PRED,test.data$diabetes,positive = "1")

# sensibilidad
Sensitivity1 <- as.numeric(tabla$byClass[1])
#Specificity
Specificity1 <- as.numeric(tabla$byClass[2])
# Precision
Accuracy1 <- tabla$overall[1]
# Calcular el error de mala clasificaci?n
error1 <- mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo1
gini1
Accuracy1
error1
Sensitivity1
Specificity1

## OTRA MANERA
PRED <- predict(modelo_logistica,test.data,type="response")
PRED <- ifelse(PRED<=mean(PRED),0,1) # cambiamos el pto de corte
PRED <- as.factor(PRED)
# Calcular la matriz de confusi?n
tabla <- confusionMatrix(PRED,test.data$Loan_Status,positive = "1")
# sensibilidad
Sensitivity11 <- as.numeric(tabla$byClass[1])
#Specificity
Specificity11 <- as.numeric(tabla$byClass[2])
# Precision
Accuracy11 <- tabla$overall[1]
# Calcular el error de mala clasificaci?n
error11 <- mean(PRED!=test.data$Loan_Status)

# indicadores
Accuracy11
error11
Sensitivity11
Specificity11

#comparando
Sensitivity <- data.frame(Sensitivity1,Sensitivity11)
colnames(Sensitivity)<-c('corte_0.5','corte_prom')
Specificity <- data.frame(Specificity1,Specificity11)
colnames(Specificity)<-c('corte_0.5','corte_prom')

Sensitivity
Specificity