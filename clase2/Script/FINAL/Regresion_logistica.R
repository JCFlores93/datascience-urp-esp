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

Train <- read.csv("data_loan_status_limpia.csv")
str(Train)
Train$Loan_Status <- as.factor(Train$Loan_Status) # 

## Observar aleatoriamente 3 valores de la data

sample_n(Train, 3)

## supuestos
correlacion <- cor(Train[,2:ncol(Train)],method = "spearman") 
# calculando la correlacion de spearman para las x
write.csv(correlacion,"correla.csv")

#seleccionamos variables con correlacion menor al 0.6
Train <- Train %>% select(
  CoapplicantIncome   ,
  Loan_Amount_Term,
  Gender,
  Dependents,
  Self_Employed,
  Credit_History,
  Total_income,
  Amauntxterm,
  Property_Area,
  Edu_Ma,
  Loan_Status
)

## categorizando las variables categoricas mediante factor
Train$Loan_Status <- as.factor(Train$Loan_Status)
Train$Gender <- as.factor(Train$Gender)
Train$Dependents <- as.factor(Train$Dependents)
Train$Credit_History <- as.factor(Train$Credit_History)
Train$Self_Employed <- as.factor(Train$Self_Employed)
Train$Property_Area <- as.factor(Train$Property_Area)
Train$Edu_Ma <- as.factor(Train$Edu_Ma)

table(Train$Loan_Status)

## Particion Muestral

set.seed(123)
training.samples <- Train$Loan_Status %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- Train[training.samples, ]
test.data <- Train[-training.samples, ]

## Modelado

modelo_logistica=glm(Loan_Status~.,data=train.data,family="binomial" )
summary(modelo_logistica)

## indicadores

proba1 <- predict(modelo_logistica, newdata=test.data,type="response")
AUC1 <- roc(test.data$Loan_Status, proba1)
## calcular el AUC
auc_modelo1=AUC1$auc
## calcular el GINI
gini1 <- 2*(AUC1$auc) -1
# Calcular los valores predichos
table(train.data$Loan_Status)
PRED <-predict(modelo_logistica,test.data,type="response")
# convertimos a clase las probabilidades
PRED <- ifelse(PRED<=0.5,0,1) # pto de corte 0.5
PRED <- as.factor(PRED)
# Calcular la matriz de confusi?n
tabla <- confusionMatrix(PRED,test.data$Loan_Status,positive = "1")
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

## Ejemplo practico de Negocio

library(party)

pdf("recodificacion.pdf")

## CoapplicantIncome
datos.tree<-ctree(Loan_Status ~ CoapplicantIncome
                  ,data=train.data, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train.data)
yprob=sapply(predict(datos.tree, newdata=train.data,type="prob"),'[[',2)
fit.roc1<-roc(train.data$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train.data$Loan_Status==1],
                        yprob[train.data$Loan_Status==0])
plot(datos.tree,main=paste("CoapplicantIncome (GINI:", 
                           round(100*datos.tree.gini,2),"% | KS:", 
                           round(100*datos.tree.ks$statistic,2), "%)",
                           "Casos: ", n1[1],  sep=" "), 
     cex=0.5,type="simple")

## Loan_Amount_Term
datos.tree<-ctree(Loan_Status ~ Loan_Amount_Term
                  ,data=train.data, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train.data)
yprob=sapply(predict(datos.tree, newdata=train.data,type="prob"),'[[',2)
fit.roc1<-roc(train.data$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train.data$Loan_Status==1],
                        yprob[train.data$Loan_Status==0])
plot(datos.tree,main=paste("Loan_Amount_Term (GINI:", 
                           round(100*datos.tree.gini,2),"% | KS:", 
                           round(100*datos.tree.ks$statistic,2), "%)",
                           "Casos: ", n1[1],  sep=" "), 
     cex=0.5,type="simple")

dev.off()

train_data=train.data
library(sqldf)
train_data <- sqldf("
select Credit_History,Property_Area,
case when Edu_Ma in (1,2) then 1 else 2 end Edu_MaF,
Loan_Status
from train_data")

# Categorizando las variables de mi nueva tabla
train_data[,1:ncol(train_data)] <- lapply(train_data[,1:ncol(train_data)],
                                          as.factor)
## Modelado

modelo_logistica_negocio=glm(Loan_Status~.,data=train_data,family="binomial" )
summary(modelo_logistica_negocio)

## indicadores

train_data$proba1 <- predict(modelo_logistica_negocio, 
                             newdata=train_data,type="response")

write.csv(train_data,"modelonegocio.csv",row.names = F)