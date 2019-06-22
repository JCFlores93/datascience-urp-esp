rm(list = ls()) # Borra lo guardadoen memoria

#cargar una base de datos
data <- read.csv('grasacorporal.csv')
names(data) #nombres de las columnas de la data
View(data)

str(data) # estructura de la data

#factor -> variables cualitativas, si es texto se muestra como character
#num -> variables cuantitativas - continua
#int -> variables cuantitativas - discreta

tabla_resumen <- summary(data) # tabla resumen
write.csv(tabla_resumen, "datosResumen.csv")
# Si no existe la variable seleccionada, la agrega.
pecho <- data$pecho
View(pecho)

# Creacion de variable
data$var1 <- data$antebrazo + data$muneca
View(data)
head(data)

# filas y columnas
data[1:3,]
data[1:5,2]
# Seleccion puntual de registros
data[c(1, 3), 2]

#usar librería
library(neuralnet)