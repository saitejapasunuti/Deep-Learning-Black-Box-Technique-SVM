########################### SUPPORT VECTOR MACHINE ##########################

library(kernlab)
library(caret)
library(plyr)

# Read the data
FF <- read.csv(file.choose())
View(FF)
class(FF)

str(FF)

# The area value has lots of zeros

hist(FF$area)
rug(FF$area)#rug is used to visualize the distribution of the data

# Transform the Area value to Y 
FF1 <- mutate(FF, y = log(area + 1))  # default is to the base e, y is lower case
hist(FF1$y)

summary(FF) # Confirms on the different scale and demands normalizing the data.

# Prediction of Forest fires requires only prediction from 
# temperature, rain, relative humidity and wind speed

# Apply Normalization technique to the whole dataset :

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}

FF$temp = normalize(FF$temp)
FF$RH   = normalize(FF$RH)
FF$wind = normalize(FF$wind)
FF$rain = normalize(FF$rain)
# We need to tweak this as a classification problem.lets base out the Size using this criteria :

attach(FF)

# Data Partition 
set.seed(123)
ind <- sample(2, nrow(FF), replace = TRUE, prob = c(0.7,0.3))
FF_train <- FF[ind==1,]
FF_test  <- FF[ind==2,]

# to train model
# e1071 package from LIBSVM library
# SVMlight algorithm klar package 

# kvsm() function uses gaussian RBF kernel 

# Building model 


model1<-ksvm(size_category~temp+rain+wind+RH, data= FF_train,kernel = "vanilladot")
model1
#Training error : 0.247978 

Area_pred <- predict(model1, FF_test)

table(Area_pred,FF_test$size_category)
#Area_pred large small
#large     0     0
#small    47    99

agreement <- Area_pred == FF_test$size_category
table(agreement)
#agreement
#FALSE  TRUE 
#47    99 

prop.table(table(agreement))
#agreement
#FALSE      TRUE 
#0.3219178 0.6780822 

# Different types of kernels 
# "rbfdot", "polydot", "tanhdot", "vanilladot", "laplacedot", 
# "besseldot", "anovadot", "splinedot", "matrix"


################## kernel = rfdot #######################

model_rfdot<-ksvm(size_category~temp+rain+wind+RH,data= FF_train,kernel = "rbfdot")
pred_rfdot<-predict(model_rfdot,newdata=FF_test)

mean(pred_rfdot==FF_test$size_category) # 68.41
#Accuracy=0.6849315


################kernel = vanilladot######################

model_vanilla<-ksvm(size_category~temp+rain+wind+RH, data= FF_train,kernel = "vanilladot")

pred_vanilla<-predict(model_vanilla,newdata=FF_test)

mean(pred_vanilla==FF_test$size_category) # 67.80
##Accuracy=0.6780822


################kernal = besseldot #################
model_besseldot<-ksvm(size_category~temp+rain+wind+RH, data= FF_train,kernel = "besseldot")

pred_bessel<-predict(model_besseldot,newdata=FF_test)
mean(pred_bessel==FF_test$size_category) # 67.80
#Accuracy=0.6780822

###############kernel = polydot#######################

model_poly<-ksvm(size_category~temp+rain+wind+RH, data= FF_train,kernel = "polydot")
pred_poly<-predict(model_poly,newdata = FF_test)
mean(pred_poly==FF_test$size_category) # 67.80
##Accuracy=0.6780822


