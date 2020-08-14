########################### SUPPORT VECTOR MACHINE ##########################

library(kernlab)
library(caret)
library(plyr)
library(e1071)

# Data(Train)
train_sal <- read.csv(file.choose())
str(train_sal)

View(train_sal)

train_sal$educationno <- as.factor(train_sal$educationno)

class(train_sal)

# Data(Test)
test_sal <- read.csv(file.choose())
str(test_sal)

View(test_sal)

test_sal$educationno <- as.factor(test_sal$educationno)
class(test_sal)


# Building model 
model1<-ksvm(train_sal$Salary~.,data= train_sal, kernel = "vanilladot")
model1

#Evaluating model
Salary_prediction <- predict(model1, test_sal)

table(Salary_prediction,test_sal$Salary)

agreement <- Salary_prediction == test_sal$Salary
table(agreement)
#agreement
#FALSE  TRUE 
#2313 12747 

prop.table(table(agreement))
#agreement
#FALSE      TRUE 
#0.1535857 0.8464143

############ kernel = rfdot #########

model_rfdot<-ksvm(train_sal$Salary~., data= train_sal,kernel = "rbfdot")

pred_rfdot<-predict(model_rfdot,newdata=test_sal)
mean(pred_rfdot==test_sal$Salary) # 85.20
#Accuracy=0.8520584

############ kernel = vanilladot############
model_vanilla<-ksvm(train_sal$Salary~.,data= train_sal,kernel = "vanilladot")
pred_vanilla<-predict(model_vanilla,newdata=test_sal)

mean(pred_vanilla==test_sal$Salary) # 84.64
#Accuracy=0.8464143

################kernal = besseldot #################
model_besseldot<-ksvm(train_sal$Salary~., data= train_sal,kernel = "besseldot")

pred_bessel<-predict(model_besseldot,newdata=test_sal)
mean(pred_bessel==test_sal$Salary) # 78.97

#Accuracy=0.7897078

###############kernel = polydot#######################

model_poly<-ksvm(train_sal$Salary~., data= train_sal,kernel = "polydot")
pred_poly<-predict(model_poly,newdata = test_sal)
mean(pred_poly==test_sal$Salary) # 84.61

##Accuracy=0.8461487