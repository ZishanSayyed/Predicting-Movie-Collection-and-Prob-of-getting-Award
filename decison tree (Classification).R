################################# Classification Decision Tree ###############################################
movie2=read.csv("C:/Users/zishan.sayyed/OneDrive - Interpublic/Documents/GitHub/Predicting-Movie-Collection-and-Prob-of-getting-Award/Movie_classification.csv")
View(movie2)

summary(movie2)


movie2$Time_taken[is.na(movie2$Time_taken)]=mean(movie2$Time_taken,na.rm = TRUE)
summary(movie2$Time_taken)

library(caTools)
set.seed(0)
split=sample.split(movie2,SplitRatio = 0.8)
train_C=subset(movie2,split==TRUE)
test_C=subset(movie2,split==FALSE)

library(rpart)
library(rpart.plot)
classTree=rpart(formula =Start_Tech_Oscar~. , data = train_C,method = "class" , control = rpart.control(maxdepth = 3))
rpart.plot(classTree,box.palette = "RdBu",digits = -3)

test_C$pred=predict(classTree,test_C,type="class")
table(test_C$Start_Tech_Oscar,test_C$pred)
(45+19)/107   #59% accuracy 




#Adaboost
library(adabag)
adaboost=boosting(Start_Tech_Oscar~.,data=train_C,boos=TRUE)
pred_ada=predict(adaboost,test_C)
table(pred_ada$class,test_C$Start_Tech_Oscar)
(28+41)/107  #64% accuracy 


adaboost=boosting(Start_Tech_Oscar~.,data=train_C,boos=TRUE,mfinal=1000)
pred_ada=predict(adaboost,test_C)
table(pred_ada$class,test_C$Start_Tech_Oscar)
(31+41)/107    #67% accuracy 

t1=adaboost$trees[[1]]
plot(t1)
text(t1,pretty = 100)


#XGBoosting 
library(xgboost)
trainY=train_C$Start_Tech_Oscar=="1"
trainX=model.matrix(Start_Tech_Oscar~.-1,data=train_C) #creating dummy variables for categorical classes 
trainX=trainX[,-12]


testY=test_C$Start_Tech_Oscar=="1"
testX=model.matrix(Start_Tech_Oscar~.-1,data=test_C)
testX=testX[,-12]

Xmatrix=xgb.DMatrix(data = trainX,label=trainY)
Xmatrix_t=xgb.DMatrix(data = testX,label=testY)


XGBoosting=xgboost(data=Xmatrix,
                   nrounds = 50,objective="multi:softmax",
                   eta=0.3,num_class=2,max_depth=100)
xgpred=predict(XGBoosting,Xmatrix_t)
table(xgpred,test_C$Start_Tech_Oscar)
(27+41)/107   #63% accuracy 


                     #Support Vector Machine(SVM) Model in R
library(e1071)

train_C$Start_Tech_Oscar=as.factor(train_C$Start_Tech_Oscar)

#Linear kernel 
svmfit=svm(Start_Tech_Oscar~.,data = train_C,kernel="linear",cost=1,scale =TRUE)
#cost =1 giving us cost of missing classification 
summary(svmfit)

#prediction  
ypred=predict(svmfit,test_C)
table(ypred,test_C$Start_Tech_Oscar)
(30+36)/107   #61% accuracy 

#finding best cost value (tuneing value)
set.seed(0)
tune_out=tune(svm,Start_Tech_Oscar~.,data=train_C,kernel="linear",ranges = list(cost=c(0.01,0.1,1,1.1,10,100)))
bestmodel=tune_out$best.model
summary(bestmodel)

ypredL=predict(bestmodel,test_C)
table(ypredL,test_C$Start_Tech_Oscar)
(30+36)/107 #61% accuracy 



#Polynomial Kernel
svmfit_P=svm(Start_Tech_Oscar~.,data = train_C,kernel="linear",cost=1,degree=2)
summary(svmfit_P)

#hyper tunning 
tune_outP=tune(svm,Start_Tech_Oscar~.,data=train_C,cross=4,kernel="polynomial",
               ranges = list(cost=c(0.01,0.1,1,1.1,10,100)),degree=c(0.5,1,2,3,4,5))
bestmodel_P=tune_outP$best.model
summary(bestmodel_P)


ypredP=predict(bestmodel_P,test_C)
length(ypredP)
table(ypredP,test_C$Start_Tech_Oscar)
57/107   #acc is 53%


#Redial  Kernel
svmfit_R=svm(Start_Tech_Oscar~.,data = train_C,kernel="radial",gamma=1,cost=1)
summary(svmfit_R)

set.seed(0)
tune_outR=tune(svm,Start_Tech_Oscar~.,data=train_C,kernel="linear",
               ranges = list(cost=c(0.01,0.1,1,1.1,10,100)),gamma=c(0.01,0.1,0.5,1,2,3,10),cross=4)
bestmodelR=tune_outR$best.model
summary(bestmodelR)
ypredR=predict(bestmodelR,test_C)
table(ypredR,test_C$Start_Tech_Oscar)
(31+36)/107  #62% acc


#conclusion = we reach max acc of 67% by adabosting technique 
