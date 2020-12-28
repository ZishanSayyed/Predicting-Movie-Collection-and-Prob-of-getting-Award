  #Decision Tree (Regression Trees)

movie=read.csv("C:/Users/ADMIN/OneDrive/Documents/ML/Decison tree/Movie_regression.csv")
View(movie)

#explanatory data analysis
summary(movie)

#from the given details we can see that 
#there is outliers in Marketing.expense and Collection 
#and there are some missing values in Time_taken 

#we removing missing values by its mean 

movie$Time_taken[is.na(movie$Time_taken)]=mean(movie$Time_taken, na.rm = TRUE)
summary(movie$Time_taken)

#dividing data into test and train 
library(caTools)
set.seed(0)
split=sample.split(movie,SplitRatio = 0.8)
train_data=subset(movie,split==TRUE)
test_data=subset(movie,split==FALSE)


#Library for decision tree 

library(rpart)
library(rpart.plot)

#Run regression tree model on train set
model=rpart(formula = Collection~.,data = train_data,control = rpart.control(maxdepth = 3))

#Plot the decision Tree
rpart.plot(model,box.palette = "RdBu",digits = -3)

#Predict value at any point
test_data$pred=predict(model,test_data,type = "vector") #since data is numeric
MSE2 <- mean((test_data$pred - test_data$Collection)^2)

#to avoid over fitting and complex interpretation we use technique of purning 
#Tree Pruning

fulltree=rpart(formula = Collection~., data = train_data,control = rpart.control(cp=0))
rpart.plot(fulltree,box.palette = "RdBu",digits = -3)
printcp(fulltree)  #cp is control parameter or tunning parameter 
plotcp(model)

#we are finding that value of cp for which xeror is min
mincp=model$cptable[which.min(model$cptable[,"xerror"]),"CP"]
punedtree=prune(fulltree,cp=mincp)
rpart.plot(punedtree,box.palette = "RdBu",digits = -3)


test_data$fulltree <- predict(fulltree, test_data, type = "vector")
MSE2full <- mean((test_data$fulltree - test_data$Collection)^2)

test_data$pruned <- predict(punedtree, test_data, type = "vector")
MSE2pruned <- mean((test_data$pruned - test_data$Collection)^2)


#Bagging technique to improve performance of DT
library(randomForest)
set.seed(0)
bagging=randomForest(Collection~., data = train_data,mtry=17) #mtry is no of dependent variable 
test_data$bagging=predict(bagging,test_data)
MSE2bagging <- mean((test_data$bagging - test_data$Collection)^2)
#in bagging train data set are different but we include all the dependent variable 

### Random Forest technique to improve performance of DT
library(randomForest)
set.seed(0)
randomfor=randomForest(Collection~., data = train_data,ntree=500)
test_data$randomfor=predict(randomfor,test_data)
MSE2randomfor <- mean((test_data$randomfor - test_data$Collection)^2)
#in random forest train data set are different but we include some of the dependent variable 



#Boosting 
library(gbm)
set.seed(0)
boosting=gbm(Collection~.,data=train_data,distribution = "gaussian",
             n.trees = 5000,interaction.depth = 4,shrinkage =0.2,verbose = F )
test_data$boost=predict(boosting,test_data,n.trees = 5000)
MSE2boost <- mean((test_data$boost - test_data$Collection)^2)


################  Support Vector Machine Model (SVM) For regression      ################
library(e1071)


#Linear Kernel 
svmfit_reg=svm(Collection~.,data = train_data,kernel="linear",cost=0.01,scale = TRUE)
summary(svmfit_reg)

#prediction 
test_data$ypredict_reg=predict(svmfit_reg,test_data)
MSE2svmL=mean((test_data$ypred - test_data$Collection)^2)

#tunning cost values
set.seed(0)
tune_out=tune(svm,Collection~.,data = train_data,kernel="linear",ranges = list(cost=c(0.01,0.1,1,1.1,10,100)))
bestmodel=tune_out$best.model
summary(bestmodel)

#prediction 
test_data$ypredict_svmL=predict(bestmodel,test_data)
MSE2svmL_bestmodel=mean((test_data$ypredict_svmL - test_data$Collection)^2)



#Polynomial Kernel 

svmfit_P=svm(Collection~.,data = train_data,kernel="polynomial",cost=1,degree=2)
summary(svmfit_P)

#hyper tunning 
tune_outP=tune(svm,Collection~.,data = train_data,cross=4,kernel="polynomial",
               ranges = list(cost=c(0.01,0.1,1,1.1,10,100)),degree=c(0.5,1,2,3,4,5))
bestmodel_P=tune_outP$best.model
summary(bestmodel_P)
#prediction 
test_data$ypredict_svmP=predict(bestmodel_P,test_data)
MSE2svmL_bestmodelPoly=mean((test_data$ypredict_svmP - test_data$Collection)^2)


#Redial Kernel 

svmfit_R=svm(Collection~.,data = train_data,kernel="radial",gamma=1,cost=1)
summary(svmfit_R)

set.seed(0)
tune_outR=tune(svm,Collection~.,data = train_data,kernel="radial",
               ranges = list(cost=c(0.01,0.1,1,1.1,10,100)),gamma=c(0.01,0.1,0.5,1,2,3,10),cross=4)
bestmodelR=tune_outR$best.model
summary(bestmodelR)
test_data$ypredict_svmR=predict(bestmodelR,test_data)
MSE2svmL_bestmodelRed=mean((test_data$ypredict_svmR - test_data$Collection)^2)


#finding best model among all

min(MSE2,MSE2bagging,MSE2boost,MSE2full,MSE2pruned,MSE2randomfor,MSE2svmL,
    MSE2svmL_bestmodel,MSE2svmL_bestmodelPoly,MSE2svmL_bestmodelRed)


# 43485582 is MSE among all and technique is Random Forest 