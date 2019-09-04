# Code for the project presentation
install.packages('e1071')

# Importing the libraries that are required
library(caret)
library(ggplot2)
library(grid)
library(gridExtra)
library(corrplot)
library(psych)

library(plyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(survival)
library(pROC)
library(DMwR)
library(scales)

# Getting the dataframe from the csv file
HRdata <- read.csv("D:/Downloads/SUNY Buffalo Semester 1/Statistical Data Mining 1/Final Project/HRData.csv")
names(HRdata)

# Split data to train and test
set.seed(12345)
inTrain <- createDataPartition(HRdata$Attrition,p=0.75,list = FALSE)
Training <- HRdata[inTrain,]
Testing <- HRdata[-inTrain,]

# Calculating the percentage of attrition
ggplot(Training,aes(Attrition,fill=Attrition))+geom_bar()
prop.table(table(Training$Attrition)) #Percentage of Attrition

# Trying out a plot
jobLevel <- ggplot(Training,aes(JobLevel,fill=Attrition))+geom_bar()
newManager <- ggplot(Training,aes(YearsWithCurrManager,fill = Attrition))+geom_bar()
jobSatisfaction <- ggplot(Training,aes(JobSatisfaction,fill=Attrition))+geom_bar()
distFromWork <- ggplot(Training,aes(DistanceFromHome,fill=Attrition))+geom_bar()
# grid.arrange(jobLevel,newManager,jobSatisfaction,distFromWork,ncol=2,top = "Some of the factors that influence attrition")
grid.arrange(jobLevel,newManager,jobSatisfaction,distFromWork,top = "Some of the factors that influence attrition")

# Feature engineering
Training_os <- Training
Training_os$TenurePerJob <- ifelse(Training_os$NumCompaniesWorked!=0, Training_os$TotalWorkingYears/Training_os$NumCompaniesWorked,0)
Training_os$YearsWithoutChange <- Training_os$YearsInCurrentRole - Training_os$YearsSinceLastPromotion

# Adding these features in test data
Testing$TenurePerJob <- ifelse(Testing$NumCompaniesWorked!=0, Testing$TotalWorkingYears/Testing$NumCompaniesWorked,0)
Testing$YearsWithoutChange <- Testing$YearsInCurrentRole - Testing$YearsSinceLastPromotion

# Converting non-numeric variables to numberic
Train <- Training
Train$BusinessTravel <- as.integer(Train$BusinessTravel)
Train$Department <- as.integer(Train$Department)
Train$Gender <- as.integer(Train$Gender)
Train$MaritalStatus <- as.integer(Train$MaritalStatus)
Train$OverTime <- as.integer(Train$OverTime)
Train$JobRole <- as.integer(Train$JobRole)
Train$EducationField <- as.integer(Train$EducationField)
Train$Over18 <- as.integer(Train$Over18)
# Train$Attrition <- as.integer(Train$Attrition)

Test <- Testing
Test$BusinessTravel <- as.integer(Test$BusinessTravel)
Test$Department <- as.integer(Test$Department)
Test$Gender <- as.integer(Test$Gender)
Test$MaritalStatus <- as.integer(Test$MaritalStatus)
Test$OverTime <- as.integer(Test$OverTime)
Test$JobRole <- as.integer(Test$JobRole)
Test$EducationField <- as.integer(Test$EducationField)
Test$Over18 <- as.integer(Test$Over18)
# Test$Attrition <- as.integer(Test$Attrition)

# Fitting different models
set.seed(12345)
fit_rpart <- train(Attrition ~.,Train,method = 'rpart', trControl = trainControl(method = 'cv',number = 3))

set.seed(12345)
fit_rf <- train(Attrition ~.,Train,method = 'rf', trControl = trainControl(method = 'repeatedcv',number = 3))

set.seed(12345)
xgbGrid <- expand.grid(nrounds = 300,
                       max_depth = 1,
                       eta = 0.3,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9)

set.seed(12345)
fit_xgb <- train(Attrition ~.,Train,method = 'xgbTree',tuneGrid = xgbGrid,trControl = trainControl(method = 'repeatedcv',number = 3,classProbs = TRUE)) 

set.seed(12345)
fit_nn <- train(Attrition ~.,Train,method = 'pcaNNet',trControl = trainControl(method = 'repeatedcv',number = 3),tuneGrid = expand.grid(size = 25,decay = 0.01))

set.seed(12345)
fit_glm <- train(Attrition~.,Train,method = 'glm',trControl = trainControl(method = 'repeatedcv',number = 3))

set.seed(12345)
fit_svm <- train(Attrition~.,Train,method = 'svmRadial',trControl = trainControl(method = 'repeatedcv',number = 3))

set.seed(12345)
fit_knn <- train(Attrition~.,Train,method = 'knn',trControl = trainControl(method = 'repeatedcv',number = 3))

set.seed(12345)
fit_glmBoost <- train(Attrition~.,Train,method = 'glmboost',trControl = trainControl(method = 'repeatedcv',number = 3))

set.seed(12345)
Predictions_rpart <- predict(fit_rpart,Test)
Predictions_rf <- predict(fit_rf, Test)
Predictions_xgb <- predict(fit_xgb, Test)
Predictions_nn <- predict(fit_nn, Test)
Predictions_glm <- predict(fit_glm, Test)
Predictions_svm <- predict(fit_svm,Test)
Predictions_knn <- predict(fit_knn,Test)
Predictions_glmboost <- predict(fit_glmBoost,Test)