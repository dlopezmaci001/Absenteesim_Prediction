########## Baseline your work ###################

library(stargazer)
library(caret)
library(glment)
library(mlbench)
library(psych)
library(nnet)
library(ROCR)
library(pROC)
library(PresenceAbsence)

##### Construction of target variable:

setwd("C:/Users/Daniel/Desktop/Machine L II/Individual_ass")

raw <- read.csv2("Absenteeism_at_work.csv",header =  TRUE, sep = ";")

# QC CHeck
head(raw)
dim(raw)

# Target variable

# I've just loaded the data set and built a new feature that considering the number
# of hours of absenteeism, produces a boolean (binomial) variable.
# in order to consider what is absenteeism, without nowing anything else from the data
# we would say absenteeism is considered when the total nº of hours is above the mean

mean(raw$Absenteeism.time.in.hours)

# This value is close to 7, therefore:

# If the number of hours is < 7 this variable (Absent) is False, and
# True if it is >= 7.

raw$Absentist[raw$Absenteeism.time.in.hours>7] <- 1 # TRUE 

raw$Absentist[raw$Absenteeism.time.in.hours<=7] <- 0 # False

####### Model 1 ###############


# Data partition

set.seed(3457)

ind <- sample(2,nrow(raw),replace=T,prob = c(0.7,0.3))
train <- raw[ind==1,]
test <- raw[ind==2,]

# Custom control parameters

custom <- trainControl(method="repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

# Linear model
set.seed(3457)
lm <- train(Absentist ~.,
            train,
            method='lm',
            trControl=custom)

# Results

lm$results
lm
summary(lm)
#plot(lm$finalModel)

# Confusion Matrix Plot

threshold = 0.5
trainColNum <- grep("Absentist",names(train))
predictionProb <- predict(lm,newdata=test,type='raw')
predicted <- ifelse(predictionProb>threshold,1,0)
actual <- ifelse(test[,trainColNum],1,0)
actual <- factor(actual, 
                 levels=c(0, 1),
                 labels=c(0, 1))

predicted <- factor(predicted, 
                    levels=c(0, 1),
                    labels=c(0, 1))

cf<- confusionMatrix(actual,predicted)
cf

# Accuracy 0.78
# False Pos 0.71

####################### Feature Engineering ######################

# From the results obtained it is clear that we need to change some variables,
# which are stored as factors and should be stored in int.

# We need to transform some factor variables into intigers

as.numeric(raw$Transportation.expense)
as.numeric(raw$Distance.from.Residence.to.Work)
raw$Work.load.Average.day <- as.numeric(as.character(raw$Work.load.Average.day))

# Taking a closer look at the data set, we find the variables, weight,height and body.mass.index
# it is expected that body mass index has a high correlation to both of these variables since 
# the body mass index is computed using both values.
# Let's define:

# H0: There is no linear relationship between body mass index and height/weight
# H1: There is linear relationship between body mass index and height/weight

cor.test(raw$Body.mass.index,raw$Weight)

cor.test(raw$Body.mass.index,raw$Height)

# As in both cases p-value < 0.05 we reject the null hypothesis.
# As we can see both p-values < 0.05 
# Now let's take away some variables we don't need

raw_prep1 <- subset(raw,select=-c(ID,Absenteeism.time.in.hours,Weight,Height))

# Let's normalise the variables so they do not affect the importance of the rest.

columnsToNormalise <- c("Transportation.expense", "Distance.from.Residence.to.Work", 
                        "Service.time", "Age", "Work.load.Average.day", 
                        "Hit.target", "Body.mass.index","Reason.for.absence",
                       "Month.of.absence","Day.of.the.week","Season","Son","Pet"
                       )

subsetToNormalise <- raw_prep1[,which(names(raw_prep1) %in% columnsToNormalise)]
subsetToKeep <- raw_prep1 [,-which(names(raw_prep1) %in% columnsToNormalise)]

# Apply the function in caret to normalize it

preObj <- preProcess(subsetToNormalise, method=c("center","scale"))
normalisedSubset <- predict(preObj, subsetToNormalise)
raw_prep1 <- cbind(subsetToKeep,normalisedSubset)

# Let's see how the model perfomance works

##### Model 2 ####

# Data partition

set.seed(3457)

ind_1 <- sample(2,nrow(raw_prep1),replace=T,prob = c(0.7,0.3))
train_1 <- raw_prep1[ind_1==1,]
test_1 <- raw_prep1[ind_1==2,]

# Custom control parameters

custom_1 <- trainControl(method="repeatedcv",
                         number = 10,
                         repeats = 5,
                         verboseIter = T)

# Linear model
set.seed(3457)
lm_1 <- train(Absentist ~.,
              train_1,
              method='lm',
              trControl=custom_1)

# Results

lm_1$results
lm_1
summary(lm_1)
#plot(lm_1$finalModel)


# # Confusion Matrix Plot

threshold = 0.5
trainColNum_lm_1 <- grep("Absentist",names(train_1))
predictionProb_lm_1 <- predict(lm_1,newdata=test_1,type='raw')
predicted_lm_1 <- ifelse(predictionProb_lm_1>threshold,1,0)
actual_lm_1 <- ifelse(test_1[,trainColNum_lm_1],1,0)
actual_lm_1 <- factor(actual_lm_1, 
                      levels=c(0, 1),
                      labels=c(0, 1))

predicted_lm_1 <- factor(predicted_lm_1, 
                         levels=c(0, 1),
                         labels=c(0, 1))

cf_lm_1<- confusionMatrix(actual_lm_1,predicted_lm_1)
cf_lm_1

# Accuracy 0.73
# False pos 0.67

##### Backward model selection ####


lm_1_1 <- lm(Absentist ~., data=raw_prep1)

summary(lm_1_1)

step(lm_1_1,direction = "backward")

# Based on backwards selection the variables used are:

# Disciplinary.failure + Social.drinker + 
# Social.smoker + Reason.for.absence + Day.of.the.week + Transportation.expense + 
# Distance.from.Residence.to.Work + Age + Son + Body.mass.index

# Now let's include the variables we need

##### Model 3 #### 

raw_prep2 <- subset(raw_prep1,select=c(Disciplinary.failure, Social.drinker, 
                                         Social.smoker, Reason.for.absence, Day.of.the.week,
                                         Transportation.expense,
                                         Distance.from.Residence.to.Work, Age, Son,
                                         Body.mass.index, Absentist))

# Let's see how the model perfomance works

# Data partition

set.seed(3457)

ind <- sample(2,nrow(raw_prep2),replace=T,prob = c(0.7,0.3))
train_2 <- raw_prep2[ind==1,]
test_2 <- raw_prep2[ind==2,]

# Custom control parameters

custom <- trainControl(method="repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

# Linear model
set.seed(3457)
lm_2 <- train(Absentist ~.,
              train_2,
              method='glm',
              trControl=custom)

# Results

lm_2$results
lm_2
summary(glm_2)
#plot(lm_2$finalModel)

# COnfusion Matrix

trainColNum_lm_2 <- grep("Absentist",names(train_2))
predictionProb_lm_2 <- predict(lm_2,newdata=test_2,type='raw')
predicted_lm_2 <- ifelse(predictionProb_lm_2>threshold,1,0)
actual_lm_2 <- ifelse(test_2[,trainColNum_lm_2],1,0)

#### Selecting best threshold ####

my_roc <- roc(actual_lm_2,predicted_lm_2)
coords <- coords(my_roc,"best",ret="threshold")
coords

# Confusion matrix optimum threshold

threshold <- coords
trainColNum_lm_2 <- grep("Absentist",names(train_2))
predictionProb_lm_2 <- predict(lm_2,newdata=test_2,type='raw')
predicted_lm_2 <- ifelse(predictionProb_lm_2>threshold,1,0)
actual_lm_2 <- ifelse(test_2[,trainColNum_lm_2],1,0)
actual_lm_2 <- factor(actual_lm_2, 
                      levels=c(0, 1),
                      labels=c(0, 1))

predicted_lm_2 <- factor(predicted_lm_2, 
                         levels=c(0, 1),
                         labels=c(0, 1))

cf_lm_2<- confusionMatrix(actual_lm_2,predicted_lm_2)
cf_lm_2


# Accuracy 0.72
# False pos 0.66

##### Using other methods like ridge lasso knn ####

##### Model 4 - Risge regression ####

set.seed(3457)

ridge_2 <- train(Absentist ~.,
                 train_2,
                 method='glmnet',
                 tuneGrid=expand.grid(alpha=0,lambda=seq(0.0001,1,length=5)),
                 trControl=custom)

# Lambda = 1e-04

# Plot results
plot(ridge_2)
ridge_2
plot(ridge_2$finalModel,xvar="lambda",label=T)
plot(ridge_2$finalModel,xvar="dev",label=T)
plot(varImp(ridge_2,scale=T))


# COnfusion Matrix

trainColNum_ridge_2 <- grep("Absentist",names(train_2))
predictionProb_ridge_2 <- predict(ridge_2,newdata=test_2,type='raw')
predicted_ridge_2 <- ifelse(predictionProb_ridge_2>threshold,1,0)
actual_ridge_2 <- ifelse(test_2[,trainColNum_ridge_2],1,0)


# Confmatrix optimal threshold

my_roc_ridge <- roc(actual_ridge_2,predicted_ridge_2)
coords_ridge <- coords(my_roc_ridge,"best",ret="threshold")
coords_ridge

threshold <- coords_ridge 
trainColNum_ridge_2 <- grep("Absentist",names(train_2))
predictionProb_ridge_2 <- predict(ridge_2,newdata=test_2,type='raw')
predicted_ridge_2 <- ifelse(predictionProb_ridge_2>threshold,1,0)
actual_ridge_2 <- ifelse(test_2[,trainColNum_ridge_2],1,0)

actual_ridge_2 <- factor(actual_ridge_2, 
                         levels=c(0, 1),
                         labels=c(0, 1))

predicted_ridge_2 <- factor(predicted_ridge_2, 
                            levels=c(0, 1),
                            labels=c(0, 1))

cf_ridge_2<- confusionMatrix(actual_ridge_2,predicted_ridge_2)
cf_ridge_2

# Accuracy 0.71
# False Pos 0.65

##### Model 5 - Lasso ####

# Lasso Regression
set.seed(3457)
lasso_2 <- train(Absentist ~.,
                 train_2,
                 method='glmnet',
                 tuneGrid=expand.grid(alpha=1,lambda=seq(0.0001,0.2,length=5)),
                 trControl=custom)

# Plot results
plot(lasso_2)
lasso_2
plot(lasso_2$finalModel,xvar="lambda",label=T)
plot(lasso_2$finalModel,xvar="dev",label=T)
plot(varImp(lasso_2,scale=F))


# COnfusion Matrix

trainColNum_lasso_2 <- grep("Absentist",names(train_2))
predictionProb_lasso_2 <- predict(lasso_2,newdata=test_2,type='raw')
predicted_lasso_2 <- ifelse(predictionProb_lasso_2>threshold,1,0)
actual_lasso_2 <- ifelse(test_2[,trainColNum_lasso_2],1,0)

# Confmatrix optimal threshold

my_roc_lasso <- roc(actual_lasso_2,predicted_lasso_2)
coords_lasso <- coords(my_roc_lasso,"best",ret="threshold")
coords_lasso

threshold <- coords_lasso
trainColNum_lasso_2 <- grep("Absentist",names(train_2))
predictionProb_lasso_2 <- predict(lasso_2,newdata=test_2,type='raw')
predicted_lasso_2 <- ifelse(predictionProb_lasso_2>threshold,1,0)
actual_lasso_2 <- ifelse(test_2[,trainColNum_lasso_2],1,0)
actual_lasso_2 <- factor(actual_lasso_2, 
                         levels=c(0, 1),
                         labels=c(0, 1))

predicted_lasso_2 <- factor(predicted_lasso_2, 
                            levels=c(0, 1),
                            labels=c(0, 1))

cf_lasso_2<- confusionMatrix(actual_lasso_2,predicted_lasso_2)
cf_lasso_2

# Accuracy 0.72
# False pos 0.66

##### Model 6 - KNN ####

set.seed(3457)

library(class)

k <- sqrt(nrow(raw_prep2))
k <- round(k,0)
k


knn1 <- knn(train=train_2,test= test_2, cl=train_2$Absentist,k=k)
  
cf_knn <- table(test_2$Absentist,knn1)
cf_knn

sum(diag(cf_knn))/sum(cf_knn)

# 81.5 % accuracy 
# False Pos 0.92


