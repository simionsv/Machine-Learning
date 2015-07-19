

<center>*Project Assignment*</center>

<center>***Title: Predicting the Manner 6 Volunteers perfoemd Weight Lifting Activity ***</center>


<center>***Sergio Vicente Simioni***</center>

<center>***July, 18, 2015***</center>

<span style ="color:blue">**Report Content**</span>

a. <span style ="color:blue">**Executive Summary**</span>
b. <span style ="color:blue">**Conclusions/Questions addressing**</span>
c. <span style ="color:blue">**Data Analysis**</span>


 <span style ="color:blue">**Content Description**</span>

 <span style ="color:blue">**Executive Summary**</span>
Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The data was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways ( "classe" ):  
                A - Regular  
                B - Throwing the elbows to the front   
                C - Lifting the dumbbell only halfway    
                D - Lowering the dumbbell only halfway  
                E - Throwing the hips to the front 

The goal of this project is to predict the manner in which the participants did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

The training and test data for this project are available on: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv



<span style ="color:blue">**Questions addressing**</span>

HOW WAS BUILT THE MODEL ( Chosen Regressions method ) ?

        I chose the two most common method to make this analysis: The CART (Classification and regression Tree) and The Random Forest. The CART method is the standard machine learning technique in ensemble terms, corresponds to a weak learner. In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets.The Random Forest (Breiman, 2001) is an ensemble approach which is a collection of "weak learners" that can come together to form a "strong learner".  
        - CARET method is easy to interpret, fast to run but hard to estimate the uncertainty    
        - RANDOM FOREST has a very good accuracy, but difficult to interpret and low speed  to run (may take a couple of hours)

HOW WAS CHOSEN THE CROSS VALIDATION?

        There are several methods to estimate the model accuracy among them, Data Split, Boostrap, K-Fold Cross validation, Repeated K-Fold Cross validation and Leave One Out Cross validation. I choose the DATA SPLIT due to the very large amount of data of thsi problem.   
        - Data Splitting: Involves partitioning the data into an explicit Training dataset used to prepare the model, (typically 70% ) and an unseen Test dataset ( typically 30% )used to evaluate the models performance on unseen data. It is useful for a very large dataset so that the test dataset can provide a meaningful estimation of the performance or for when you are using slow methods and need a quick approximation of performance     
        - K-Fold Cross Validation; Involves splitting the dataset into K-subsets. For each subset is held out while the model is trained on all other subsets. This process is completed until accuracy is determine for each instance in the dataset, and an overall accuracy estimate is provided.     
        - Leave One Out Cross Validation: a data instance is left out and a model constructed on all other data instances in the training set . This is repeated for all data instances.


WHAT YOU THINK THE EXPECTED OUT OF SAMPLE ERROR IS ( = 1 - ACCURACY )?

        The two methods chosen provided diferente accuracy. the CART Method provided an accuracy of 60,65%, consequently the expected out of the sample error 1- 60,65% = 39,35% and on the other hand the RANDOM FOREST method provided a much better accuracy 99,32%, giving an expected out of sample error of 1 - 99,32% = 0.68%.


USE THE FINAL PREDICT MODEL TO PREDICT 20 DIFFERENT TEST CASES.   
        See TABLE A, in the last session



<span style ="color:blue">**Data Analysis**</span>

<span style ="color:blue">**Getting and Cleaning Data **</span>

*Loading the libraries necessary to run the codes*
```{r,message=FALSE, warning=FALSE}
library(dplyr)
library(caret)
library(rattle)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
```

*Loading the files necessary to develop the project*
```{r}
training_original <- read.csv("J:/pml-training.csv", sep=";", stringsAsFactor = FALSE,na.strings=c("#DIV/0!"))
testing_original  <- read.csv("J:/pml-testing.csv",  sep=";", stringsAsFactor = FALSE,na.strings=c("#DIV/0!"))

```

```{r}
str(training_original)
```

*The project has the following dimensions ( number of rows versus number of columns)*
```{r}
dim(training_original)
```


*In order to manipulate the data to perform the regression and correlation analysis, the columns of the project should be converted to numeric, the conversion of the 160 columns was done using the "FOR" loop excluding only the column "classe".*
```{r,warning=FALSE}

a<- ncol(training_original)
for (i in 1:(a-1)){training_original[,i] <- as.numeric(training_original[,i])}
b<- ncol(testing_original)
for (i in 1:(b-1)){testing_original [,i] <- as.numeric(testing_original [,i])}

```

*The variables which the sum of the column is zero were excluded.* 
```{r}
training_original <- training_original[, colSums(is.na(training_original)) ==0]
```


*The variables which the variation od the column is zero also were excluded.*
```{r}

classe  <- training_original[,ncol(training_original)]
zeroVar <- nearZeroVar(training_original, saveMetrics=TRUE)
training_original <- training_original[, zeroVar$nzv==FALSE]
```


*The variables with high correlation with another variable was removed using the collinearity approach.*
```{r}
#Removing columns with collinearity
correlation <- cor(training_original[,-ncol(training_original)])
top <- findCorrelation( correlation, cutoff =.75)
training_original <- training_original[, -top]
```

*Some variables were eliminated based on their null influence on the final results.*
```{r}
training_original <- select(training_original, -1,-2,-3,-4)
training_original$classe <- as.factor(training_original$classe)
```

*In order to maintain the project reproducible, it was utilized the set.seed (1000).*
```{r}
set.seed(1000)

```

*The data was split 70% for the training dataset and 30% for the testing dataset.*
```{r}
inTrain  <-  createDataPartition(training_original$classe, p=0.7, list=FALSE)
training <-  training_original[inTrain,]
testing  <-  training_original[-inTrain,]
```


*CART (Classification and regression Tree) RESULTS.*
```{r}

modFit_T<- train(classe~., method="rpart", data= training)
modFit_T
print(modFit_T$finalModel)
predictions_T <- predict(modFit_T, newdata=testing)
confusionMatrix(predictions_T, testing$classe)
```


```{r}
cart <- rpart(classe~., data=training, method="class")
prp(cart)
```


*RANDOM FOREST RESULTS.*
```{r}
modFit_RF<- train(classe~.,  method="rf", data = training)
modFit_RF
print(modFit_RF$finalModel)
predictions_RF <- predict(modFit_RF, newdata=testing)
confusionMatrix(predictions_RF, testing$classe)
```

```{r}
head(getTree(modFit_RF$finalModel, k=2))
pred<-testing$classe
table(pred, testing$classe)
```


*The final predict model with better accuracy ( Randon Forest ) was used to predict the 20 different test cases proposed in the initial request of the project* 

```{r}
predictions_RR <- predict(modFit_RF, newdata=testing_original)

```


*Table A: showing the prediction of the 20 different test cases*
```{r}
table(predictions_RR)
predictions_RR
```


     
        
           
           
sources:
http://groupware.les.inf.puc-rio.br/har.
https://citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics/
http://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/
