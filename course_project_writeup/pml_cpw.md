---
title: "pml_cpw"
author: "Dhruv K Pant"
date: "June 18, 2014"
output: html_document
---
## Prediction Analysis of personal activity data 

### Summary 
The goal of the following analysis is to understand the variables which are important in 
the data on personal activity with the aim of predicting which class  a group of persons belong to given 
new data on their personal activity.  
First, we need to clean that data, which has several NAs as entries. We identify the columns in the training 
data matrix which have a high percentage of NAs and remove those columns from further analysis. Additionally, 
only those columns which have arm/forearm/dumbbell/belt are retained. 
In a similar manner, variables from the testing dataset are removed and the variables which are common to the testing 
and the training datasets are kept for building a prediction model.
Correlation among the variables is explored next, in order to assess multicollinearity and to remove highly correlated 
variables. Those variable pairs which have a Pearson correlation coefficient of 0.8 or higher are identified. Potentially
one variable of each pair could be removed. 
A seed is set using set.seed function, in order that the computation is reproducible.
A classification and regression tree is built in two ways- 1) doing a preprocessing of the data by centering and scaling the 
data and 2) without doing the centering and scaling. For the models generated, the importance of the variables is assessed.
This identified 14 variables as being important in the prediction.
Next, two other algorithms namely boosting and Random Forests and used to generate models for prediction. 
In one approach, all 53 variables are used in the model fitting. In the other approach, 14 variables are used. 
A confusion matrix enables one to identify how good the predictions are.
The result indicates that the prediction accuracy is better in the bigger (53 variable) model than 
the 14 variable model, but not by much. The choice of variable selection would be a tradeoff between slightly 
improved accuracy and processing time.



### Read in input data 

```r
tr <- read.csv("pml-training.csv")
te <- read.csv("pml-testing.csv")
```


### Explore and preprocess the data

```r
# str(tr); dim(tr) colnames(tr)
table(tr$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


### Identifying NA's 

```r
### On the training set
cols.w.nas <- apply(tr, 2, function(x) sum(is.na(x)))
### Remove columns which have high percentage of NAs
cols.to.remove <- names(cols.w.nas[cols.w.nas > 0])
tr2 <- tr[, !colnames(tr) %in% cols.to.remove]
### keep variables pertaining to belt/forearm/arm/dumbbell
ind1 <- grepl("_arm|_forearm|_dumbbell|_belt", colnames(tr2))
tr3 <- cbind(tr2[, ind1], classe = tr2$classe)
# ind2 <- !grepl('yaw',colnames(tr3)) tr4 <- tr3[,ind2] x <- head(tr4,n=100)
cn.tr <- colnames(tr3)
```



```r
### On the testing set
cols.w.nas <- apply(te, 2, function(x) sum(is.na(x)))
cols.to.remove <- names(cols.w.nas[cols.w.nas > 0])
te2 <- te[, !colnames(te) %in% cols.to.remove]
ind1 <- grepl("_arm|_forearm|_dumbbell|_belt", colnames(te2))
te3 <- cbind(te2[, ind1], problem_id = te2$problem_id)
cn.te <- colnames(te3)

common <- intersect(cn.tr, cn.te)
```


### Explore correlation among variables

```r
x <- cor(subset(tr4, select = -classe))
```

```
## Error: object 'tr4' not found
```

```r
diag(x) <- 0
```

```
## Error: object 'x' not found
```

```r
which(abs(x) > 0.8, arr.ind = T)
```

```
## Error: object 'x' not found
```


### Variable selection

```r
tr4 <- cbind(tr3[, colnames(tr3) %in% common], classe = tr3$classe)
te4 <- cbind(te3[, colnames(te3) %in% common], problem_id = te3$problem_id)
```


### Fit different models on the data 

```r
### center and scale, log transformation, Box-Cox??
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12365)

### Generate a classification and regression tree
mod.rpart.scaled <- train(classe ~ ., data = tr4, preProcess = c("center", "scale"), 
    method = "rpart")
```

```
## Loading required package: rpart
```

```r
mod.rpart <- train(classe ~ ., data = tr4, method = "rpart")
predict(mod.rpart.scaled, te4)
```

```
##  [1] C A C A A C C A A A C C C A C A A A A C
## Levels: A B C D E
```

```r

### boosting, random forest on 53 variables
mod.rf <- train(classe ~ ., data = tr4, method = "rf", trControl = trainControl(method = "cv"), 
    number = 3)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
mod.gbm <- train(classe ~ ., data = tr4, method = "gbm", verbose = F)  ### takes v long
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
# mod.rf.oob <- train(classe~.,
# data=tr4,method='rf',trControl=trainControl(method='oob'),number=3)

### assess variable importance from the rpart models ### and consider only
### those variables for the RF and boosting models
tmp <- varImp(mod.rpart)
imp.var <- rownames(tmp$importance)[tmp$importance > 0]
tr5 <- cbind(tr4[, colnames(tr4) %in% imp.var], classe = tr4$classe)
te5 <- cbind(te4[, colnames(te4) %in% imp.var], problem_id = te4$problem_id)
# tr6 <- head(tr5,n=1000)
# featurePlot(y=tr6$classe,x=subset(tr6,select=-classe),plot='pairs')

### boosting, random forests on 14 variables (identified as important from a
### classification and regression tree model)
mod.rf.small <- train(classe ~ ., data = tr5, method = "rf", trControl = trainControl(method = "cv"), 
    number = 3)
```




### Predictions and Error analysis

```r
pred.gbm <- predict(mod.gbm, te4)
pred.rf <- predict(mod.rf, te4)
pred.rf.oob <- predict(mod.rf.oob, te4)
```

```
## Error: object 'mod.rf.oob' not found
```

```r
pred.rf.small <- predict(mod.rf.small, te5)

### Look at confusion matrix for prediction done with 53 variables
confusionMatrix(mod.rf)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.3  0.1  0.0  0.0
##          C  0.0  0.0 17.4  0.2  0.0
##          D  0.0  0.0  0.0 16.2  0.0
##          E  0.0  0.0  0.0  0.0 18.3
```

```r

### Look at confusion matrix for prediction done with only 14 variables
confusionMatrix(mod.rf.small)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.1  0.1  0.0  0.0
##          C  0.0  0.2 17.2  0.2  0.0
##          D  0.0  0.0  0.1 16.1  0.1
##          E  0.0  0.0  0.0  0.0 18.3
```

```r

### store the predictions in object 'answers' for the course project
### submission
answers <- pred.rf
```






