# Packages
library(readr)
library(dplyr)
library(data.table)
library(xgboost)
library(caret)
library(ggplot2)
library(cowplot)
library(stargazer)
library(missForest)
library(randomForest)

# Load data
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# Explore data
str(train)
str(test)

# Save Survived and passenger ID for later
Survived <- train$Survived
PassengerId <- test$PassengerId

## Prepare and put data into one dataframe

# Create new column that indicates if it the train or the test set
train$Data <- "train"
test$Data <- "test"

# Delete Survived in train set; we saved it earlier
train <- train %>%
  dplyr::select(-Survived)

# Bind rows to create a single data set
total <- rbind(train, test)

# Transform integer variable Pclass to factor
total <- total %>% 
  dplyr::mutate(Pclass = as.factor(Pclass))

# Create new family size variable
total$Fsize <- total$SibSp + total$Parch + 1

# Discretize family size
total$FsizeD[total$Fsize == 1] <- 'singleton'
total$FsizeD[total$Fsize < 5 & total$Fsize > 1] <- 'small'
total$FsizeD[total$Fsize > 4] <- 'large'

# Title
# https://trevorstephens.com/kaggle-titanic-tutorial/r-part-4-feature-engineering/
total$Name <- as.character(total$Name)
total$Title <- sapply(total$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
total$Title <- sub(' ', '', total$Title)
total$Title[total$Title %in% c('Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir')] <- 'Mr'
total$Title[total$Title %in% c('Mme', 'Dona', 'Lady', 'the Countess')] <- 'Mrs'
total$Title[total$Title %in% c('Mme', 'Ms', 'Mlle')] <- 'Miss'
total$Title <- factor(total$Title)

# Create a Deck variable. Get passenger deck A - F:
total$Deck <- factor(sapply(total$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# Create the column child, and indicate whether child or adult
total$Child[total$Age < 18] <- 'Child'
total$Child[total$Age >= 18] <- 'Adult'

# Adding Mother variable
total$Mother <- 'Not Mother'
total$Mother[total$Sex == 'female' & total$Parch > 0 & total$Age > 18 & total$Title != 'Miss'] <- 'Mother'





# Delete "unneccessary" columns
total <- total %>%
  dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)

# Transform all character variables into factor variables
total <- total %>% 
  dplyr::mutate(Sex = as.factor(Sex),
                Embarked = as.factor(Embarked),
                Data = as.factor(Data))

# Create one-hot coded matrix
total_matrix <-model.matrix(~.-1, data = total)

# Split back into train and test set and delete Data
train <- as.data.frame(total_matrix) %>%
  dplyr::filter(Datatrain == 1)
test <- as.data.frame(total_matrix) %>%
  dplyr::filter(Datatrain == 0)
train <- train %>%
  dplyr::select(-Datatrain)
test <- test %>%
  dplyr::select(-Datatrain)

# Put Survived back into train
train$Survived <- Survived

# Randomly select 80% of the observations without replacement 
set.seed(20)
train_id <- sample(1:nrow(train), size = floor(0.8 * nrow(train)), replace=FALSE) 

# Split in train and validation (80/20)
training <- train[train_id,]
validation <- train[-train_id,]

# Returns the NA object unchanged, if not changed, NA would be dropped
options(na.action='na.pass')

# Prepare matrix for XGBoost algorithm
training_matrix <-model.matrix(Survived ~.-1, data = training)
validation_matrix <-model.matrix(Survived ~.-1, data = validation)
test_matrix <-model.matrix(~.-1, data = test)

dtrain <- xgb.DMatrix(data = training_matrix, label = training$Survived) 
dvalid <- xgb.DMatrix(data = validation_matrix, label = validation$Survived)
dtest <- xgb.DMatrix(data = test_matrix)

# Base XGBoost model
set.seed(20)
params <- list(booster = "gbtree", 
               objective = "binary:logistic")
xgb_base <- xgb.train (params = params,
                       data = dtrain,
                       nrounds =1000,
                       print_every_n = 10,
                       eval_metric = "auc",
                       eval_metric = "error",
                       early_stopping_rounds = 50,
                       watchlist = list(train= dtrain, val= dvalid))

# Make prediction on dvalid
validation$pred_survived_base <- predict(xgb_base, dvalid)
validation$pred_survived_factor_base <- factor(ifelse(validation$pred_survived_base > 0.5, 1, 0), 
                                               labels=c("Not Survived","Survived"))

# Check accuracy with the confusion matrix
confusionMatrix(validation$pred_survived_factor_base, 
                factor(validation$Survived ,
                       labels=c("Not Survived", "Survived")),
                positive = "Survived", 
                dnn = c("Prediction", "Actual Data"))

# Test
test$pred_survived_base <- predict(xgb_base, dtest)
test$Survived <- factor(ifelse(test$pred_survived_base > 0.5, 1, 0))
test_final <- cbind(as.data.frame(PassengerId), test$Survived)
test_final <- test_final %>%
  rename("Survived" = "test$Survived")
write_csv(test_final, "data/datasubmission_base.csv")

## Random search

# Take start time to measure time of random search algorithm
start.time <- Sys.time()

# Create empty lists
lowest_error_list = list()
parameters_list = list()

# Create 100 rows with random hyperparameters
set.seed(20)
for (iter in 1:100){
  param <- list(booster = "gbtree",
                objective = "binary:logistic",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=dtrain,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= 300,
                    eval_metric = "error",
                    early_stopping_rounds= 30,
                    print_every_n = 100,
                    watchlist = list(train= dtrain, val= dvalid)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_list[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_error_df, parameters_df)

# Quickly display highest accuracy
max(randomsearch$`1 - min(mdcv$evaluation_log$val_error)`)

# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#write_csv(randomsearch, "data/randomsearch.csv")

# Load random search output
randomsearch <- read_csv("data/randomsearch.csv")

# Prepare table
randomsearch <- as.data.frame(randomsearch) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)

# Tuned-XGBoost model
set.seed(20)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearch[1,]$max_depth,
               eta = randomsearch[1,]$eta,
               subsample = randomsearch[1,]$subsample,
               colsample_bytree = randomsearch[1,]$colsample_bytree,
               min_child_weight = randomsearch[1,]$min_child_weight)
xgb_tuned <- xgb.train(params = params,
                       data = dtrain,
                       nrounds =1000,
                       print_every_n = 10,
                       eval_metric = "auc",
                       eval_metric = "error",
                       early_stopping_rounds = 30,
                       watchlist = list(train= dtrain, val= dvalid))

# Make prediction on dvalid
validation$pred_survived_tuned <- predict(xgb_tuned, dvalid)
validation$pred_survived_factor_tuned <- factor(ifelse(validation$pred_survived_tuned > 0.5, 1, 0), 
                                                labels=c("Not Survived","Survived"))

# Check accuracy with the confusion matrix
confusionMatrix(validation$pred_survived_factor_tuned, 
                factor(validation$Survived ,
                       labels=c("Not Survived", "Survived")),
                positive = "Survived", 
                dnn = c("Prediction", "Actual Data"))

# Test
test$pred_survived_tuned <- predict(xgb_tuned, dtest)
test$Survived <- factor(ifelse(test$pred_survived_tuned > 0.5, 1, 0))
test_final <- cbind(as.data.frame(PassengerId), test$Survived)
test_final <- test_final %>%
  rename("Survived" = "test$Survived")
write_csv(test_final, "data/datasubmission_tuned.csv")
