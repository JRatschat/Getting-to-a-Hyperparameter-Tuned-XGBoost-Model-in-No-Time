# Packages
library(readr)
library(dplyr)
library(data.table)
library(xgboost)
library(caret)

# Load data
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# Explore data
str(train)
str(test)

# Delete "unneccessary" columns; the name variable could be used to extract the title
train <- train %>%
  dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)
test <- test %>%
  dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)

# Transform all character variables into factor variables
train <- train %>% 
  dplyr::mutate(Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))
test <- test %>% 
  dplyr::mutate(Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

# Transform integer variable Pclass to factor
train <- train %>% 
  dplyr::mutate(Pclass = as.factor(Pclass))
test <- test %>% 
  dplyr::mutate(Pclass = as.factor(Pclass))

## Create new family size variable
train$Family <- train$SibSp + train$Parch
test$Family <- test$SibSp + test$Parch

# Randomly select 80% of the observations without replacement 
set.seed(20)
train_id <- sample(1:nrow(train), size = floor(0.8 * nrow(train)), replace=FALSE) 

# Split in train and validation (80/20)
training <- train[train_id,]
validation <- train[-train_id,]

# Returns the NA object unchanged, if not changed, NA would be dropped
options(na.action='na.pass')

# Prepare matrix for XGBoost algorithm
xsell_training_matrix <-model.matrix(Survived ~ .-1, data = training)
xsell_validation_matrix <-model.matrix(Survived ~ .-1, data = validation)
xsell_test_matrix <-model.matrix(~.-1, data = test)
dtrain <- xgb.DMatrix(data = xsell_training_matrix, label = training$Survived) 
dvalid <- xgb.DMatrix(data = xsell_validation_matrix, label = validation$Survived)
dtest <- xgb.DMatrix(data = xsell_test_matrix)

?xgb.DMatrix

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

## Random search

# Take start time to measure time of random search algorithm
start.time <- Sys.time()

# Create empty lists
lowest_error_list = list()
parameters_list = list()


# Create 100 rows with random hyperparameters
set.seed(20)
for (iter in 1:500){
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

write_csv(randomsearch, "data/randomsearch.csv")










