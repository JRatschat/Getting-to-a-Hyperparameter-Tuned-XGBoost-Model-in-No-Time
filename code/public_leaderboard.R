# Packages
library(readr)
library(dplyr)
library(data.table)
library(xgboost)
library(caret)
library(ggplot2)
library(cowplot)
library(gghighlight)

# Evaluation
public_leaderboard <- read_csv("data/titanic-publicleaderboard.csv")
public_leaderboard <- as.data.frame(public_leaderboard) %>%
  dplyr::select(Score) %>%
  arrange(-Score)
ggplot(public_leaderboard) + geom_histogram(aes(x=Score), binwidth = 0.001) + 
  scale_x_continuous(limits=c(0.6,0.85)) +
  gghighlight(Score > 0.7799)