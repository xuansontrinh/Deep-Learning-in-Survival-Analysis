# Include library
library(mlr3)
library(mlr3viz)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(mlr3keras)
library(mlr3filters)

library(paradox)

library(ggplot2)
library(GGally)
library(data.table)

# Initialize DeepHit learner
learner = lrn("surv.deephit")
print(learner)
learner$param_set$ids()
train_data = readRDS('data/train_data.Rds')
train_data$d1
