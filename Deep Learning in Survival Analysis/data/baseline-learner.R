library(mlr3)
library(mlr3proba)
library(mlr3learners)
library(mlr3viz)
library(purrr)

# create tasks
train_data = readRDS("train_data.Rds")

tsks_train = imap(
  train_data, # competing risks case
  ~TaskSurv$new(
    id = .y,
    backend = .x,
    time = "time",
    event = "status")
)

# create learner
lrn_kaplan = lrn("surv.kaplan")

# train learner
km_list = map(
  tsks_train,
  ~lrn_kaplan$train(.x))

# save submission
saveRDS(km_list, "submissions/AB_baseline_2020-12-27.Rds")
