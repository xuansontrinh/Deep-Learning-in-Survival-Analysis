# install.packages("remotes")
# remotes::install_github("mlr-org/mlr3extralearners")

# install_pycox(
#   method = "conda",
#   conda = "auto",
#   pip = TRUE,
#   install_torch = TRUE
# )

library(data.table)
library(mlr3verse)
library(mlr3proba)
library(mlr3learners)
library(mlr3extralearners)
library(survivalmodels)
library(purrr)
library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)
library(paradox)

# READ DATA FROM FILES
train_data = readRDS("data/train_data.Rds")
test_data = readRDS("data/test_list_x.Rds")

# DEFINE MEASURE
measure = msr("surv.graf") # Integrated Graf Score (Integrated Brier Score)

# CREATE TASKS
tsks_train = imap(
  train_data,
  ~TaskSurv$new(
    id = .y,
    backend = .x,
    time = "time",
    event = "status")
)



# DEFINE PIPELINE OPERATORS

one_hot_encoding <- po("encode",
                       method = "one-hot",
                       affect_columns = selector_type("factor")
)

mean_imputation <- po("imputemean",
                      affect_columns = selector_type(c("numeric", "integer")))

mode_imputation <- po("imputemode",
                      affect_columns = selector_type(c("factor", "ordered")))


# DEFINE SETTINGS TO USE FOR BENCHMARKING + CHALLENGE

paper_deephit_gen <- function(n_features) {
  paper_deephit_learner <- lrn("surv.deephit")
  paper_deephit_learner$param_set$values$num_nodes = 
    c(n_features * 3, n_features * 5, n_features * 3)
  paper_deephit_learner$param_set$values$batch_size = 50
  paper_deephit_learner$param_set$values$dropout = 0.6
  paper_deephit_learner$param_set$values$optimizer = "adam"
  paper_deephit_learner$param_set$values$learning_rate = 0.01
  paper_deephit_learner$param_set$values$early_stopping = TRUE
  paper_deephit_learner$param_set$values$frac = 0.3
  
  
  paper_deephit_pipeline <- 
    mean_imputation %>>%
    mode_imputation %>>%
    po("scale") %>>% 
    one_hot_encoding %>>% 
    paper_deephit_learner
  GraphLearner$new(paper_deephit_pipeline, id="paper.tuned.deephit")
}

at_deephit_gen_equal_size_fixed_range <- function(n_features, sample_size){
  min_nodes = 32
  max_nodes = 64
  frac = 0.2
  if (sample_size < 100) {
    frac = 0.4
    min_nodes = 16
    max_nodes = 32
  }
  deephit_learner = lrn("surv.deephit", 
                        optimizer="adam",
                        frac=frac,
                        early_stopping=TRUE,
                        batch_size=50)
  deephit_pipeline <- 
    mean_imputation %>>%
    mode_imputation %>>%
    po("scale") %>>% 
    one_hot_encoding %>>% 
    deephit_learner
  deephit_glearner <- GraphLearner$new(deephit_pipeline, id="at.deephit")
  
  deephit_search_space <- ParamSet$new(list(
    ParamInt$new("n_layers", lower = 2, upper = 4),
    ParamInt$new("nodes_per_layer", lower = min_nodes, upper = max_nodes),
    ParamDbl$new("surv.deephit.learning_rate", lower = 0.005, upper = 0.02),
    ParamDbl$new("surv.deephit.dropout", lower = 0.1, upper = 0.3),
    ParamInt$new("surv.deephit.epochs", lower = 75, upper = 125),
    ParamDbl$new("surv.deephit.weight_decay", lower = 0.0001, upper = 0.001)
  )
  )
  deephit_search_space$trafo = function(x, param_set) {
    x$surv.deephit.num_nodes = rep(as.integer(as.character(x$nodes_per_layer)), x$n_layers)
    x$nodes_per_layer = x$n_layers = NULL
    return(x)
  }
  terminator = trm("evals", n_evals = 40)
  tuner = tnr("random_search")
  inner_cv5 <- rsmp("cv", folds = 5L)
  
  AutoTuner$new(
    learner = deephit_glearner,
    resampling = inner_cv5,
    measure = measure,
    search_space = deephit_search_space,
    terminator = terminator,
    tuner = tuner
  )
}

at_deephit_gen_equal_size_varied_range <- function(n_features, sample_size){
  mean_nodes = as.integer((min(n_features, sample_size) + 10)/2)
  frac = 0.2
  
  if (mean_nodes > 48) {
    mean_nodes = 48
  }
  if (sample_size < 100) {
    frac = 0.4
  }
  
  max_nodes = mean_nodes + as.integer(mean_nodes/3)
  min_nodes = mean_nodes - as.integer(mean_nodes/3)
  
  deephit_learner = lrn("surv.deephit", 
                        optimizer="adam",
                        frac=frac,
                        early_stopping=TRUE,
                        batch_size=50)
  deephit_pipeline <- 
    mean_imputation %>>%
    mode_imputation %>>%
    po("scale") %>>% 
    one_hot_encoding %>>% 
    deephit_learner
  deephit_glearner <- GraphLearner$new(deephit_pipeline, id="custom.at.deephit")
  
  deephit_search_space <- ParamSet$new(list(
    ParamInt$new("n_layers", lower = 2, upper = 4),
    ParamInt$new("nodes_per_layer", lower = min_nodes, upper = max_nodes),
    ParamDbl$new("surv.deephit.learning_rate", lower = 0.005, upper = 0.02),
    ParamDbl$new("surv.deephit.dropout", lower = 0.1, upper = 0.3),
    ParamInt$new("surv.deephit.epochs", lower = 75, upper = 125),
    ParamDbl$new("surv.deephit.weight_decay", lower = 0.0001, upper = 0.001)
  )
  )
  deephit_search_space$trafo = function(x, param_set) {
    x$surv.deephit.num_nodes = rep(as.integer(as.character(x$nodes_per_layer)), x$n_layers)
    x$nodes_per_layer = x$n_layers = NULL
    return(x)
  }
  terminator = trm("evals", n_evals = 40)
  tuner = tnr("random_search")
  inner_cv5 <- rsmp("cv", folds = 5L)
  
  AutoTuner$new(
    learner = deephit_glearner,
    resampling = inner_cv5,
    measure = measure,
    search_space = deephit_search_space,
    terminator = terminator,
    tuner = tuner
  )
}

at_deephit_gen_different_size_fixed_range <- function(n_features, sample_size){
  min_nodes = 32
  max_nodes = 64
  frac = 0.2
  if (sample_size < 100) {
    frac = 0.4
    min_nodes = 16
    max_nodes = 32
  }
  deephit_learner = lrn("surv.deephit", 
                        optimizer="adam",
                        frac=frac,
                        early_stopping=TRUE,
                        batch_size=50,
                        epochs=100)
  deephit_pipeline <- 
    mean_imputation %>>%
    mode_imputation %>>%
    po("scale") %>>% 
    one_hot_encoding %>>% 
    deephit_learner
  deephit_glearner <- GraphLearner$new(deephit_pipeline, id="at.deephit")
  
  deephit_search_space <- ParamSet$new(list(
    ParamInt$new("n_layers", lower = 2, upper = 4),
    ParamInt$new("nodes_per_layer", lower = min_nodes, upper = max_nodes),
    ParamDbl$new("surv.deephit.learning_rate", lower = 0.005, upper = 0.02),
    ParamDbl$new("surv.deephit.dropout", lower = 0.1, upper = 0.3),
    ParamInt$new("surv.deephit.epochs", lower = 75, upper = 125),
    ParamDbl$new("surv.deephit.weight_decay", lower = 0.0001, upper = 0.001)
  )
  )
  deephit_search_space$trafo = function(x, param_set) {
    num_nodes = c(x$nodes_per_layer)
    i = 2
    while (i < x$n_layers) {
      num_nodes = c(num_nodes, as.integer(5/3*x$nodes_per_layer))
      i = i + 1
    }
    num_nodes = c(num_nodes, x$nodes_per_layer)
    x$surv.deephit.num_nodes = num_nodes
    x$nodes_per_layer = x$n_layers = NULL
    return(x)
  }
  terminator = trm("evals", n_evals = 40)
  tuner = tnr("random_search")
  inner_cv5 <- rsmp("cv", folds = 5L)
  
  AutoTuner$new(
    learner = deephit_glearner,
    resampling = inner_cv5,
    measure = measure,
    search_space = deephit_search_space,
    terminator = terminator,
    tuner = tuner
  )
}

at_deephit_gen_different_size_varied_range <- function(n_features, sample_size){
  mean_nodes = as.integer((min(n_features, sample_size) + 10)/2)
  frac = 0.2
  
  if (mean_nodes > 48) {
    mean_nodes = 48
  }
  if (sample_size < 100) {
    frac = 0.4
  }
  
  max_nodes = mean_nodes + as.integer(mean_nodes/3)
  min_nodes = mean_nodes - as.integer(mean_nodes/3)
  
  deephit_learner = lrn("surv.deephit", 
                        optimizer="adam",
                        frac=frac,
                        early_stopping=TRUE,
                        batch_size=50)
  deephit_pipeline <- 
    mean_imputation %>>%
    mode_imputation %>>%
    po("scale") %>>% 
    one_hot_encoding %>>% 
    deephit_learner
  deephit_glearner <- GraphLearner$new(deephit_pipeline, id="custom.at.deephit")
  
  deephit_search_space <- ParamSet$new(list(
    ParamInt$new("n_layers", lower = 2, upper = 4),
    ParamInt$new("nodes_per_layer", lower = min_nodes, upper = max_nodes),
    ParamDbl$new("surv.deephit.learning_rate", lower = 0.005, upper = 0.02),
    ParamDbl$new("surv.deephit.dropout", lower = 0.1, upper = 0.3),
    ParamInt$new("surv.deephit.epochs", lower = 75, upper = 125),
    ParamDbl$new("surv.deephit.weight_decay", lower = 0.0001, upper = 0.001)
  )
  )
  deephit_search_space$trafo = function(x, param_set) {
    num_nodes = c(x$nodes_per_layer)
    i = 2
    while (i < x$n_layers) {
      num_nodes = c(num_nodes, as.integer(5/3*x$nodes_per_layer))
      i = i + 1
    }
    num_nodes = c(num_nodes, x$nodes_per_layer)
    x$surv.deephit.num_nodes = num_nodes
    x$nodes_per_layer = x$n_layers = NULL
    return(x)
  }
  terminator = trm("evals", n_evals = 40)
  tuner = tnr("random_search")
  inner_cv5 <- rsmp("cv", folds = 5L)
  
  AutoTuner$new(
    learner = deephit_glearner,
    resampling = inner_cv5,
    measure = measure,
    search_space = deephit_search_space,
    terminator = terminator,
    tuner = tuner
  )
}

outer_cv5 <- rsmp("cv", folds = 5L)

# # BENCHMARK PAPER-INSPIRED BASELINE LEARNER
# 
# set.seed(20210130)
# paper_bmr_list = list()
# for (tsk in names(tsks_train)) {
#   n_features <- ncol(tsks_train[[tsk]]$data()) - 2
#   design = benchmark_grid(
#     tasks = tsks_train[[tsk]],
#     learners = list(lrn("surv.kaplan"), paper_deephit_gen(n_features)),
#     resamplings = outer_cv5
#   )
#   bmr = benchmark(design)
#   paper_bmr_list[[tsk]] = bmr
# }
# 
# paper_p_list = list()
# for (tsk in names(tsks_train)) {
#   paper_p_list[[tsk]] = autoplot(paper_bmr_list[[tsk]], measure=measure)
# }
# ggarrange(plotlist = paper_p_list, ncol = 3, nrow=3)
# 
# 
# # TRAIN THE BASELINE LEARNER ON THE WHOLE DATASET AND PREDICT BLACKBOX TEST SET
# 
# set.seed(20210130)
# km_list = list()
# 
# for (tsk in names(tsks_train)) {
#   n_features = ncol(tsks_train[[tsk]]$data()) - 2
#   sample_size = nrow(tsks_train[[tsk]]$data())
#   
#   model = paper_deephit_gen(n_features)$train(tsks_train[[tsk]])
#   test_pred = model$predict_newdata(test_data[[tsk]])
#   km_list[[tsk]] = test_pred
# }


# # BENCHMARK CODE (APPLIED FOR EACH EXPERIMENT BY REPLACING THE FUNCTION NAME)

# set.seed(20210130)
# at_bmr_list = list()
# for (tsk in names(tsks_train)) {
#   n_features = ncol(tsks_train[[tsk]]$data()) - 2
#   sample_size = nrow(tsks_train[[tsk]]$data())
# 
#   # batch_size = max(30, as.integer(sample_size/3))
# 
#   design = benchmark_grid(
#     tasks = tsks_train[[tsk]],
#     learners = list(lrn("surv.kaplan"),
#                     at_deephit_gen_different_size_fixed_range(
#                       n_features = n_features,
#                       sample_size = sample_size)),
#     resamplings = outer_cv5
#   )
#   bmr = benchmark(design)
#   at_bmr_list[[tsk]] = bmr
# }
# 
# at_p_list = list()
# for (tsk in names(tsks_train)) {
#   at_p_list[[tsk]] = autoplot(at_bmr_list[[tsk]], measure=measure)
# }
# ggarrange(plotlist = at_p_list, ncol = 3, nrow=3)
# 
# for (tsk in names(tsks_train)) {
# print(dcast(melt(as.data.table(at_bmr_list[[tsk]]$aggregate(measure))[,
#                                                                 c("learner_id",
#                                                                   "surv.graf")],
#            id.vars = "learner_id"), variable ~ learner_id))
# }




# # CHALLENGE

# # Model 1: at_deephit_gen_equal_size_fixed_range
# 
# set.seed(20210130)
# km_list = list()
# model_list_equal_size_fixed_range = list()
# 
# for (tsk in names(tsks_train)) {
  # n_features = ncol(tsks_train[[tsk]]$data()) - 2
  # sample_size = nrow(tsks_train[[tsk]]$data())
  # 
  # model = at_deephit_gen_equal_size_fixed_range(
  #   n_features = n_features,
  #   sample_size = sample_size)$train(tsks_train[[tsk]])
  # model_list_equal_size_fixed_range[[tsk]] = model
  # test_pred = model$predict_newdata(test_data[[tsk]])
  # km_list[[tsk]] = test_pred
# }
# 
# # save submission
# saveRDS(km_list, sprintf("submissions/xst_equal_size_fixed_range_%s.Rds",
#                          format(Sys.time(), "%Y_%m_%d")))

# # Model 2: at_deephit_gen_equal_size_varied_range
# 
# set.seed(20210130)
# km_list = list()
# model_list_equal_size_varied_range = list()
# 
# for (tsk in names(tsks_train)) {
#   n_features = ncol(tsks_train[[tsk]]$data()) - 2
#   sample_size = nrow(tsks_train[[tsk]]$data())
# 
#   model = at_deephit_gen_equal_size_varied_range(
#     n_features = n_features,
#     sample_size = sample_size)$train(tsks_train[[tsk]])
#   model_list_equal_size_varied_range[[tsk]] = model
#   test_pred = model$predict_newdata(test_data[[tsk]])
#   km_list[[tsk]] = test_pred
# }
# 
# # save submission
# saveRDS(km_list, sprintf("submissions/xst_equal_size_varied_range_%s.Rds", 
#                          format(Sys.time(), "%Y_%m_%d")))

# # Model 3: at_deephit_gen_different_size_fixed_range
# 
# set.seed(20210130)
# km_list = list()
# model_list_different_size_fixed_range = list()
# 
# for (tsk in names(tsks_train)) {
#   n_features = ncol(tsks_train[[tsk]]$data()) - 2
#   sample_size = nrow(tsks_train[[tsk]]$data())
#   
#   model = at_deephit_gen_different_size_fixed_range(
#     n_features = n_features,
#     sample_size = sample_size)$train(tsks_train[[tsk]])
#   model_list_different_size_fixed_range[[tsk]] = model
#   test_pred = model$predict_newdata(test_data[[tsk]])
#   km_list[[tsk]] = test_pred
# }
# 
# # save submission
# saveRDS(km_list, sprintf("submissions/xst_different_size_fixed_range_%s.Rds", 
#                          format(Sys.time(), "%Y_%m_%d")))
# 
# 
# # Model 4: at_deephit_gen_different_size_varied_range
# 
# set.seed(20210130)
# km_list = list()
# model_list_different_size_varied_range = list()
# 
# for (tsk in names(tsks_train)) {
#   n_features = ncol(tsks_train[[tsk]]$data()) - 2
#   sample_size = nrow(tsks_train[[tsk]]$data())
#   
#   model = at_deephit_gen_different_size_varied_range(
#     n_features = n_features,
#     sample_size = sample_size)$train(tsks_train[[tsk]])
#   model_list_different_size_varied_range[[tsk]] = model
#   test_pred = model$predict_newdata(test_data[[tsk]])
#   km_list[[tsk]] = test_pred
# }
# 
# # save submission
# saveRDS(km_list, sprintf("submissions/xst_different_size_varied_range_%s.Rds", 
#                          format(Sys.time(), "%Y_%m_%d")))
