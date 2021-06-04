# BENCHMARK PAPER-INSPIRED BASELINE LEARNER

set.seed(20210130)
paper_bmr_list = list()
for (tsk in names(tsks_train)) {
  n_features <- ncol(tsks_train[[tsk]]$data()) - 2
  design = benchmark_grid(
    tasks = tsks_train[[tsk]],
    learners = list(lrn("surv.kaplan"), paper_deephit_gen(n_features)),
    resamplings = outer_cv5
  )
  bmr = benchmark(design)
  paper_bmr_list[[tsk]] = bmr
}

paper_p_list = list()
for (tsk in names(tsks_train)) {
  paper_p_list[[tsk]] = autoplot(paper_bmr_list[[tsk]], measure=measure)
}
ggarrange(plotlist = paper_p_list, ncol = 3, nrow=3)


# TRAIN THE BASELINE LEARNER ON THE WHOLE DATASET AND PREDICT BLACKBOX TEST SET

set.seed(20210130)
km_list = list()

for (tsk in names(tsks_train)) {
  n_features = ncol(tsks_train[[tsk]]$data()) - 2
  sample_size = nrow(tsks_train[[tsk]]$data())

  model = paper_deephit_gen(n_features)$train(tsks_train[[tsk]])
  test_pred = model$predict_newdata(test_data[[tsk]])
  km_list[[tsk]] = test_pred
}

# save submission
saveRDS(km_list, sprintf("submissions/xst_paper_inspired_%s.Rds",
                         format(Sys.time(), "%Y_%m_%d")))