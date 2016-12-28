library(sneer)
context("SNE")

test_that("tsne", {
tsne_iris <- embed_prob(iris[, 1:4],
                       method = tsne(verbose = FALSE),
                       init_inp = inp_from_perp(
                           perplexity = 50,
                           input_weight_fn = sqrt_exp_weight,
                           verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 160,
                       opt = mize_bold_nag_adapt(),
                       reporter = make_reporter(verbose = FALSE),
                       export = c("report"),
                       verbose = FALSE)
expect_equal(tsne_iris$report$norm, 0.0522, tolerance = 1e-4, label = "tsne")
})

test_that("ssne", {
ssne_iris <- embed_prob(iris[, 1:4],
                       method = ssne(verbose = FALSE),
                       init_inp = inp_from_perp(
                          perplexity = 50,
                          input_weight_fn = sqrt_exp_weight,
                          verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 65,
                       reporter = make_reporter(verbose = FALSE,
                                                report_every = 2),
                       export = c("report"),
                       verbose = FALSE,
                       opt = mize_bold_nag_adapt())
expect_equal(ssne_iris$report$norm, 0.0727, tolerance = 1e-4, label = "ssne")
# Used below to compare when ssne has beta = 5
expect_equal(ssne_iris$report$iter, 64,
             label = "ssne with non-default beta num iterations")
})

test_that("asne", {
asne_iris <- embed_prob(iris[, 1:4],
                       method = asne(verbose = FALSE),
                       init_inp = inp_from_perp(
                           perplexity = 50,
                           input_weight_fn = sqrt_exp_weight,
                           verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 70,
                       reporter = make_reporter(verbose = FALSE),
                       export = c("report"),
                       verbose = FALSE,
                       opt = mize_bold_nag_adapt())
expect_equal(asne_iris$report$norm, 0.0848, tolerance = 1e-4, label = "asne")
})

test_that("initial PCA result is reported correctly", {
tsne_iris <- embed_prob(iris[, 1:4],
                      method = tsne(verbose = FALSE),
                      init_out = out_from_PCA(verbose = FALSE),
                      init_inp = inp_from_perp(
                          perplexity = 25,
                          input_weight_fn = sqrt_exp_weight,
                          verbose = FALSE),
                      preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                   verbose = FALSE),
                      max_iter = 0, verbose = FALSE,
                      reporter = make_reporter(report_every = 1,
                                               verbose = FALSE),
                      export = c("report"))
expect_equal(tsne_iris$report$cost, 1.598, tolerance = 1e-3,
             label = "PCA init cost")
expect_equal(tsne_iris$report$norm, 0.929, tolerance = 1e-3,
             label = "PCA init norm cost")
})

test_that("characterize TASNE", {
tasne_iris <- embed_prob(iris[, 1:4],
                       method = tasne(verbose = FALSE),
                       init_inp = inp_from_perp(
                         perplexity = 25,
                         input_weight_fn = sqrt_exp_weight,
                         verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 0, verbose = FALSE,
                       reporter = make_reporter(report_every = 1,
                                                verbose = FALSE),
                       export = c("report"))
expect_equal(tasne_iris$report$norm, 0.931, tolerance = 1e-4, label = "tasne")
})

test_that("characterize TPSNE", {
  tpsne_iris <- embed_prob(iris[, 1:4],
                           method = tpsne(verbose = FALSE),
                           init_inp = inp_from_perp(
                             perplexity = 25,
                             input_weight_fn = sqrt_exp_weight,
                             verbose = FALSE),
                           preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                        verbose = FALSE),
                           max_iter = 0, verbose = FALSE,
                           reporter = make_reporter(report_every = 1,
                                                    verbose = FALSE),
                           export = c("report"))
  expect_equal(tpsne_iris$report$norm, 0.932, tolerance = 1e-3, label = "tasne")
})

test_that("tsne with Jacobs opt", {
  tsne_iris_jacobs <- embed_prob(iris[, 1:4],
                                 method = tsne(verbose = FALSE),
                                 init_inp = inp_from_perp(
                                   perplexity = 25,
                                   input_weight_fn = sqrt_exp_weight,
                                   verbose = FALSE),
                                 preprocess = make_preprocess(
                                   range_scale_matrix = TRUE,
                                   verbose = FALSE),
                                 max_iter = 10,
                                 opt = mize_opt("DBD", step_up = 0.2,
                                                 step_up_fun = "+",
                                                 step_down = 0.8),
                                 verbose = FALSE,
                                 reporter = make_reporter(report_every = 1,
                                                          keep_costs = TRUE,
                                                          verbose = FALSE),
                                 export = c("report"))
  jacobs_costs <- tsne_iris_jacobs$report$costs[,"cost"]
  expect_equal(jacobs_costs,  c(1.598, 1.594, 1.589, 1.584, 1.577, 1.57, 1.562,
                                1.553, 1.543, 1.532, 1.52),
               tolerance = 5e-4, scale = 1, label = "tsne with Jacobs opt costs")
})



test_that("different beta gives same converged results", {
  ssne_iris <- embed_prob(iris[, 1:4],
                          method = ssne(beta = 5),
                          init_inp = inp_from_perp(
                            perplexity = 50,
                            input_weight_fn = sqrt_exp_weight,
                            verbose = FALSE),
                          preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                       verbose = FALSE),
                          max_iter = 60,
                          reporter = make_reporter(verbose = FALSE,
                                                   report_every = 2),
                          export = c("report"),
                          verbose = FALSE,
                          opt = mize_bold_nag_adapt())
  # should be the same as the SSNE test above
  expect_equal(ssne_iris$report$norm, 0.0727, tolerance = 1e-4,
               label = "ssne with non-default beta")
  # should be different from the SSNE test above (or else we can't detect
  # changes to beta are ignored...)
  expect_equal(ssne_iris$report$iter, 60,
               label = "ssne with non-default beta num iterations")
})
