library(sneer)
context("SNE")

tsne_iris <- embed_prob(iris[, 1:4],
                       method = tsne(verbose = FALSE),
                       init_inp = inp_from_perp(
                           perplexity = 50,
                           input_weight_fn = sqrt_exp_weight,
                           verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 200,
                       reporter = make_reporter(verbose = FALSE),
                       export = c("report"),
                       verbose = FALSE)
expect_equal(tsne_iris$report$norm, 0.0578, tolerance = 5e-5, scale = 1)

ssne_iris <- embed_prob(iris[, 1:4],
                       method = ssne(verbose = FALSE),
                       init_inp = inp_from_perp(
                          perplexity = 50,
                          input_weight_fn = sqrt_exp_weight,
                          verbose = FALSE),
                       preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                    verbose = FALSE),
                       max_iter = 60,
                       reporter = make_reporter(verbose = FALSE),
                       export = c("report"),
                       verbose = FALSE,
                       opt = bold_nag())
expect_equal(ssne_iris$report$norm, 0.07366, tolerance = 5e-5, scale = 1)

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
                       opt = bold_nag(max_momentum = 0.85))
expect_equal(asne_iris$report$norm, 0.08479, tolerance = 5e-6, scale = 1)

pca_iris <- embed_prob(iris[, 1:4],
                      method = tsne(verbose = FALSE),
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
expect_equal(pca_iris$report$cost, 1.598, tolerance = 5e-4, scale = 1)

context("Jacobs")
tsne_iris_jacobs <- embed_prob(iris[, 1:4],
                              method = tsne(verbose = FALSE),
                              init_inp = inp_from_perp(
                                  perplexity = 25,
                                  input_weight_fn = sqrt_exp_weight,
                                  verbose = FALSE),
                              preprocess = make_preprocess(
                                range_scale_matrix = TRUE,
                                verbose = FALSE),
                              max_iter = 10, opt =
                                make_opt(
                                  step_size = jacobs(inc_fn = partial(`+`, 0.2),
                                    dec_mult = 0.8, min_step_size = 0.1),
                                  normalize_direction = FALSE),
                              verbose = FALSE,
                              reporter = make_reporter(report_every = 1,
                                                       keep_costs = TRUE,
                                                       verbose = FALSE),
                              export = c("report"))
jacobs_costs <- tsne_iris_jacobs$report$costs[,"cost"]
expect_equal(jacobs_costs,  c(1.598, 1.594, 1.589, 1.584, 1.577, 1.57, 1.562,
                              1.553, 1.543, 1.532, 1.52),
             tolerance = 5e-4, scale = 1)


test_that("different beta gives different results", {
  ssne_iris <- embed_prob(iris[, 1:4],
                          method = ssne(beta = 0.5),
                          init_inp = inp_from_perp(
                            perplexity = 50,
                            input_weight_fn = sqrt_exp_weight,
                            verbose = FALSE),
                          preprocess = make_preprocess(range_scale_matrix = TRUE,
                                                       verbose = FALSE),
                          max_iter = 60,
                          reporter = make_reporter(verbose = FALSE),
                          export = c("report"),
                          verbose = FALSE,
                          opt = bold_nag())
  expect_equal(ssne_iris$report$norm, 0.07327, tolerance = 5e-5, scale = 1)
})
