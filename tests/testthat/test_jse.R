library(sneer)
context("JSE")

# JSE with kappa -> 0 should be equivalent to ASNE
asne_iris <-
  embed_prob(iris[, 1:4], method = asne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

jse_iris_kappa0 <-
  embed_prob(iris[, 1:4], method = jse(kappa = 0, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(asne_iris$cost, jse_iris_kappa0$cost, tolerance = 0.001)

# JSE with kappa = 1 should be equivalent to rASNE
rasne_iris <-
  embed_prob(iris[, 1:4], method = rasne(verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

jse_iris_kappa1 <-
  embed_prob(iris[, 1:4], method = jse(kappa = 1,
                                       verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(rasne_iris$cost, jse_iris_kappa1$cost, tolerance = 0.001)


