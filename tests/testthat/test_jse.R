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

# HSJSE kappa 0 alpha 0 equivalent to SSNE
ssne_iris <-
  embed_prob(iris[, 1:4], method = ssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hsjse_iris_kappa0alpha0 <-
  embed_prob(iris[, 1:4], method = hsjse(kappa = 0, alpha = 0, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(ssne_iris$cost, hsjse_iris_kappa0alpha0$cost, tolerance = 0.001,
             scale = 1)


# HSJSE kappa 1 alpha 0 "reverse" SSNE
rssne_iris <-
  embed_prob(iris[, 1:4], method = rssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hsjse_iris_kappa1alpha0 <-
  embed_prob(iris[, 1:4], method = hsjse(kappa = 1, alpha = 0, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(rssne_iris$cost, hsjse_iris_kappa1alpha0$cost, tolerance = 0.001)


# HSJSE kappa 0 alpha 1 equivalent to t-SNE
tsne_iris <-
  embed_prob(iris[, 1:4], method = tsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hsjse_iris_kappa0alpha1 <-
  embed_prob(iris[, 1:4], method = hsjse(kappa = 0, alpha = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(tsne_iris$cost, hsjse_iris_kappa0alpha1$cost, tolerance = 0.001)


# HSJSE kappa 1 alpha 1 equivalent to "reverse" t-SNE
rtsne_iris <-
embed_prob(iris[, 1:4], method = rtsne(verbose = FALSE), max_iter = 50,
           init_inp = inp_from_perp(verbose = FALSE),
           init_out = out_from_PCA(verbose = FALSE),
           preprocess = make_preprocess(verbose = FALSE),
           reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                    verbose = FALSE),
           export = c("report"), verbose = FALSE, opt = bold_nagger())

hsjse_iris_kappa1alpha1 <-
  embed_prob(iris[, 1:4], method = hsjse(kappa = 1, alpha = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      extra_costs = c('kl'),
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(rtsne_iris$cost, hsjse_iris_kappa1alpha1$cost, tolerance = 0.001,
             scale = 1)
