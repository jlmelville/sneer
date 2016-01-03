library(sneer)
context("HSSNE")

# HSSNE with alpha approaching zero should be equivalent to SSNE
ssne_iris <-
  embed_prob(iris[, 1:4], method = ssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hssne_iris_alpha0 <-
  embed_prob(iris[, 1:4], method = hssne(alpha = 1.5e-8, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, ssne_iris$report$costs),
             mapply(formatC, hssne_iris_alpha0$report$costs))

# HSSNE with alpha = 1 should be equivalent to t-SNE
tsne_iris <-
  embed_prob(iris[, 1:4], method = tsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hssne_iris_alpha1 <-
  embed_prob(iris[, 1:4], method = hssne(alpha = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, tsne_iris$report$costs),
             mapply(formatC, hssne_iris_alpha1$report$costs))
