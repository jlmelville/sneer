library(sneer)
context("NeRV")

# NeRV with lambda = 1 should be equivalent to ASNE
asne_iris <-
  embed_prob(iris[, 1:4], method = asne(verbose = FALSE), max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

nerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = nerv(lambda = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, asne_iris$report$costs),
             mapply(formatC, nerv_iris_lambda1$report$costs))

# t-NeRV with lambda = 1 should be equivalent to t-SNE
tsne_iris <-
  embed_prob(iris[, 1:4], method = tsne(verbose = FALSE), max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

tnerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = tnerv(lambda = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, tsne_iris$report$costs),
             mapply(formatC, tnerv_iris_lambda1$report$costs))


# SNeRV with lamda = 1 should be equivalent to SSNE
ssne_iris <-
  embed_prob(iris[, 1:4], method = ssne(verbose = FALSE), max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

snerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = snerv(lambda = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, ssne_iris$report$costs),
             mapply(formatC, snerv_iris_lambda1$report$costs))

# HSNeRV with lambda = 1, alpha = 1.5e-8 should be equivalent to SSNE
hsnerv_iris_lambda1alpha0 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 1, alpha = 1.5e-8, verbose = FALSE),
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, ssne_iris$report$costs),
             mapply(formatC, hsnerv_iris_lambda1alpha0$report$costs))

# HSNeRV with lambda = 1, alpha = 1 should be equivalent to t-SNE
hsnerv_iris_lambda1alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 1, alpha = 1, verbose = FALSE),
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, tsne_iris$report$costs),
             mapply(formatC, hsnerv_iris_lambda1alpha1$report$costs))

# HSNeRV with lambda = 0.5, alpha = 1 should be equivalent to
# t-NeRV with lambda 0.5
tnerv_iris_lambda0_5 <-
  embed_prob(iris[, 1:4], method = tnerv(lambda = 0.5, verbose = FALSE),
             max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hsnerv_iris_lambda0_5alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 0.5, alpha = 1, verbose = FALSE),
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, tnerv_iris_lambda0_5$report$costs),
             mapply(formatC, hsnerv_iris_lambda0_5alpha1$report$costs))


# HSNeRV with lambda = 0.5, alpha = 1.5e-8 should be equivalent to
# SNeRV with lambda 0.5
snerv_iris_lambda0_5 <-
  embed_prob(iris[, 1:4], method = snerv(lambda = 0.5, verbose = FALSE),
             max_iter = 50,
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

hsnerv_iris_lambda0_5alpha0_5 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 0.5, alpha = 1.5e-8, verbose = FALSE),
             init_inp = make_init_inp(prob_perp_bisect(verbose = FALSE)),
             init_out = make_init_out(from_PCA = TRUE, verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE, opt = bold_nagger())

expect_equal(mapply(formatC, snerv_iris_lambda0_5$report$costs),
             mapply(formatC, hsnerv_iris_lambda0_5alpha0_5$report$costs))
