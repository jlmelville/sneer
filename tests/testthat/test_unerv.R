library(sneer)
context("UNeRV")

test_that("NeRV with unit precisions has SNE methods as limiting cases", {
# UNeRV with lambda = 1 should be equivalent to ASNE
asne_iris <-
  embed_prob(iris[, 1:4], method = asne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

unerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = unerv(lambda = 1, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, asne_iris$report$costs),
             mapply(formatC, unerv_iris_lambda1$report$costs))

# UNeRV with lambda = 0 should be equivalent to rASNE
rasne_iris <-
  embed_prob(iris[, 1:4], method = rasne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

unerv_iris_lambda0 <-
  embed_prob(iris[, 1:4], method = unerv(lambda = 0, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, rasne_iris$report$costs),
             mapply(formatC, unerv_iris_lambda0$report$costs))

# t-NeRV with lambda = 1 should be equivalent to t-SNE
tsne_iris <-
  embed_prob(iris[, 1:4], method = tsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

tnerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = tnerv(lambda = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, tsne_iris$report$costs),
             mapply(formatC, tnerv_iris_lambda1$report$costs))


# t-NeRV with lambda = 0 should be equivalent to rt-SNE
rtsne_iris <-
  embed_prob(iris[, 1:4], method = rtsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

tnerv_iris_lambda0 <-
  embed_prob(iris[, 1:4], method = tnerv(lambda = 0, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, rtsne_iris$report$costs),
             mapply(formatC, tnerv_iris_lambda0$report$costs))

# SNeRV with lamda = 1 should be equivalent to SSNE
ssne_iris <-
  embed_prob(iris[, 1:4], method = ssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

usnerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = usnerv(lambda = 1, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, ssne_iris$report$costs),
             mapply(formatC, usnerv_iris_lambda1$report$costs))

# USNeRV with lamda = 0 should be equivalent to rSSNE
rssne_iris <-
  embed_prob(iris[, 1:4], method = rssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

usnerv_iris_lambda0 <-
  embed_prob(iris[, 1:4],
             method = usnerv(lambda = 0, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, rssne_iris$report$costs),
             mapply(formatC, usnerv_iris_lambda0$report$costs))

# HSNeRV with lambda = 1, alpha = 0 should be equivalent to SSNE
uhsnerv_iris_lambda1alpha0 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 1, alpha = 0, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, ssne_iris$report$costs),
             mapply(formatC, uhsnerv_iris_lambda1alpha0$report$costs))

# UHSNeRV with lambda = 1, alpha = 1 should be equivalent to t-SNE
uhsnerv_iris_lambda1alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 1, alpha = 1, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, tsne_iris$report$costs),
             mapply(formatC, uhsnerv_iris_lambda1alpha1$report$costs))

# HSNeRV with lambda = 0, alpha = 0 should be equivalent to rSSNE
uhsnerv_iris_lambda0alpha0 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 0, alpha = 0, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, rssne_iris$report$costs),
             mapply(formatC, uhsnerv_iris_lambda0alpha0$report$costs))

# UHSNeRV with lambda = 0, alpha = 1 should be equivalent to rt-SNE
uhsnerv_iris_lambda0alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 0, alpha = 1, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, rtsne_iris$report$costs),
             mapply(formatC, uhsnerv_iris_lambda0alpha1$report$costs))

# UHSNeRV with lambda = 0.5, alpha = 1 should be equivalent to
# t-NeRV with lambda 0.5
tnerv_iris_lambda0_5 <-
  embed_prob(iris[, 1:4], method = tnerv(lambda = 0.5, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

uhsnerv_iris_lambda0_5alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 0.5, alpha = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, tnerv_iris_lambda0_5$report$costs),
             mapply(formatC, uhsnerv_iris_lambda0_5alpha1$report$costs))


# UHSNeRV with lambda = 0.5, alpha = 0 should be equivalent to
# USNeRV with lambda 0.5
usnerv_iris_lambda0_5 <-
  embed_prob(iris[, 1:4], method = usnerv(lambda = 0.5, beta = 1,
                                         verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

uhsnerv_iris_lambda0_5alpha0_5 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = uhsnerv(lambda = 0.5, alpha = 0, beta = 1,
                             verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(mapply(formatC, usnerv_iris_lambda0_5$report$costs),
             mapply(formatC, uhsnerv_iris_lambda0_5alpha0_5$report$costs))
})
