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
  embed_prob(iris[, 1:4], method = nerv(lambda = 1, beta = 1, verbose = FALSE),
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
  embed_prob(iris[, 1:4], method = nerv(lambda = 0, beta = 1, verbose = FALSE),
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
})


rtsne_iris <-
  embed_prob(iris[, 1:4], method = rtsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

tsne_iris <-
  embed_prob(iris[, 1:4], method = tsne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

test_that("tNeRV with unit precisions has tSNE methods as limiting cases", {
# t-NeRV with lambda = 1 should be equivalent to t-SNE

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

expect_equal(tsne_iris$report$costs, tnerv_iris_lambda1$report$costs)


# t-NeRV with lambda = 0 should be equivalent to rt-SNE


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

expect_equal(rtsne_iris$report$costs, tnerv_iris_lambda0$report$costs)
})

# Used in HSNeRV tests too
ssne_iris <-
  embed_prob(iris[, 1:4], method = ssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

rssne_iris <-
  embed_prob(iris[, 1:4], method = rssne(verbose = FALSE), max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

test_that("SNeRV with unit precisions has SSNE methods as limiting cases", {
# SNeRV with lamda = 1 should be equivalent to SSNE

usnerv_iris_lambda1 <-
  embed_prob(iris[, 1:4], method = snerv(lambda = 1, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(ssne_iris$report$costs, usnerv_iris_lambda1$report$costs,
             tol = 1e-6)

# USNeRV with lamda = 0 should be equivalent to rSSNE
usnerv_iris_lambda0 <-
  embed_prob(iris[, 1:4],
             method = snerv(lambda = 0, beta = 1, verbose = FALSE),
             max_iter = 50,
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(rssne_iris$report$costs,
             usnerv_iris_lambda0$report$costs)
})

test_that("HSNeRV with unit precisions has HSSNE methods as limiting cases", {

# HSNeRV with lambda = 1, alpha = 0 should be equivalent to SSNE
uhsnerv_iris_lambda1alpha0 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 1, alpha = 0, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(ssne_iris$report$costs, uhsnerv_iris_lambda1alpha0$report$costs,
             tol = 1e-6)

# UHSNeRV with lambda = 1, alpha = 1 should be equivalent to t-SNE
uhsnerv_iris_lambda1alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 1, alpha = 1, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(tsne_iris$report$costs,
             uhsnerv_iris_lambda1alpha1$report$costs)

# HSNeRV with lambda = 0, alpha = 0 should be equivalent to rSSNE
uhsnerv_iris_lambda0alpha0 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 0, alpha = 0, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(rssne_iris$report$costs,
             uhsnerv_iris_lambda0alpha0$report$costs)

# UHSNeRV with lambda = 0, alpha = 1 should be equivalent to rt-SNE
uhsnerv_iris_lambda0alpha1 <-
  embed_prob(iris[, 1:4], max_iter = 50,
             method = hsnerv(lambda = 0, alpha = 1, beta = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(rtsne_iris$report$costs,
             uhsnerv_iris_lambda0alpha1$report$costs)
})

test_that("UHSNeRV is consistent with SNeRV and t-NeRV", {

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
             method = hsnerv(lambda = 0.5, alpha = 1, verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(tnerv_iris_lambda0_5$report$costs,
             uhsnerv_iris_lambda0_5alpha1$report$costs)


# UHSNeRV with lambda = 0.5, alpha = 0 should be equivalent to
# USNeRV with lambda 0.5
usnerv_iris_lambda0_5 <-
  embed_prob(iris[, 1:4], method = snerv(lambda = 0.5, beta = 1,
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
             method = hsnerv(lambda = 0.5, alpha = 0, beta = 1,
                             verbose = FALSE),
             init_inp = inp_from_perp(verbose = FALSE),
             init_out = out_from_PCA(verbose = FALSE),
             preprocess = make_preprocess(verbose = FALSE),
             reporter = make_reporter(report_every = 5, keep_costs = TRUE,
                                      verbose = FALSE),
             export = c("report"), verbose = FALSE,
             opt = mize_bold_nag())

expect_equal(usnerv_iris_lambda0_5$report$costs,
             uhsnerv_iris_lambda0_5alpha0_5$report$costs)
})
