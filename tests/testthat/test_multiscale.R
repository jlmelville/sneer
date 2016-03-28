library(sneer)
context("Multiscaling")

# Test that the cost shows three spikes due to the perplexity change
# but that the optimizer is able to reduce the cost after each spike
# does not change the output kernel precision as perplexity changes
test_that("multiscaling with equal perplexities is the same as single scale", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(beta = 0.02),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perp(perplexity = 25, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(verbose = FALSE, report_every = 5),
    max_iter = 20,
    export = c("report")
  )

  ssne_iris_m1 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = c(25), verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(verbose = FALSE, report_every = 5),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris_m1$report$cost,
               ssne_iris$report$cost,
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE with 1 scale")

  ssne_iris_m2 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = c(25, 25)),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(verbose = FALSE, report_every = 5),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris_m2$report$cost,
               ssne_iris$report$cost,
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE with 2 equal scales")
})

test_that("multiscaling SSNE with perp scaling", {
  ssne_iris_ms3 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 10, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris_ms3$report$costs[,"norm"],
               c(0.9291,  0.9261,  0.9219,  0.9171,  0.9119, # perp 75
                 0.9054,  0.8995,  0.8934,  0.8871,  0.8806, # perp 50
                 0.8548,  0.8471,  0.8391,  0.8311,  0.8229, # perp 25
                 0.8146,  0.8062,  0.7977,  0.7890,  0.7803,  0.7716),
               tolerance = 5e-4, scale = 1, label = "ms-SSNE scaled prec")

})

test_that("ms SSNE with unit scaling", {
  ssne_iris_ums3 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 10,
                                    modify_kernel_fn = NULL,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris_ums3$report$costs[,"norm"],
               c(0.9360,  0.8311,  0.7047,  0.5821,  0.4716, # perp 75
                 0.1659,  0.1291,  0.1019,  0.0824,  0.0687, # perp 50
                 0.1153,  0.1152,  0.1123,  0.1093,  0.1049, # perp 25
                 0.1005,  0.0965,  0.0933,  0.0910,  0.0900,  0.0898),
               tolerance = 5e-4, scale = 1, label = "ms-SSNE unit prec")
})
