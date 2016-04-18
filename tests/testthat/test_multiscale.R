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

  # can't have two identical perplexities without causing the d_int
  # finite difference estimate to blow up, but a difference of 0.01
  # gives a cost within tolerance
  ssne_iris_m2 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = c(25, 25.01),
                                    num_scale_iters = 0),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(verbose = FALSE, report_every = 5),
    max_iter = 20,
    export = c("report")
  )

  expect_equal(ssne_iris_m2$report$cost,
               ssne_iris$report$cost,
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE with 2 nearly-equal scales")
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
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ms3$method$kernels, function(k) { k$beta }),
               c(1/150, 1/100, 1/50), info = "scaled kernels")

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
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ums3$method$kernels, function(k) { k$beta }),
               c(1, 1, 1), info = "uniform kernels")

  expect_equal(ssne_iris_ums3$report$costs[,"norm"],
               c(0.9360,  0.8311,  0.7047,  0.5821,  0.4716, # perp 75
                 0.1659,  0.1291,  0.1019,  0.0824,  0.0687, # perp 50
                 0.1153,  0.1152,  0.1123,  0.1093,  0.1049, # perp 25
                 0.1005,  0.0965,  0.0933,  0.0910,  0.0900,  0.0898),
               tolerance = 5e-4, scale = 1, label = "ms-SSNE unit prec")
})


test_that("Can apply multiple scales in one iteration", {
  ssne_iris_ums3_s0 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 0,
                                    modify_kernel_fn = NULL,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report", "method")
  )
  expect_equal(sapply(ssne_iris_ums3_s0$method$kernels, function(k) { k$beta }),
               c(1, 1, 1), info = "uniform kernels")

   expect_equal(ssne_iris_ums3_s0$report$costs[,"norm"],
               c(0.2961,  0.2527,  0.2066,  0.1703,  0.1456,
                 0.1286,  0.1157,  0.1056,  0.0982,  0.0935,
                 0.0910,  0.0900,  0.0899,  0.0899,  0.0898,
                 0.0895,  0.0893,  0.0892,  0.0892,  0.0891,  0.0891),
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE unit prec scale all at once")

  ssne_iris_ms3_s0 <- embed_prob(
    iris[, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 0,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 20,
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ms3_s0$method$kernels, function(k) { k$beta }),
               c(1/150, 1/100, 1/50), info = "scaled kernels")

  expect_equal(ssne_iris_ms3_s0$report$costs[,"norm"],
               c(0.9180,  0.9143,  0.9092,  0.9035,  0.8973,
                 0.8907,  0.8839,  0.8769,  0.8696,  0.8622,
                 0.8545,  0.8467,  0.8388,  0.8307,  0.8225,
                 0.8141,  0.8057,  0.7971,  0.7884,  0.7796,  0.7708),
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE scaled prec scale all at once")
})

test_that("Can combine multiscaling with asymmetric weights", {
  ssne_iris_tms3_s10 <- embed_prob(
    iris[1:10, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(8, 4, length.out = 3),
                                    num_scale_iters = 10,
                                    modify_kernel_fn =
                                      transfer_kernel_bandwidths,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 2,
                             verbose = FALSE),
    max_iter = 10,
    export = c("report", "method")
  )

  all_betas <- sapply(ssne_iris_tms3_s10$method$kernels, function(k) { k$beta })

  expected_betas <- matrix(c(
    c( 0.1359, 0.1792, 0.1418, 0.1577,  0.1374,
      0.07737, 0.1729, 0.1759, 0.08813, 0.1215),
    c(0.3431, 1.0787, 0.7401, 0.7476, 0.3432,
      0.1717, 0.8990, 0.5226, 0.3159, 0.4832),
    c(0.6140, 2.0297, 1.2530, 1.5626, 0.6007,
      0.3288, 1.7873, 1.0821, 0.6475, 1.0703)), ncol = 3)

  expect_equal(all_betas[, 1], expected_betas[, 1],
               info = "precs for perp 8", tolerance = 1e-4, scale = 1)
  expect_equal(all_betas[, 2], expected_betas[, 2],
               info = "precs for perp 6", tolerance = 1e-4, scale = 1)
  expect_equal(all_betas[, 3], expected_betas[, 3],
               info = "precs for perp 4", tolerance = 1e-4, scale = 1)

  # repeat this test scaling over 0 iterations
  # should be the same
  ssne_iris_tms3_s0 <- embed_prob(
    iris[1:10, 1:4],
    method = ssne_plugin(verbose = FALSE),
    opt = back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(8, 4, length.out = 3),
                                    num_scale_iters = 0,
                                    modify_kernel_fn =
                                      transfer_kernel_bandwidths,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 10,
    export = c("report", "method")
  )

  all_betas_s0 <- sapply(ssne_iris_tms3_s0$method$kernels, function(k) { k$beta })
  expect_equal(all_betas_s0[, 1], expected_betas[, 1],
               info = "0 iter scale precs for perp 8", tolerance = 1e-4, scale = 1)
  expect_equal(all_betas_s0[, 2], expected_betas[, 2],
               info = "0 iter scale precs for perp 6", tolerance = 1e-4, scale = 1)
  expect_equal(all_betas_s0[, 3], expected_betas[, 3],
               info = "0 iter scale precs for perp 4", tolerance = 1e-4, scale = 1)
})

