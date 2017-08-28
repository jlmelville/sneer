library(sneer)
context("Multiscaling")

test_that("default multiscale perplexities make sense", {
  expect_equal(ms_perps(matrix(nrow = 100000)),
               c(32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32,
                 16, 8, 4, 2))
  expect_equal(ms_perps(matrix(nrow = 100000), 10),
               c(1024, 512, 256, 128, 64, 32, 16, 8, 4, 2))
  expect_equal(ms_perps(matrix(nrow = 100), 10),
               c(64, 32, 16, 8, 4, 2))
})

test_that("default step perplexities make sense", {
  expect_equal(step_perps(matrix(nrow = 100000)),
               c(50000, 37508, 25016, 12524, 32),
               info = "default 5 equally spaced perplexities, from N/2 to 32")
  expect_equal(step_perps(matrix(nrow = 100000), 10),
               c(50000, 44448, 38896, 33344, 27792, 22240, 16688, 11136, 5584,
                 32),
               info = "can specify number of steps")
  expect_equal(step_perps(matrix(nrow = 60), 4),
               c(30, 25, 20, 15),
               info = "scale down to N/4 if max perp would be <= 32")
  expect_equal(step_perps(matrix(nrow = 28), 8),
               c(14, 13, 12, 11, 10, 9, 8, 7),
               info = "scale down to N/4 if data set size <= 32")
})
# Test that the cost shows three spikes due to the perplexity change
# but that the optimizer is able to reduce the cost after each spike
# does not change the output kernel precision as perplexity changes
test_that("multiscaling with equal perplexities is the same as single scale", {
  ssne_iris <- embed_prob(
    iris[, 1:4],
    method = ssne(beta = 0.02),
    opt = mize_back_nag_adapt(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perp(perplexity = 25, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(verbose = FALSE, report_every = 5),
    max_iter = 20,
    export = c("report")
  )

  ssne_iris_m1 <- embed_prob(
    iris[, 1:4],
    method = ssne(verbose = FALSE),
    opt = mize_back_nag_adapt(),
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
    method = ssne(verbose = FALSE),
    opt = mize_back_nag_adapt(),
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
    method = ssne(verbose = FALSE),
    opt = mize_grad_descent(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 10, verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             convergence_iter = 10,
                             verbose = FALSE),
    max_iter = 14,
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ms3$method$kernels, function(k) { k$beta }),
               c(1/150, 1/100, 1/50), info = "scaled kernels")

  expect_equal(ssne_iris_ms3$report$costs[,"norm"][1:15],
               c(0.9291,  0.9240,  0.9181,  0.9114,  0.9036, # perp 75
                 0.8935,  0.8830,  0.8710,  0.8572,  0.8414, # perp 50
                 0.7985,  0.7759,  0.7505,  0.7222,  0.6907),
               tolerance = 5e-4, scale = 1, label = "ms-SSNE norm costs")

})

test_that("ms SSNE with unit scaling", {
  ssne_iris_ums3 <- embed_prob(
    iris[, 1:4],
    method = ssne(verbose = FALSE),
    opt = mize_grad_descent(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 10,
                                    modify_kernel_fn = NULL,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             convergence_iter = 10,
                             verbose = FALSE, reltol = NULL),
    max_iter = 14,
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ums3$method$kernels, function(k) { k$beta }),
               c(1, 1, 1), info = "uniform kernels")

  expect_equal(ssne_iris_ums3$report$costs[,"norm"][1:15],
               c(0.9359,  0.7661,  0.6064,  0.4603,  0.3301, # perp 75
                 0.1016,  0.0735,  0.0700,  0.0559,  0.0549, # perp 50
                 0.1135,  0.1069,  0.1010,  0.0960,  0.0922  # perp 25
                 ),
               tolerance = 5e-4, scale = 1, label = "ms-SSNE unit prec")
})


test_that("Can apply multiple scales in one iteration", {
  ssne_iris_ums3_s0 <- embed_prob(
    iris[, 1:4],
    method = ssne(verbose = FALSE),
    opt = mize_bfgs(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 0,
                                    modify_kernel_fn = NULL,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             convergence_iter = 10,
                             verbose = FALSE),
    max_iter = 9,
    export = c("report", "method")
  )
  expect_equal(sapply(ssne_iris_ums3_s0$method$kernels, function(k) { k$beta }),
               c(1, 1, 1), info = "uniform kernels")

  expect_equal(ssne_iris_ums3_s0$report$costs[,"norm"],
               c(0.2961,  0.1581,  0.1175,  0.1002,  0.0935,
                 0.0893,  0.0891,  0.0891,  0.0890,  0.0890),
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE unit prec scale all at once")

  ssne_iris_ms3_s0 <- embed_prob(
    iris[, 1:4],
    method = ssne(verbose = FALSE),
    opt = mize_bfgs(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(75, 25, length.out = 3),
                                    num_scale_iters = 0,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 1,
                             verbose = FALSE),
    max_iter = 9,
    export = c("report", "method")
  )

  expect_equal(sapply(ssne_iris_ms3_s0$method$kernels, function(k) { k$beta }),
               c(1/150, 1/100, 1/50), info = "scaled kernels")

  expect_equal(ssne_iris_ms3_s0$report$costs[,"norm"],
               c(0.9180,  0.1627,  0.1012,  0.0858,  0.0773,
                 0.0725,  0.0711,  0.0705,  0.0702,  0.0701),
               tolerance = 5e-4, scale = 1,
               label = "ms-SSNE scaled prec scale all at once")
})

test_that("Can combine multiscaling with asymmetric weights", {
  ssne_iris_tms3_s10 <- embed_prob(
    iris[1:10, 1:4],
    method = ssne(verbose = FALSE),
    opt = mize_bfgs(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(8, 4, length.out = 3),
                                    num_scale_iters = 6,
                                    modify_kernel_fn =
                                      transfer_kernel_precisions,
                                    verbose = FALSE),
    init_out = out_from_PCA(verbose = FALSE),
    reporter = make_reporter(keep_costs = TRUE, report_every = 2,
                             convergence_iter = 6,
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
    method = ssne(verbose = FALSE),
    opt = mize_bfgs(),
    preprocess = make_preprocess(auto_scale = TRUE, verbose = FALSE),
    init_inp = inp_from_perps_multi(perplexities = seq(8, 4, length.out = 3),
                                    num_scale_iters = 0,
                                    modify_kernel_fn =
                                      transfer_kernel_precisions,
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

