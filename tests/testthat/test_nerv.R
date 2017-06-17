library(sneer)
context("NeRV")

expect_norm_costs <- function(method, expected_norm, info = "",
                              tolerance = 1e-5, scale = 1) {
  method$verbose <- FALSE
  result <- embed_prob(iris[, 1:4], method = method,
                        max_iter = 9,
                        init_inp = inp_from_perp(verbose = FALSE,
                                                 modify_kernel_fn =
                                                   transfer_kernel_precisions),
                        init_out = out_from_PCA(verbose = FALSE),
                        preprocess = make_preprocess(verbose = FALSE),
                        reporter = make_reporter(report_every = 1,
                                                 keep_costs = TRUE,
                                                 verbose = FALSE),
                        export = c("report"), verbose = FALSE,
                       opt = mize_bfgs())
  expect_equal(result$report$costs[, "norm"], expected_norm, info = info,
               tolerance = tolerance, scale = scale)
}

test_that("NeRV gives different results as lambda is varied", {
  expect_norm_costs(nerv(lambda = 0),
                    c(0.00922, 0.00771, 0.00742, 0.00741, 0.00732,
                      0.00722, 0.00717, 0.00710, 0.00703, 0.00701),
                    info = "nerv lambda = 0", tolerance = 1e-4)

  expect_norm_costs(nerv(lambda = 0.5),
                    c(0.0144, 0.0123, 0.0118, 0.0117, 0.0116,
                      0.0115, 0.0114, 0.0114, 0.0112, 0.0111),
                    info = "nerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(nerv(lambda = 1),
                    c(0.0623, 0.0550, 0.0527, 0.0518, 0.0513,
                      0.0508, 0.0504, 0.0501, 0.0497, 0.0492),
                    info = "nerv lambda = 1", tolerance = 1e-4)
})

test_that("SNeRV gives different results as lambda is varied", {
  expect_norm_costs(snerv(lambda = 0),
                    c(0.0117, 0.0091, 0.0086, 0.0083, 0.0082,
                      0.0081, 0.0080, 0.0080, 0.0079, 0.0078),
                    info = "snerv lambda = 0", tolerance = 1e-4)

  expect_norm_costs(snerv(lambda = 0.5),
                    c(0.0181, 0.0148, 0.0140, 0.0137, 0.0136,
                      0.0136, 0.0135, 0.0134, 0.0134, 0.0133),
                    info = "snerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(snerv(lambda = 1),
                    c(0.0673, 0.0568, 0.0538, 0.0528, 0.0523,
                      0.0518, 0.0513, 0.0510, 0.0508, 0.0506),
                    info = "snerv lambda = 1", tolerance = 1e-4)
})

test_that("HSNeRV gives different results as lambda and alpha is varied", {
  # alpha = 0 should be like snerv
  expect_norm_costs(hsnerv(lambda = 0, alpha = 0),
                    c(0.0117, 0.0091, 0.0086, 0.0083, 0.0082,
                      0.0081, 0.0080, 0.0080, 0.0079, 0.0078),
                    info = "hsnerv lambda = 0 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 0),
                    c(0.01808, 0.0148, 0.0140, 0.0137, 0.0136,
                      0.0136, 0.0135, 0.0134, 0.0134, 0.0133),
                    info = "hsnerv lambda = 0.5 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 0),
                    c(0.0673, 0.0568, 0.0538, 0.0528, 0.0523,
                      0.0518, 0.0513, 0.0510, 0.0508, 0.0506),
                    info = "hsnerv lambda = 1 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0, alpha = 1),
                    c(0.1327, 0.112, 0.0990, 0.0909, 0.0802, 0.0670,
                      0.0491, 0.0381, 0.0332, 0.0306),
                    info = "hsnerv lambda = 0 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 1),
                    c(0.1444, 0.1264, 0.1160, 0.1056, 0.0938,
                      0.0781, 0.0618, 0.0530, 0.0495, 0.0471),
                    info = "hsnerv lambda = 0.5 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 1),
                    c(0.2346, 0.206, 0.194, 0.183, 0.172,
                      0.160, 0.151, 0.146, 0.140, 0.137),
                    info = "hsnerv lambda = 1 alpha = 1", tolerance = 1e-3)
})
