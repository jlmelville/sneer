library(sneer)
context("NeRV")

expect_norm_costs <- function(method, expected_norm, info = "",
                              tolerance = 1e-5, scale = 1) {
  method$verbose <- FALSE
  result <- embed_prob(iris[, 1:4], method = method,
                        max_iter = 9,
                        init_inp = inp_from_perp(verbose = FALSE),
                        init_out = out_from_PCA(verbose = FALSE),
                        preprocess = make_preprocess(verbose = FALSE),
                        reporter = make_reporter(report_every = 1,
                                                 keep_costs = TRUE,
                                                 verbose = FALSE),
                        export = c("report"), verbose = FALSE,
                       opt = mize_back_nag())
  expect_equal(result$report$costs[, "norm"], expected_norm, info = info,
               tolerance = tolerance, scale = scale)
}

test_that("NeRV gives different results as lambda is varied", {
  expect_norm_costs(nerv(lambda = 0),
                    c(0.00922, 0.00771, 0.00742, 0.00741, 0.00732,
                      0.00723, 0.00717, 0.00714, 0.00711, 0.00708
                      ),
                    info = "nerv lambda = 0")

  expect_norm_costs(nerv(lambda = 0.5),
                    c(0.0144, 0.0123, 0.0120, 0.0119, 0.0118,
                      0.0117, 0.0116, 0.0116, 0.0115, 0.0115),
                    info = "nerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(nerv(lambda = 1),
                    c(0.0623, 0.0550, 0.0542, 0.0529, 0.0522,
                      0.0518, 0.0514, 0.0511, 0.0508, 0.0506),
                    info = "nerv lambda = 1", tolerance = 1e-4)
})

test_that("SNeRV gives different results as lambda is varied", {
  expect_norm_costs(snerv(lambda = 0),
                    c(0.01165, 0.00933, 0.00865, 0.00866, 0.00852,
                      0.00834, 0.00823, 0.00817, 0.00811, 0.00804),
                    info = "snerv lambda = 0")

  expect_norm_costs(snerv(lambda = 0.5),
                    c(0.0181, 0.0148, 0.0142, 0.0142, 0.0139,
                      0.0137, 0.0137, 0.0136, 0.0135, 0.0135),
                    info = "snerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(snerv(lambda = 1),
                    c(0.0673, 0.0569, 0.0552, 0.0539, 0.0531,
                      0.0528, 0.0524, 0.0516, 0.0516, 0.0514),
                    info = "snerv lambda = 1", tolerance = 1e-4)
})

test_that("HSNeRV gives different results as lambda and alpha is varied", {
  expect_norm_costs(hsnerv(lambda = 0, alpha = 0),
                    c(0.01165, 0.00933, 0.00865, 0.00866, 0.00852,
                      0.00834, 0.00823, 0.00817, 0.00811, 0.00804),
                    info = "hsnerv lambda = 0 alpha = 0")

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 0),
                    c(0.01808, 0.01485, 0.01416, 0.01416, 0.01393,
                      0.01374, 0.01366, 0.01361, 0.01355, 0.01348),
                    info = "hsnerv lambda = 0.5 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 0),
                    c(0.06728, 0.05685, 0.05516, 0.05393, 0.05312,
                      0.05279, 0.05235, 0.05194, 0.05164, 0.05135),
                    info = "hsnerv lambda = 1 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0, alpha = 1),
                    c(0.1327, 0.1252, 0.1173, 0.111, 0.106, 0.1018,
                      0.09825, 0.09491, 0.09171, 0.0887),
                    info = "hsnerv lambda = 0 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 1),
                    c(0.1444, 0.1371, 0.1298, 0.1241, 0.1194,
                      0.1152, 0.1114, 0.1078, 0.1044, 0.1012),
                    info = "hsnerv lambda = 0.5 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 1),
                    c(0.2346, 0.2204, 0.2079, 0.199, 0.192,
                      0.1857, 0.18, 0.1747, 0.1697, 0.1652),
                    info = "hsnerv lambda = 1 alpha = 1", tolerance = 1e-3)
})
