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
                    c(0.00922, 0.00771, 0.00735, 0.00724, 0.00718,
                      0.00715, 0.00712, 0.00709, 0.00707, 0.00704),
                    info = "nerv lambda = 0")

  expect_norm_costs(nerv(lambda = 0.5),
                    c(0.0144, 0.0123, 0.0118, 0.0117, 0.0116,
                      0.0116, 0.0115, 0.0115, 0.0114, 0.0114),
                    info = "nerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(nerv(lambda = 1),
                    c(0.0623, 0.0550, 0.0530, 0.0521, 0.0515,
                      0.0512, 0.0509, 0.0507, 0.0505, 0.0503),
                    info = "nerv lambda = 1", tolerance = 1e-4)
})

test_that("SNeRV gives different results as lambda is varied", {
  expect_norm_costs(snerv(lambda = 0),
                    c(0.01165, 0.00933, 0.00864, 0.00839, 0.00829,
                      0.00821, 0.00816, 0.00809, 0.00804, 0.00799),
                    info = "snerv lambda = 0")

  expect_norm_costs(snerv(lambda = 0.5),
                    c(0.0181, 0.0148, 0.0140, 0.0138, 0.0137,
                      0.0136, 0.0136, 0.0135, 0.0135, 0.0134),
                    info = "snerv lambda = 0.5", tolerance = 1e-4)

  expect_norm_costs(snerv(lambda = 1),
                    c(0.0673, 0.0568, 0.0542, 0.0531, 0.0525,
                      0.0522, 0.0519, 0.0516, 0.0514, 0.0511),
                    info = "snerv lambda = 1", tolerance = 1e-4)
})

test_that("HSNeRV gives different results as lambda and alpha is varied", {
  expect_norm_costs(hsnerv(lambda = 0, alpha = 0),
                    c(0.01165, 0.00933, 0.00864, 0.00839, 0.00829,
                      0.00822, 0.00816, 0.00809, 0.00804, 0.00799),
                    info = "hsnerv lambda = 0 alpha = 0")

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 0),
                    c(0.0181, 0.0148, 0.0140, 0.0138, 0.0137,
                      0.0136, 0.0136, 0.0135, 0.0135, 0.0134),
                    info = "hsnerv lambda = 0.5 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 0),
                    c(0.0673, 0.0568, 0.0542, 0.0531, 0.0525,
                      0.0522, 0.0519, 0.0516, 0.0514, 0.0511),
                    info = "hsnerv lambda = 1 alpha = 0", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0, alpha = 1),
                    c(0.1327, 0.1252, 0.1171, 0.1105, 0.1051,
                      0.1006, 0.0967, 0.0931, 0.0897, 0.0866),
                    info = "hsnerv lambda = 0 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 0.5, alpha = 1),
                    c(0.1444, 0.1371, 0.1297, 0.1236, 0.1185,
                      0.1140, 0.1099, 0.1062, 0.1027, 0.0994),
                    info = "hsnerv lambda = 0.5 alpha = 1", tolerance = 1e-4)

  expect_norm_costs(hsnerv(lambda = 1, alpha = 1),
                    c(0.235, 0.220, 0.208, 0.198, 0.190,
                      0.184, 0.179, 0.175, 0.171, 0.168),
                    info = "hsnerv lambda = 1 alpha = 1", tolerance = 1e-3)
})
